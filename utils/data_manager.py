import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split


class DataManager:
    """
    Класс для управления данными в задаче UDE.
    Отвечает за загрузку, нормализацию, расчет производных и подготовку данных для обучения.
    """
    
    def __init__(self, device: torch.device):
        """
        Инициализирует менеджер данных.
        
        Args:
            device: Устройство для вычислений (CPU/GPU)
        """
        self.device = device
        self.data = {}
        self.normalization_stats = {}
    
    def load_data(self, data_path: str) -> Dict[str, Any]:
        """
        Загружает данные из файла.
        
        Args:
            data_path: Путь к файлу с данными
            
        Returns:
            Словарь с загруженными данными
        """
        print(f"Загрузка данных из {data_path}...")
        self.data = torch.load(data_path, map_location='cpu')
        return self.data
    
    def prepare_training_data(self, test_size: float = 0.1, random_state: int = 42) -> Dict[str, Any]:
        """
        Подготавливает данные для обучения: нормализация, вычисление ускорений, разделение.
        
        Args:
            test_size: Доля данных для валидации
            random_state: Случайное зерно для воспроизводимости
            
        Returns:
            Словарь с подготовленными данными для обучения
        """
        if not self.data:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Подготовка данных для обучения...")
        
        # Получаем данные
        noisy_positions = self.data['noisy_positions']
        noisy_velocities = self.data['noisy_velocities']
        true_positions = self.data['true_positions']
        true_velocities = self.data['true_velocities']
        times = self.data['times']
        masses = self.data['masses'].to(self.device)
        
        # Определяем dt для расчета ускорений
        if len(times) > 1:
            dt = (times[1] - times[0]).item()
        else:
            dt = 0.001
            print("Предупреждение: Невозможно определить dt. Используется значение по умолчанию 0.001")
        
        # Разделение на обучающую и валидационную выборки
        n_samples = noisy_positions.shape[0]
        train_indices, val_indices = train_test_split(
            np.arange(n_samples), 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Обучение на {len(train_indices)} образцах, валидация на {len(val_indices)} образцах")
        
        # Вычисление нормализационных статистик
        print("Вычисление статистик для нормализации...")
        pos_mean = torch.mean(noisy_positions[train_indices], dim=(0, 1))
        pos_std = torch.std(noisy_positions[train_indices], dim=(0, 1))
        vel_mean = torch.mean(noisy_velocities[train_indices], dim=(0, 1))
        vel_std = torch.std(noisy_velocities[train_indices], dim=(0, 1))
        
        # Предотвращение деления на ноль
        pos_std = torch.max(pos_std, torch.ones_like(pos_std) * 1e-8)
        vel_std = torch.max(vel_std, torch.ones_like(vel_std) * 1e-8)
        
        # Сохранение статистик
        self.normalization_stats = {
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'vel_mean': vel_mean,
            'vel_std': vel_std,
            'dt': dt
        }
        
        # Нормализация позиций
        normalized_positions = (noisy_positions - pos_mean) / pos_std
        normalized_positions_for_loss = normalized_positions[:-1]
        
        # Вычисление ускорений (цель)
        print("Вычисление целевых ускорений...")
        chunk_size = min(1000, noisy_velocities.shape[0] - 1)
        num_chunks = (noisy_velocities.shape[0] - 1 + chunk_size - 1) // chunk_size
        
        normalized_accel_list = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, noisy_velocities.shape[0] - 1)
            
            norm_vel_chunk_1 = (noisy_velocities[start_idx:end_idx] - vel_mean) / vel_std
            norm_vel_chunk_2 = (noisy_velocities[start_idx+1:end_idx+1] - vel_mean) / vel_std
            
            accel_chunk = (norm_vel_chunk_2 - norm_vel_chunk_1) / dt
            normalized_accel_list.append(accel_chunk)
            
            del norm_vel_chunk_1, norm_vel_chunk_2, accel_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        normalized_accel = torch.cat(normalized_accel_list, dim=0)
        del normalized_accel_list
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Масштаб для ускорений
        accel_scale = vel_std / (dt * pos_std)
        
        # Статистика ускорений
        accel_magnitudes = torch.norm(normalized_accel, dim=-1)
        max_accel = torch.max(accel_magnitudes).item()
        mean_accel = torch.mean(accel_magnitudes).item()
        print(f"Статистика целевых ускорений: Max={max_accel:.4f}, Mean={mean_accel:.4f}")
        
        # Подготовка результата
        result = {
            # Индексы для разделения данных
            'train_indices': train_indices,
            'val_indices': val_indices,
            
            # Нормализованные данные
            'normalized_positions': normalized_positions,
            'normalized_positions_for_loss': normalized_positions_for_loss,
            'normalized_accel': normalized_accel,
            
            # Исходные данные
            'times': times,
            'masses': masses,
            'true_positions': true_positions,
            'true_velocities': true_velocities,
            'noisy_positions': noisy_positions,
            'noisy_velocities': noisy_velocities,
            
            # Статистики и параметры
            'normalization_stats': self.normalization_stats,
            'accel_scale': accel_scale,
            'dt': dt,
        }
        
        return result
    
    def denormalize_positions(self, normalized_positions: torch.Tensor) -> torch.Tensor:
        """
        Денормализует позиции.
        
        Args:
            normalized_positions: Нормализованные позиции
            
        Returns:
            Денормализованные позиции
        """
        return normalized_positions * self.normalization_stats['pos_std'] + self.normalization_stats['pos_mean']
    
    def denormalize_velocities(self, normalized_velocities: torch.Tensor) -> torch.Tensor:
        """
        Денормализует скорости.
        
        Args:
            normalized_velocities: Нормализованные скорости
            
        Returns:
            Денормализованные скорости
        """
        return normalized_velocities * self.normalization_stats['vel_std'] + self.normalization_stats['vel_mean']
    
    def create_batch(self, indices: List[int], to_device: bool = True) -> Dict[str, torch.Tensor]:
        """
        Создает батч данных по указанным индексам.
        
        Args:
            indices: Индексы для выборки
            to_device: Переносить ли тензоры на устройство
            
        Returns:
            Словарь с данными батча
        """
        device = self.device if to_device else torch.device('cpu')
        
        positions = self.data['normalized_positions_for_loss'][indices].to(device)
        targets = self.data['normalized_accel'][indices].to(device)
        
        return {
            'positions': positions,
            'targets': targets
        }
    
    @staticmethod
    def save_data(data: Dict[str, Any], output_path: str) -> None:
        """
        Сохраняет данные в файл.
        
        Args:
            data: Данные для сохранения
            output_path: Путь для сохранения
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Сохранение данных в {output_path}...")
        
        # Заменяем device на строку для корректного сохранения
        for key, value in data.items():
            if isinstance(value, dict) and 'device' in value:
                value['device'] = str(value['device'])
        
        torch.save(data, output_path)
        print(f"Данные сохранены в {output_path}")


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Dataset для работы с траекторными данными.
    """
    
    def __init__(
        self, 
        positions: torch.Tensor, 
        accelerations: torch.Tensor,
        indices: Optional[List[int]] = None
    ):
        """
        Инициализирует dataset.
        
        Args:
            positions: Тензор позиций [n_steps, n_particles, 3]
            accelerations: Тензор ускорений [n_steps, n_particles, 3]
            indices: Индексы для выборки. Если None, используются все данные.
        """
        self.positions = positions
        self.accelerations = accelerations
        self.indices = indices if indices is not None else list(range(len(positions)))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        return self.positions[real_idx], self.accelerations[real_idx] 