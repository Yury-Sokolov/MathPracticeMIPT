import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import matplotlib.pyplot as plt

from core.simulation import Simulation
from potential.potential import MesonExchangePotential
from utils.visualization import plot_training_loss, plot_force_profile_evolution


class Trainer:
    """
    Класс для обучения моделей UDE (Universal Differential Equation).
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        masses: torch.Tensor,
        optimizer: optim.Optimizer,
        loss_manager: Any,
        accel_scale: torch.Tensor,
        pos_mean: torch.Tensor,
        pos_std: torch.Tensor,
        scheduler: Optional[Any] = None,
        wandb_logger: Optional[Any] = None,
        true_potential: Optional[MesonExchangePotential] = None,
        model_save_path: str = "./models/ude_model.pt",
        config: Optional[Dict[str, Any]] = None,
        memory_efficient: bool = True,
        memory_cleanup_freq: int = 20
    ):
        """
        Инициализирует тренер.
        
        Args:
            model: Модель для обучения
            device: Устройство для обучения
            masses: Массы частиц
            optimizer: Оптимизатор
            loss_manager: Менеджер функций потерь
            accel_scale: Масштаб для ускорений
            pos_mean: Среднее значение позиций для нормализации
            pos_std: Стандартное отклонение позиций для нормализации
            scheduler: Планировщик скорости обучения
            wandb_logger: Логгер W&B
            true_potential: Истинный потенциал для сравнения (если есть)
            model_save_path: Путь для сохранения модели
            config: Конфигурация обучения
            memory_efficient: Использовать ли оптимизации памяти
            memory_cleanup_freq: Частота очистки кеша CUDA (в шагах)
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_manager = loss_manager
        self.scheduler = scheduler
        self.wandb_logger = wandb_logger
        self.true_potential = true_potential
        self.model_save_path = model_save_path
        self.config = config or {}
        
        self.sim_model = Simulation(
            potential=None, neural_network=model,
            t_end=1.0,
            device=device,
            dt_min=1e-16, tau_max=0.01, Imax=0.1,
            adaptive_dt=False
        )
        self.sim_model.nucleons['masses'] = masses
        
        self.accel_scale = accel_scale.to(device)
        self.pos_mean = pos_mean.to(device)
        self.pos_std = pos_std.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.force_profile_history = []
        self.epochs_history = []
        
        self.batch_size = self.config.get('batch_size', 128)
        self.epochs = self.config.get('epochs', 100)
        self.clip_grad = self.config.get('clip_grad', 1.0)
        self.symmetry_weight = self.config.get('symmetry_weight', 0.5)
        self.smoothness_weight = self.config.get('smoothness_weight', 0.1)
        self.patience = self.config.get('patience', 15)
        self.batch_accumulation_steps = self.config.get('batch_accumulation_steps', 4)
        self.r_cutoff = self.config.get('r_cutoff', 5.0)
        
        self.best_val_loss = float('inf')
        self.best_force_error = float('inf')
        self.patience_counter = 0
        self.zero_force_counter = 0
        
        self.memory_efficient = memory_efficient
        self.memory_cleanup_freq = memory_cleanup_freq
    
    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        potential_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Запускает процесс обучения.
        
        Args:
            train_data: Данные для обучения
            val_data: Данные для валидации
            potential_params: Параметры истинного потенциала (если есть)
            
        Returns:
            Статистика обучения
        """
        print(f"\nЗапуск обучения на {self.epochs} эпохах...")
        start_time = time.time()
        
        if potential_params and not self.config.get('skip_pretraining', False):
            self._pretrain_model(potential_params)
        
        normalized_positions_for_loss = train_data['normalized_positions_for_loss'].to(self.device)
        normalized_accel = train_data['normalized_accel'].to(self.device)
        
        num_steps_loss = normalized_accel.shape[0]
        batch_size = min(self.batch_size, num_steps_loss)
        num_batches = (num_steps_loss + batch_size - 1) // batch_size
        
        print(f"Всего шагов для обучения: {num_steps_loss}, Размер батча: {batch_size}, Батчей на эпоху: {num_batches}")
        
        for epoch in range(self.epochs):
            epoch_stats = self._train_epoch(
                normalized_positions_for_loss, 
                normalized_accel, 
                num_steps_loss,
                batch_size
            )
            
            val_loss = self._validate_epoch(val_data)
            self.val_losses.append(val_loss)
            
            force_stats = self._check_force_profile(potential_params)
            self.force_profile_history.append(force_stats['force_profile'])
            
            if epoch % max(1, self.epochs // 10) == 0:
                self.epochs_history.append(epoch)
            
            avg_epoch_loss = epoch_stats['loss']
            avg_epoch_mse = epoch_stats['mse_loss']
            avg_epoch_sym = epoch_stats['symmetry_loss']
            avg_epoch_smooth = epoch_stats['smoothness_loss']
            avg_epoch_mag = epoch_stats['force_magnitude_loss']
            
            self.train_losses.append(avg_epoch_loss)
            
            print(f"Эпоха {epoch+1}/{self.epochs} - Потеря: {avg_epoch_loss:.6f}, Валидация: {val_loss:.6f}, " 
                  f"MSE: {avg_epoch_mse:.6f}, Sym: {avg_epoch_sym:.6f}, Smooth: {avg_epoch_smooth:.6f}, Mag: {avg_epoch_mag:.6f}, " 
                  f"Ошибка силы: {force_stats['force_error']:.6f}, Макс. сила: {force_stats['max_force']:.6f}")
            
            if self.wandb_logger:
                self._log_wandb_epoch_metrics(epoch_stats, val_loss, force_stats)
            
            if force_stats['is_zero_like']:
                self.zero_force_counter += 1
                print(f"\nПредупреждение: Модель сходится к нулевому решению {self.zero_force_counter} эпох подряд! Макс. сила: {force_stats['max_force']:.6f}")
                
                if self.zero_force_counter >= 3:
                    self._handle_zero_force_problem()
            else:
                self.zero_force_counter = 0
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"Новая лучшая потеря на валидации: {self.best_val_loss:.6f}. Сохранение модели...")
                torch.save(self.model.state_dict(), self.model_save_path.replace('.pt', '_best_val.pt'))
                
                if force_stats['force_error'] < self.best_force_error:
                    self.best_force_error = force_stats['force_error']
                    print(f"Новая лучшая ошибка силы: {self.best_force_error:.6f}. Сохранение модели...")
                    torch.save(self.model.state_dict(), self.model_save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Раннее прекращение после {self.patience} эпох без улучшения")
                    break
            
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
                elif hasattr(self.scheduler, 'get_last_lr'):
                    self.scheduler.step(val_loss)
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Обучение завершено за {training_time:.2f} секунд!")
        
        print(f"Сохранение обученной модели в {self.model_save_path}...")
        torch.save(self.model.state_dict(), self.model_save_path)
        
        if os.path.exists(self.model_save_path.replace('.pt', '_best_val.pt')):
            print(f"Загрузка лучшей модели из {self.model_save_path.replace('.pt', '_best_val.pt')}...")
            self.model.load_state_dict(torch.load(self.model_save_path.replace('.pt', '_best_val.pt')))
        
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'force_profile_history': self.force_profile_history,
            'epochs_history': self.epochs_history,
            'best_val_loss': self.best_val_loss,
            'best_force_error': self.best_force_error,
            'training_time': training_time,
            'model': self.model
        }
        
        return results
        
    def _pretrain_model(self, potential_params: Dict[str, Any]) -> None:
        """
        Предварительное обучение модели для инициализации весов.
        
        Args:
            potential_params: Параметры истинного потенциала
        """
        print("Предварительная инициализация весов для соответствия истинной кривой потенциала...")
        
        r_cutoff = potential_params.get('r_cutoff', 5.0)
        
        test_dists_close = torch.linspace(0.2, 1.0, 30, device=self.device)
        test_dists_far = torch.linspace(1.0, r_cutoff, 30, device=self.device)
        test_dists = torch.cat([test_dists_close, test_dists_far[1:]])
        
        test_vectors = torch.zeros((len(test_dists), 3), device=self.device)
        test_vectors[:, 0] = test_dists
        
        g_att = potential_params['g_att']
        g_rep = potential_params['g_rep']
        m_pi = potential_params['m_pi']
        m_rho = potential_params['m_rho']
        
        true_potential_values = g_rep * torch.exp(-m_rho*test_dists) / test_dists - g_att * torch.exp(-m_pi*test_dists) / test_dists
        true_potential_values -= torch.min(true_potential_values)

        term1 = g_rep * torch.exp(-m_rho*test_dists) * (m_rho/test_dists + 1/(test_dists**2))
        term2 = g_att * torch.exp(-m_pi*test_dists) * (m_pi/test_dists + 1/(test_dists**2))
        true_force_magnitudes = term1 - term2
        
        true_forces = torch.zeros_like(test_vectors)
        true_forces[:, 0] = true_force_magnitudes
        
        init_optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        print("Предварительное обучение нейронной сети...")
        for pre_epoch in tqdm(range(3), desc="Предварительное обучение"):
            init_optimizer.zero_grad()
            
            with torch.enable_grad():
                test_vectors_clone = test_vectors.clone().requires_grad_(True)
                
                pred_potentials = self.model.compute_potential(test_vectors_clone)
                pred_min = torch.min(pred_potentials)
                pred_potentials = pred_potentials - pred_min
                
                max_true = torch.max(true_potential_values)
                max_pred = torch.max(pred_potentials.detach())
                
                if max_pred > 1e-6:
                    pred_scale = max_true / max_pred
                else:
                    pred_scale = 1.0
                    
                pred_potentials_scaled = pred_potentials * pred_scale
                pot_loss = F.mse_loss(pred_potentials_scaled, true_potential_values)

                total_potential_sum = torch.sum(pred_potentials_scaled)
                pred_forces = -torch.autograd.grad(
                    total_potential_sum, test_vectors_clone,
                    create_graph=True, retain_graph=True
                )[0]
                
                weights = 1.0 / (test_dists.detach() + 0.5)
                weights = weights / weights.sum() 
                force_diff = (pred_forces - true_forces) ** 2
                force_loss = torch.sum(weights.unsqueeze(1) * force_diff) 

                combined_loss = pot_loss + 5.0 * force_loss 
                combined_loss.backward()
                
                total_pot_loss = pot_loss.item()
                total_force_loss = force_loss.item()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            init_optimizer.step()
            
            print(f"  Предварительная эпоха {pre_epoch+1}: Потеря потенциала={total_pot_loss:.6f}, Потеря силы={total_force_loss:.6f}")
        
        with torch.no_grad():
            pred_potentials = self.model.compute_potential(test_vectors)
                
            pred_min = torch.min(pred_potentials)
            pred_potentials = pred_potentials - pred_min
            pred_scale = torch.max(true_potential_values) / torch.max(pred_potentials)
            pred_potentials_scaled = pred_potentials * pred_scale
            
            pot_error = F.mse_loss(pred_potentials_scaled, true_potential_values).item()
            print(f"Ошибка потенциала после предварительного обучения: {pot_error:.6f}")
        
        del init_optimizer, test_vectors_clone
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _check_force_profile(self, potential_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Проверяет профиль сил, генерируемый моделью.
        
        Args:
            potential_params: Параметры истинного потенциала для сравнения
            
        Returns:
            Статистика профиля сил
        """
        distances = torch.linspace(0.2, 4.0, 40, device=self.device)
        test_vectors = torch.zeros((len(distances), 3), device=self.device)
        test_vectors[:, 0] = distances
        
        test_vectors_grad = test_vectors.clone().requires_grad_(True)
        
        potentials = self.model.compute_potential(test_vectors_grad)
        total_potential = potentials.sum()
        forces = -torch.autograd.grad(
            total_potential, test_vectors_grad, 
            create_graph=False,
            retain_graph=False
        )[0]
        force_x = forces[:, 0]
        
        force_error = 0.0
        true_force_x = None
        if potential_params:
            with torch.no_grad():
                g_att = potential_params['g_att']
                g_rep = potential_params['g_rep']
                m_pi = potential_params['m_pi']
                m_rho = potential_params['m_rho']
                term1 = g_rep * torch.exp(-m_rho*distances) * (m_rho/distances + 1/(distances**2))
                term2 = g_att * torch.exp(-m_pi*distances) * (m_pi/distances + 1/(distances**2))
                true_force_x = term1 - term2
                
                force_error = torch.mean((force_x - true_force_x) ** 2).item()
        
        with torch.no_grad():
            max_force = torch.max(torch.abs(force_x)).item()
            mean_force = torch.mean(torch.abs(force_x)).item()
            is_zero_like = max_force < 0.1
            
            potentials_np = potentials.cpu().numpy()
            force_x_np = force_x.cpu().numpy()
        
        return {
            'max_force': max_force,
            'mean_force': mean_force,
            'is_zero_like': is_zero_like,
            'force_profile': force_x_np,
            'force_error': force_error,
            'potentials': potentials_np,
            'true_force_x': true_force_x.cpu().numpy() if true_force_x is not None else None
        } 
    
    def _train_epoch(
        self,
        normalized_positions_for_loss: torch.Tensor,
        normalized_accel: torch.Tensor,
        num_steps_loss: int,
        batch_size: int
    ) -> Dict[str, float]:
        """
        Выполняет одну эпоху обучения.
        
        Args:
            normalized_positions_for_loss: Нормализованные позиции для вычисления потерь
            normalized_accel: Нормализованные целевые ускорения
            num_steps_loss: Общее количество шагов для вычисления потерь
            batch_size: Размер батча
            
        Returns:
            Статистика эпохи (потери)
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_symmetry_loss = 0.0
        epoch_force_magnitude_loss = 0.0
        epoch_smoothness_loss = 0.0
        
        permuted_indices = torch.randperm(num_steps_loss).tolist()
        
        effective_batch_size = max(1, batch_size // (4 if hasattr(self.model, 'kan') else 1))
        accumulation_steps = self.batch_accumulation_steps
        num_batches = (num_steps_loss + effective_batch_size - 1) // effective_batch_size
        
        batch_pbar = tqdm(range(0, num_batches, accumulation_steps), 
                         desc=f"Обучение", leave=False)
        
        for i in batch_pbar:
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            batch_loss = 0.0
            batch_mse_loss = 0.0
            batch_symmetry_loss = 0.0
            batch_force_magnitude_loss = 0.0
            batch_smoothness_loss = 0.0
            
            actual_accumulation_steps = min(accumulation_steps, num_batches - i)
            
            for acc_step in range(actual_accumulation_steps):
                batch_idx = i + acc_step
                start_idx = batch_idx * effective_batch_size
                end_idx = min((batch_idx + 1) * effective_batch_size, num_steps_loss)
                batch_indices = permuted_indices[start_idx:end_idx]
                current_batch_actual_size = len(batch_indices)
                
                if current_batch_actual_size == 0:
                    continue
                
                current_positions_batch = normalized_positions_for_loss[batch_indices].to(self.device)
                current_target_accel_batch = normalized_accel[batch_indices].to(self.device)
                
                with torch.set_grad_enabled(True):
                    unnorm_positions_batch = current_positions_batch * self.pos_std + self.pos_mean
                    
                    predicted_accels_batch, _, _ = self.sim_model.compute_forces(unnorm_positions_batch)
                    
                    predicted_accels_norm_batch = predicted_accels_batch / self.accel_scale
                    
                    n_particles = current_positions_batch.shape[1]
                    batch_actual_size = current_positions_batch.shape[0]
                    
                    mse_loss_batch = F.huber_loss(
                        predicted_accels_norm_batch.reshape(-1, 3),
                        current_target_accel_batch.reshape(-1, 3),
                        delta=1.0,
                        reduction='mean'
                    )
                    
                    symmetry_loss_tensor = torch.tensor(0.0, device=self.device)
                    smoothness_loss_tensor = torch.tensor(0.0, device=self.device)
                    magnitude_loss_tensor = torch.tensor(0.0, device=self.device)
                    
                    if n_particles > 1:
                        num_pairs_per_instance = min(
                            3 if hasattr(self.model, 'kan') else 5, 
                            n_particles * (n_particles - 1) // 2
                        ) 
                        
                        pair_indices_batch = torch.randint(
                            0, n_particles, 
                            (batch_actual_size, num_pairs_per_instance * 2), 
                            device=self.device
                        )
                        
                        all_rel_pos_ij = []
                        all_true_potentials = []
                        
                        for b_idx in range(batch_actual_size):
                            instance_positions = current_positions_batch[b_idx]
                            instance_pairs = pair_indices_batch[b_idx]
                            
                            for p_idx in range(0, len(instance_pairs), 2):
                                if p_idx + 1 >= len(instance_pairs):
                                    break
                                i_idx, j_idx = instance_pairs[p_idx], instance_pairs[p_idx + 1]
                                if i_idx == j_idx: 
                                    continue
                                
                                pos_i = instance_positions[i_idx]
                                pos_j = instance_positions[j_idx]
                                rel_pos_ij = pos_i - pos_j
                                r_norm = torch.norm(rel_pos_ij)
                                
                                if r_norm > 1e-6:
                                    all_rel_pos_ij.append(rel_pos_ij.unsqueeze(0))
                                    
                                    if self.true_potential and hasattr(self.true_potential, 'g_att'):
                                        g_att = self.true_potential.g_att
                                        g_rep = self.true_potential.g_rep
                                        m_pi = self.true_potential.m_pi
                                        m_rho = self.true_potential.m_rho
                                        
                                        if r_norm > 0.2 and r_norm < self.r_cutoff:
                                            true_pot = (
                                                g_rep * torch.exp(-m_rho*r_norm) / r_norm - 
                                                g_att * torch.exp(-m_pi*r_norm) / r_norm
                                            )
                                            all_true_potentials.append(true_pot.unsqueeze(0))
                                        else:
                                            all_true_potentials.append(torch.tensor([torch.nan], device=self.device))
                        
                        if len(all_rel_pos_ij) > 0:
                            all_rel_pos_ij_tensor = torch.cat(all_rel_pos_ij, dim=0).requires_grad_(True)
                            
                            if hasattr(self.loss_manager, 'symmetry_loss'):
                                symmetry_loss_tensor = self.loss_manager.symmetry_loss(self.model, all_rel_pos_ij_tensor)
                            
                            if hasattr(self.loss_manager, 'smoothness_loss'):
                                smoothness_loss_tensor = self.loss_manager.smoothness_loss(self.model, all_rel_pos_ij_tensor)
                            
                            if len(all_true_potentials) > 0:
                                all_true_potentials_tensor = torch.cat(all_true_potentials, dim=0)
                                valid_pot_indices = ~torch.isnan(all_true_potentials_tensor)
                                
                                if torch.any(valid_pot_indices):
                                    valid_rel_pos = all_rel_pos_ij_tensor[valid_pot_indices]
                                    valid_true_pots = all_true_potentials_tensor[valid_pot_indices]
                                    
                                    if valid_rel_pos.shape[0] > 0:
                                        pred_pot_magnitude = self.model.compute_potential(valid_rel_pos)
                                        magnitude_loss_tensor = F.mse_loss(pred_pot_magnitude, valid_true_pots)
                    
                    current_epoch = len(self.train_losses)
                    
                    if current_epoch < self.epochs // 10:
                        mse_weight = 1.0
                        symmetry_weight = self.symmetry_weight * 0.2
                        smoothness_weight = self.smoothness_weight * 0.2
                        force_magnitude_weight = 0.2
                    elif current_epoch > self.epochs * 0.8:
                        mse_weight = 0.8
                        symmetry_weight = self.symmetry_weight * 1.5
                        smoothness_weight = self.smoothness_weight * 1.5
                        force_magnitude_weight = 1.5
                    else:
                        mse_weight = 1.0
                        symmetry_weight = self.symmetry_weight
                        smoothness_weight = self.smoothness_weight
                        force_magnitude_weight = 1.0
                    
                    if self.zero_force_counter > 1:
                        force_magnitude_weight *= 3.0
                    
                    if hasattr(self.model, 'kan'):
                        symmetry_weight *= 0.3
                        smoothness_weight *= 0.3
                    
                    combined_loss_batch = (
                        mse_weight * mse_loss_batch + 
                        symmetry_weight * symmetry_loss_tensor + 
                        smoothness_weight * smoothness_loss_tensor + 
                        force_magnitude_weight * magnitude_loss_tensor
                    )
                    
                    accumulation_scale = 1.0 / actual_accumulation_steps
                    scaled_loss = combined_loss_batch * accumulation_scale
                    scaled_loss.backward()
                    
                    batch_mse_loss += mse_loss_batch.item() * accumulation_scale
                    batch_symmetry_loss += symmetry_loss_tensor.item() * accumulation_scale
                    batch_smoothness_loss += smoothness_loss_tensor.item() * accumulation_scale
                    batch_force_magnitude_loss += magnitude_loss_tensor.item() * accumulation_scale
                    batch_loss += combined_loss_batch.item() * accumulation_scale
                
                del current_positions_batch, current_target_accel_batch, predicted_accels_batch, predicted_accels_norm_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += batch_loss
            epoch_mse_loss += batch_mse_loss
            epoch_symmetry_loss += batch_symmetry_loss
            epoch_smoothness_loss += batch_smoothness_loss
            epoch_force_magnitude_loss += batch_force_magnitude_loss
            
            avg_batch_loss = batch_loss
            avg_batch_mse = batch_mse_loss
            avg_batch_sym = batch_symmetry_loss
            avg_batch_smooth = batch_smoothness_loss
            avg_batch_mag = batch_force_magnitude_loss
            current_lr = self.optimizer.param_groups[0]['lr']
            
            batch_pbar.set_postfix({
                "Loss": f"{avg_batch_loss:.4e}",
                "MSE": f"{avg_batch_mse:.4e}",
                "Sym": f"{avg_batch_sym:.4e}",
                "Smooth": f"{avg_batch_smooth:.4e}",
                "Mag": f"{avg_batch_mag:.4e}",
                "LR": f"{current_lr:.3e}",
                "Mem": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
            })
        
        batch_pbar.close()
        
        epoch_stats = {
            'loss': epoch_loss / num_batches,
            'mse_loss': epoch_mse_loss / num_batches,
            'symmetry_loss': epoch_symmetry_loss / num_batches,
            'smoothness_loss': epoch_smoothness_loss / num_batches,
            'force_magnitude_loss': epoch_force_magnitude_loss / num_batches,
        }
        
        return epoch_stats
    
    def _validate_epoch(self, val_data: Dict[str, torch.Tensor]) -> float:
        """
        Валидирует модель на валидационной выборке.
        
        Args:
            val_data: Валидационные данные
            
        Returns:
            Средняя потеря на валидационной выборке
        """
        self.model.eval()
        val_loss = 0.0
        val_batches = 0
        
        normalized_positions_for_loss = val_data.get('normalized_positions_for_loss')
        normalized_accel = val_data.get('normalized_accel')
        
        if normalized_positions_for_loss is None or normalized_accel is None:
            print("Предупреждение: Не найдены данные для валидации")
            return 0.0
        
        val_indices = val_data.get('val_indices', 
                                   list(range(len(normalized_positions_for_loss))))
        
        val_batch_size = min(16, len(val_indices)) if hasattr(self.model, 'kan') else len(val_indices)
        
        for i in range(0, len(val_indices), val_batch_size):
            batch_val_indices = val_indices[i:i+val_batch_size]
            val_loss_batch = 0.0
            
            for val_idx in batch_val_indices:
                if val_idx >= len(normalized_positions_for_loss):
                    continue
                
                val_pos = normalized_positions_for_loss[val_idx].to(self.device)
                val_target_accel = normalized_accel[val_idx].to(self.device)
                
                val_pred_accels, _, _ = self.sim_model.compute_forces(
                    val_pos * self.pos_std + self.pos_mean
                )
                
                val_pred_accels_norm = val_pred_accels / self.accel_scale
                
                val_loss_step = F.huber_loss(
                    val_pred_accels_norm,
                    val_target_accel,
                    delta=1.0
                )
                
                val_loss_batch += val_loss_step.item()
                val_batches += 1
                
                del val_pos, val_target_accel, val_pred_accels, val_pred_accels_norm
            
            val_loss += val_loss_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return val_loss / max(1, val_batches)
    
    def _handle_zero_force_problem(self) -> None:
        """
        Обрабатывает проблему схождения к нулевому решению.
        Пытается загрузить лучшую модель и реинициализировать веса или оптимизатор.
        """
        print("Применяю корректирующие действия для нулевого решения...")
        
        best_model_path = self.model_save_path.replace('.pt', '_best_val.pt')
        if os.path.exists(best_model_path):
            print("Загружаю лучшую модель и применяю реинициализацию...")
            self.model.load_state_dict(torch.load(best_model_path))
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if 'output_layer' in name and 'weight' in name:
                        param.data *= 10.0
                    elif 'scaling_factor' in name:
                        param.data *= 10.0
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001) * 10.0,
                weight_decay=self.config.get('weight_decay', 1e-5) * 0.1,
                betas=(0.9, 0.999)
            )
        else:
            print("Лучшая модель не найдена. Реинициализирую веса...")
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if 'output_layer' in name and 'weight' in name:
                        param.data *= 10.0
                    elif 'scaling_factor' in name:
                        param.data *= 10.0
            
            self.zero_force_counter = 0
    
    def _log_wandb_epoch_metrics(
        self, 
        epoch_stats: Dict[str, float], 
        val_loss: float, 
        force_stats: Dict[str, Any]
    ) -> None:
        """
        Логирует метрики эпохи в Weights & Biases.
        
        Args:
            epoch_stats: Статистика эпохи (потери)
            val_loss: Потеря на валидационной выборке
            force_stats: Статистика профиля сил
        """
        if not self.wandb_logger:
            return
        
        self.wandb_logger.log({
            "epoch/train_loss": epoch_stats['loss'],
            "epoch/val_loss": val_loss,
            "epoch/mse_loss": epoch_stats['mse_loss'],
            "epoch/symmetry_loss": epoch_stats['symmetry_loss'],
            "epoch/smoothness_loss": epoch_stats['smoothness_loss'],
            "epoch/force_magnitude_loss": epoch_stats['force_magnitude_loss'],
            "epoch/force_error": force_stats['force_error'],
            "epoch/max_force": force_stats['max_force'],
            "epoch/mean_force": force_stats['mean_force'],
            "epoch/is_zero_like": int(force_stats['is_zero_like']),
            "epoch/learning_rate": self.optimizer.param_groups[0]['lr'],
        })
        
        if force_stats.get('true_force_x') is not None:
            distances = np.linspace(0.2, 4.0, 40)
            self.wandb_logger.log({
                "epoch/force_profile": self.wandb_logger.plot.line_series(
                    xs=distances,
                    ys=[force_stats['force_profile'], force_stats['true_force_x']],
                    keys=["Predicted", "True"],
                    title="Force Profile",
                    xname="Distance"
                ),
                "epoch/potential_profile": self.wandb_logger.plot.line_series(
                    xs=distances,
                    ys=[force_stats['potentials']],
                    keys=["Potential"],
                    title="Potential Profile",
                    xname="Distance"
                )
            }) 