import os
import yaml
import argparse
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
import sys


class Config:
    """
    Класс для управления конфигурацией проекта.
    Загружает настройки из YAML-файла и позволяет переопределять их аргументами командной строки.
    """

    def __init__(self, config_path: str):
        """
        Инициализирует конфигурацию из файла.
        
        Args:
            config_path: Путь к YAML-файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_yaml(config_path)
    
    @staticmethod
    def _load_yaml(file_path: str) -> Dict[str, Any]:
        """Загружает YAML-файл."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Получает значение по пути ключей в конфигурации.
        
        Args:
            key_path: Путь ключей, разделенных точкой (например, "model.hidden_dim")
            default: Значение по умолчанию, если ключ не найден
            
        Returns:
            Значение из конфигурации
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Устанавливает значение по пути ключей в конфигурации.
        
        Args:
            key_path: Путь ключей, разделенных точкой
            value: Новое значение
        """
        keys = key_path.split('.')
        config = self.config
        
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Обновляет конфигурацию значениями из аргументов командной строки.
        
        Args:
            args: Аргументы, полученные из argparse
        """
        # Преобразуем объект Namespace в словарь
        args_dict = vars(args)
        
        # Обновляем конфигурацию значениями из аргументов
        for key, value in args_dict.items():
            if value is not None:  # Обновляем только явно указанные аргументы
                # Находим соответствующий ключ в конфигурации
                for section in self.config:
                    if isinstance(self.config[section], dict) and key in self.config[section]:
                        self.config[section][key] = value
                        break
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Сохраняет текущую конфигурацию в YAML-файл.
        
        Args:
            output_path: Путь для сохранения. Если None, перезаписывает исходный файл.
        """
        save_path = output_path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    @classmethod
    def add_common_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        Добавляет общие аргументы в парсер аргументов.
        
        Args:
            parser: Парсер аргументов argparse
        """
        parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                           help='Path to configuration file')
        parser.add_argument('--mode', type=str, required=True,
                           choices=['generate_data', 'train_ude', 'analyze_results', 'run_simulation'],
                           help='Mode of operation')
    
    @classmethod
    def add_data_generation_args(cls, parser: argparse.ArgumentParser) -> None:
        """Добавляет аргументы для режима генерации данных."""
        group = parser.add_argument_group('Data Generation')
        group.add_argument('--data_file', type=str, help='Path to save generated data')
        group.add_argument('--num_collisions', type=int, help='Number of collision simulations')
        group.add_argument('--noise_level', type=float, help='Level of Gaussian noise to add')
        group.add_argument('--n_particles', type=int, help='Number of particles in simulation')
        group.add_argument('--max_impact_parameter', type=float, 
                           help='Maximum impact parameter for collisions')
    
    @classmethod
    def add_training_args(cls, parser: argparse.ArgumentParser) -> None:
        """Добавляет аргументы для режима обучения."""
        group = parser.add_argument_group('Training')
        group.add_argument('--model_save_path', type=str, help='Path to save the trained model')
        group.add_argument('--data_file', type=str, help='Path to load training data')
        group.add_argument('--learning_rate', type=float, help='Learning rate')
        group.add_argument('--epochs', type=int, help='Number of training epochs')
        group.add_argument('--batch_size', type=int, help='Batch size')
        group.add_argument('--nn_hidden_dim', type=int, help='Hidden dimension of neural network')
        group.add_argument('--model_type', type=str, choices=['potential_nn', 'kan'],
                           help='Neural network model type')
    
    @classmethod
    def add_evaluation_args(cls, parser: argparse.ArgumentParser) -> None:
        """Добавляет аргументы для режима анализа/оценки."""
        group = parser.add_argument_group('Evaluation')
        group.add_argument('--model_load_path', type=str, help='Path to load the model for analysis')
        group.add_argument('--data_file', type=str, help='Path to load data for analysis')
        group.add_argument('--analysis_output_dir', type=str, help='Directory to save analysis outputs')
    
    @classmethod
    def add_wandb_args(cls, parser: argparse.ArgumentParser) -> None:
        """Добавляет аргументы для W&B."""
        group = parser.add_argument_group('Weights & Biases')
        group.add_argument('--use_wandb', action='store_true', help='Use W&B for logging')
        group.add_argument('--wandb_project', type=str, help='W&B project name')
        group.add_argument('--wandb_entity', type=str, help='W&B entity name')
        group.add_argument('--wandb_run_name', type=str, help='Custom name for W&B run')
        group.add_argument('--wandb_tags', type=str, nargs='+', help='Tags for W&B run')


def create_arg_parser(mode: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Создает парсер аргументов в зависимости от режима работы.
    
    Args:
        mode: Режим работы (generate_data, train_ude, analyze_results)
        
    Returns:
        Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(description='Universal Differential Equation (UDE) for nucleon simulation')
    
    Config.add_common_args(parser)
    
    if mode == 'generate_data' or mode is None:
        Config.add_data_generation_args(parser)
    
    if mode == 'train_ude' or mode is None:
        Config.add_training_args(parser)
    
    if mode == 'analyze_results' or mode is None:
        Config.add_evaluation_args(parser)
    
    Config.add_wandb_args(parser)
    
    return parser


def load_config(args: argparse.Namespace) -> Config:
    """
    Загружает конфигурацию и обновляет ее аргументами командной строки.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Объект конфигурации
    """
    config = Config(args.config)
    config.update_from_args(args)
    return config 


def validate_config(config: DictConfig, mode: str) -> None:
    """
    Валидирует конфигурацию для заданного режима работы.
    
    Args:
        config: Конфигурация для валидации
        mode: Режим работы приложения
        
    Raises:
        ValueError: Если конфигурация не содержит обязательные параметры для режима
    """
    if mode == 'generate_data':
        validate_config_for_data_generation(config)
    elif mode == 'train_ude':
        validate_config_for_training(config)
    elif mode == 'analyze_results':
        validate_config_for_analysis(config)
    elif mode == 'run_simulation':
        validate_config_for_simulation(config)


def validate_config_for_data_generation(config: DictConfig) -> None:
    """
    Валидирует конфигурацию для режима генерации данных.
    
    Args:
        config: Конфигурация для валидации
        
    Raises:
        ValueError: Если отсутствуют обязательные параметры
    """
    required_params = [
        'data.output_file',
        'data.n_simulations',
        'simulation.n_particles',
        'simulation.max_impact_parameter',
        'simulation.relative_velocity',
    ]
    
    missing_params = [param for param in required_params if not get_nested_value(config, param)]
    
    if missing_params:
        raise ValueError(f"Отсутствуют обязательные параметры для генерации данных: {', '.join(missing_params)}")
    
    
def validate_config_for_training(config: DictConfig) -> None:
    """
    Валидирует конфигурацию для режима обучения UDE.
    
    Args:
        config: Конфигурация для валидации
        
    Raises:
        ValueError: Если отсутствуют обязательные параметры
    """
    required_params = [
        'data.train_file',
        'model.type',
        'training.epochs',
        'training.batch_size',
        'training.learning_rate',
        'training.model_save_path',
    ]
    
    missing_params = [param for param in required_params if not get_nested_value(config, param)]
    
    if missing_params:
        raise ValueError(f"Отсутствуют обязательные параметры для обучения: {', '.join(missing_params)}")
    
    
def validate_config_for_analysis(config: DictConfig) -> None:
    """
    Валидирует конфигурацию для режима анализа результатов.
    
    Args:
        config: Конфигурация для валидации
        
    Raises:
        ValueError: Если отсутствуют обязательные параметры
    """
    required_params = [
        'evaluation.model_load_path',
        'evaluation.data_file',
        'evaluation.analysis_output_dir',
    ]
    
    missing_params = [param for param in required_params if not get_nested_value(config, param)]
    
    if missing_params:
        raise ValueError(f"Отсутствуют обязательные параметры для анализа: {', '.join(missing_params)}")
    
    
def validate_config_for_simulation(config: DictConfig) -> None:
    """
    Валидирует конфигурацию для режима запуска симуляции.
    
    Args:
        config: Конфигурация для валидации
        
    Raises:
        ValueError: Если отсутствуют обязательные параметры
    """
    required_params = [
        'simulation.t_end',
        'simulation.n_particles',
    ]
    
    missing_params = [param for param in required_params if not get_nested_value(config, param)]
    
    if missing_params:
        raise ValueError(f"Отсутствуют обязательные параметры для симуляции: {', '.join(missing_params)}")
    
    # Если используется обученная модель, проверяем наличие файла модели
    if get_nested_value(config, 'evaluation.model_load_path'):
        model_path = get_nested_value(config, 'evaluation.model_load_path')
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели не найден: {model_path}")


def get_nested_value(config: DictConfig, key_path: str) -> Any:
    """
    Получает значение из вложенной конфигурации по пути к ключу.
    
    Args:
        config: Конфигурация OmegaConf
        key_path: Путь к ключу, разделенный точками (например, 'model.hidden_dim')
        
    Returns:
        Any: Значение по указанному пути или None, если путь не существует
    """
    parts = key_path.split('.')
    current = config
    
    for part in parts:
        if part not in current:
            return None
        current = current[part]
    
    return current 