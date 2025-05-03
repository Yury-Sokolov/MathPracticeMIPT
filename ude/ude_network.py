import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Any
try:
    import kan
except ImportError:
    print("Warning: pykan not installed. KAN models will not be available.")
    kan = None

class ScaleSILU(nn.Module):
    """Scaled SiLU activation for better gradient flow"""
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return F.silu(x) * self.scale

class ResidualBlock(nn.Module):
    """
    Блок остаточной связи для глубоких нейронных сетей.
    Предотвращает проблему исчезающего градиента.
    """
    
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        """
        Инициализация блока остаточной связи.
        
        Args:
            dim: Размерность входа/выхода
            dropout_rate: Вероятность отключения нейронов для регуляризации
        """
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.activation = ScaleSILU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через блок остаточной связи.
        
        Args:
            x: Входной тензор [batch_size, dim]
            
        Returns:
            Выходной тензор той же размерности
        """
        identity = x
        out = self.lin1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + identity

class DistanceFeatures(nn.Module):
    """
    Модуль для вычисления признаков из расстояний.
    Преобразует относительные позиции в физически информативные признаки.
    """
    
    def __init__(self, output_dim: int = 16, r_cutoff: float = 5.0):
        """
        Инициализирует модуль признаков расстояний.
        
        Args:
            output_dim: Размерность выходных признаков
            r_cutoff: Радиус отсечения
        """
        super(DistanceFeatures, self).__init__()
        
        self.output_dim = output_dim
        self.r_cutoff = r_cutoff
        
        self.rbf_centers = nn.Parameter(
            torch.linspace(0.3, r_cutoff, output_dim), 
            requires_grad=False
        )
        self.rbf_widths = nn.Parameter(
            torch.ones(output_dim) * 0.5, 
            requires_grad=True
        )
    
    def forward(self, rel_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через модуль.
        
        Args:
            rel_pos: Относительные позиции [batch_size, 3]
            
        Returns:
            Кортеж (признаки, расстояния)
        """
        distances = torch.norm(rel_pos, dim=1, keepdim=True)
        
        safe_distances = torch.clamp(distances, min=1e-6)
        
        directions = rel_pos / safe_distances
        
        rbf_features = torch.exp(
            -self.rbf_widths.abs() * (distances - self.rbf_centers.unsqueeze(0)) ** 2
        )
        
        return rbf_features, distances.squeeze(1)

class PotentialNN(nn.Module):
    """
    Нейронная сеть для моделирования потенциальной энергии взаимодействия.
    Принимает относительные позиции и предсказывает силы.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        max_potential: float = 50.0,
        dropout_rate: float = 0.1,
        device: str = "cuda"
    ):
        """
        Инициализация сети потенциальной энергии.
        
        Args:
            hidden_dim: Размерность скрытых слоев
            num_blocks: Количество блоков остаточной связи
            max_potential: Максимальное значение потенциальной энергии
            dropout_rate: Вероятность отключения нейронов
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.max_potential = max_potential
        self.device = device
        
        self.input_layer = nn.Sequential(
            nn.Linear(3, hidden_dim),
            ScaleSILU()
        )
        
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def compute_potential(self, r: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет потенциальную энергию для заданных относительных позиций.
        
        Args:
            r: Тензор относительных позиций [batch_size, 3]
            
        Returns:
            Тензор потенциальных энергий [batch_size]
        """
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получено {type(r)}")
            
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Ожидается тензор формы [batch_size, 3], получено {r.shape}")
            
        dist = torch.norm(r, dim=1, keepdim=True)
        
        safe_dist = torch.clamp(dist, min=1e-10)
        
        r_normalized = r / safe_dist
        
        x = self.input_layer(r_normalized)
        
        for block in self.residual_blocks:
            x = block(x)
        
        raw_potential = self.output_layer(x)
        
        scaled_potential = raw_potential / (1.0 + dist)
        clamped_potential = torch.clamp(scaled_potential, -self.max_potential, self.max_potential)
        
        return clamped_potential.squeeze(-1)
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: вычисляет силы на основе градиента потенциала.
        
        Args:
            r: Тензор относительных позиций [batch_size, 3] с требованием градиента
            
        Returns:
            Тензор сил [batch_size, 3]
        """
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получено {type(r)}")
            
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Ожидается тензор формы [batch_size, 3], получено {r.shape}")
            
        if not r.requires_grad:
            r = r.detach().clone().requires_grad_(True)
        
        potential = self.compute_potential(r)
        
        potential_sum = torch.sum(potential)
        
        forces = -torch.autograd.grad(
            potential_sum, r, create_graph=True, retain_graph=True
        )[0]
        
        return forces
    
class KANPotentialModel(nn.Module):
    """
    Модель потенциальной энергии на основе Kolmogorov-Arnold Networks (KAN).
    
    KAN используют теорему Колмогорова-Арнольда для представления
    многомерных функций через композиции одномерных функций,
    что потенциально обеспечивает лучшую обобщающую способность.
    """
    
    def __init__(
        self,
        hidden_dim: int = 16,
        num_layers: int = 2,
        max_potential: float = 50.0,
        device: str = "cuda"
    ):
        """
        Инициализация KAN модели потенциала.
        
        Args:
            hidden_dim: Размерность скрытых слоев KAN
            num_layers: Количество слоев в KAN
            max_potential: Максимальное значение потенциальной энергии
            device: Устройство для вычислений
        """
        super().__init__()
        
        if kan is None:
            raise ImportError("Package 'kan' is required but not installed")
            
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_potential = max_potential
        self.device = device
        
        self.kan_model = kan.KAN(
            in_dim=3,
            out_dim=1,
            hidden_dims=[hidden_dim] * num_layers,
            grid_size=10,
            device=device
        )
        
        self.scale_factor = nn.Parameter(torch.tensor([1.0]))
    
    def compute_potential(self, r: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет потенциальную энергию для заданных относительных позиций.
        
        Args:
            r: Тензор относительных позиций [batch_size, 3]
            
        Returns:
            Тензор потенциальных энергий [batch_size]
        """
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получено {type(r)}")
            
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Ожидается тензор формы [batch_size, 3], получено {r.shape}")
        
        dist = torch.norm(r, dim=1, keepdim=True)
        
        safe_dist = torch.clamp(dist, min=1e-10)
        
        r_normalized = r / safe_dist
        
        raw_potential = self.kan_model(r_normalized)
        
        scaled_potential = raw_potential * self.scale_factor / (1.0 + dist)
        
        clamped_potential = torch.clamp(
            scaled_potential, -self.max_potential, self.max_potential
        )
        
        return clamped_potential.squeeze(-1)
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: вычисляет силы на основе градиента потенциала.
        
        Args:
            r: Тензор относительных позиций [batch_size, 3] с требованием градиента
            
        Returns:
            Тензор сил [batch_size, 3]
        """
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получено {type(r)}")
            
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Ожидается тензор формы [batch_size, 3], получено {r.shape}")
            
        if not r.requires_grad:
            r = r.detach().clone().requires_grad_(True)
        
        potential = self.compute_potential(r)
        
        potential_sum = torch.sum(potential)
        
        forces = -torch.autograd.grad(
            potential_sum, r, create_graph=True, retain_graph=True
        )[0]
        
        return forces
    
    def get_formula(self, precision: int = 4) -> str:
        """
        Возвращает формулу потенциала в символьном виде.
        
        Args:
            precision: Количество знаков после запятой в коэффициентах
            
        Returns:
            Строка с формулой потенциала
        """
        var_names = ['x', 'y', 'z']
        formula = self.kan_model.get_formula(var_names, precision)
        
        scale_str = f"{self.scale_factor.item():.{precision}f}"
        formula = f"({scale_str}) * ({formula})"
        
        formula = f"({formula}) / (1 + sqrt(x^2 + y^2 + z^2))"
        
        return formula

    def get_simplified_formula(self, precision: int = 4) -> str:
        """
        Возвращает упрощенную формулу потенциала.
        
        Args:
            precision: Количество знаков после запятой в коэффициентах
            
        Returns:
            Строка с упрощенной формулой
        """
        full_formula = self.get_formula(precision)
        
        return full_formula