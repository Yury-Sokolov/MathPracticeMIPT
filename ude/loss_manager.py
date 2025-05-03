import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional


class LossManager:
    """
    Менеджер функций потерь для обучения UDE моделей.
    Объединяет различные компоненты потерь для физически обоснованного обучения.
    """
    
    def __init__(
        self,
        potential_weight: float = 0.5,
        force_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        symmetric_weight: float = 0.5
    ):
        """
        Инициализирует менеджер потерь.
        
        Args:
            potential_weight: Вес потери потенциала
            force_weight: Вес потери силы
            smoothness_weight: Вес потери гладкости
            symmetric_weight: Вес потери симметрии
        """
        self.potential_weight = potential_weight
        self.force_weight = force_weight
        self.smoothness_weight = smoothness_weight
        self.symmetric_weight = symmetric_weight
    
    def symmetry_loss(self, model: nn.Module, rel_positions: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет потерю симметрии для модели потенциала.
        Потенциал должен быть инвариантен к изменению знака относительной позиции.
        
        Args:
            model: Модель потенциала
            rel_positions: Относительные позиции [batch_size, 3]
            
        Returns:
            Потеря симметрии
        """
        with torch.enable_grad():
            forward_potential = model.compute_potential(rel_positions)
            
            reverse_positions = -rel_positions
            reverse_potential = model.compute_potential(reverse_positions)
            
            symmetry_loss = F.mse_loss(forward_potential, reverse_potential)
            
            return symmetry_loss
    
    def smoothness_loss(self, model: nn.Module, rel_positions: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет потерю гладкости для модели потенциала.
        Это стимулирует гладкий, непрерывный потенциал без резких изменений.
        
        Args:
            model: Модель потенциала
            rel_positions: Относительные позиции [batch_size, 3]
            
        Returns:
            Потеря гладкости
        """
        with torch.enable_grad():
            epsilon = 1e-3
            perturbed_positions = rel_positions + torch.randn_like(rel_positions) * epsilon
            
            original_potential = model.compute_potential(rel_positions)
            perturbed_potential = model.compute_potential(perturbed_positions)
            
            potential_diff = torch.abs(perturbed_potential - original_potential)
            position_diff = torch.norm(perturbed_positions - rel_positions, dim=1)
            
            smoothness_loss = torch.mean(potential_diff / (position_diff + 1e-8))
            
            return smoothness_loss
    
    def force_magnitude_loss(
        self, 
        model: nn.Module, 
        rel_positions: torch.Tensor, 
        true_forces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Вычисляет потерю по величине силы.
        Если предоставлены истинные силы, сравнивает с ними предсказанные.
        Иначе, стимулирует ненулевые разумные силы.
        
        Args:
            model: Модель потенциала
            rel_positions: Относительные позиции [batch_size, 3]
            true_forces: Истинные силы [batch_size, 3], если есть
            
        Returns:
            Потеря по величине силы
        """
        with torch.enable_grad():
            potentials = model.compute_potential(rel_positions)
            total_potential = torch.sum(potentials)
            
            forces = -torch.autograd.grad(
                total_potential, rel_positions, 
                create_graph=True, retain_graph=True
            )[0]
            
            if true_forces is not None:
                force_loss = F.mse_loss(forces, true_forces)
            else:
                force_norms = torch.norm(forces, dim=1)
                
                force_loss = torch.mean(torch.exp(-force_norms))
                
                max_force = 10.0
                force_loss += torch.mean(torch.relu(force_norms - max_force))
            
            return force_loss
    
    def compute_physics_loss(
        self, 
        model: nn.Module, 
        rel_positions: torch.Tensor,
        true_forces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Вычисляет общую физическую потерю, комбинируя различные компоненты.
        
        Args:
            model: Модель потенциала
            rel_positions: Относительные позиции [batch_size, 3]
            true_forces: Истинные силы [batch_size, 3], если есть
            
        Returns:
            Комбинированная физическая потеря
        """
        sym_loss = self.symmetry_loss(model, rel_positions)
        smooth_loss = self.smoothness_loss(model, rel_positions)
        force_loss = self.force_magnitude_loss(model, rel_positions, true_forces)
        
        combined_loss = (
            self.symmetric_weight * sym_loss +
            self.smoothness_weight * smooth_loss +
            self.force_weight * force_loss
        )
        
        return combined_loss 