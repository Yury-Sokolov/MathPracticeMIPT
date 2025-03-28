import numpy as np
from abc import ABC, abstractmethod

class Potential(ABC):
    """Абстрактный класс для потенциалов взаимодействия"""
    
    @abstractmethod
    def compute(self, r):
        """Вычисляет потенциал на расстоянии r"""
        pass
    
    @abstractmethod
    def compute_force(self, r):
        """Вычисляет силу на расстоянии r"""
        pass

class YukawaPotential(Potential):
    """Потенциал Юкавы: V(r) = V0 * exp(-alpha*r) / r"""
    
    def __init__(self, V0, alpha, r1):
        self.V0 = V0
        self.alpha = alpha
        self.r1 = r1
    
    def compute(self, r):
        """
        Потенциал Юкавы: V(r) = V0 * exp(-alpha*r) / r
        С обрезанием на r1 и сглаживанием при r→0
        """
        r_safe = np.maximum(r, 1e-10)
        
        potential = self.V0 * np.exp(-self.alpha * r_safe) / r_safe
        
        potential = np.where(r_safe < self.r1, potential, 0.0)
        
        return potential
    
    def compute_force(self, r):
        """
        Сила, соответствующая потенциалу Юкавы: F(r) = -∇V(r)
        Аналитическое дифференцирование:
        F(r) = -V0 * exp(-alpha*r) * (1/r^2 + alpha/r)
        """
        r_safe = np.maximum(r, 1e-10)
        
        force_magnitude = self.V0 * np.exp(-self.alpha * r_safe) * (1/r_safe**2 + self.alpha/r_safe)
        
        force_magnitude = np.where(r_safe < self.r1, force_magnitude, 0.0)
        
        return force_magnitude