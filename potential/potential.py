import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

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
    
    def plot(self, r_min=0.01, r_max=10.0, num_points=1000, show_force=True, figsize=(10, 6), save_path=None):
        """
        Строит график потенциала и силы от расстояния
        
        Параметры:
        ----------
        r_min : float
            Минимальное расстояние для графика
        r_max : float
            Максимальное расстояние для графика
        num_points : int
            Количество точек для построения графика
        show_force : bool
            Показывать ли график силы
        figsize : tuple
            Размер графика (ширина, высота) в дюймах
        save_path : str, optional
            Путь для сохранения графика. Если None, график не сохраняется
        """
        r_values = np.linspace(r_min, r_max, num_points)
        potential_values = self.compute(r_values)
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # График потенциала
        ax1.plot(r_values, potential_values, 'b-', label='Потенциал V(r)')
        ax1.set_xlabel('Расстояние r')
        ax1.set_ylabel('Потенциал V(r)')
        ax1.tick_params(axis='y')
        
        if show_force:
            # График силы на второй оси Y
            force_values = self.compute_force(r_values)
            ax2 = ax1.twinx()
            ax2.plot(r_values, force_values, 'r-', label='Сила F(r)')
            ax2.set_ylabel('Сила F(r)',)
            ax2.tick_params(axis='y')
        
        # Добавление заголовка и легенды
        plt.title('Зависимость потенциала и силы от расстояния')
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_force:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1.legend(loc='best')
        
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig, ax1

class YukawaPotential(Potential):
    """Потенциал Юкавы: V(r) = V0 * exp(-alpha*r) / r"""
    
    def __init__(self, V0, alpha, r1):
        self.V0 = V0
        self.alpha = alpha
        self.r1 = r1
    
    def compute(self, r):

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
        
        force_magnitude =  self.V0 * np.exp(-self.alpha * r_safe) * (1/r_safe**2 + self.alpha/r_safe)
        
        force_magnitude = np.where(r_safe < self.r1, force_magnitude, 0.0)
        
        return force_magnitude