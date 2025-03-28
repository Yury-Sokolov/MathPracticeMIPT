import numpy as np

class Nucleon:
    """Класс, представляющий нуклон (частицу) в симуляции"""
    
    def __init__(self, position=None, velocity=None, mass=1.0):
        """
        Инициализация нуклона
        
        Args:
            position (np.ndarray): Начальная позиция [x, y, z]
            velocity (np.ndarray): Начальная скорость [vx, vy, vz]
            mass (float): Масса нуклона
        """
        self.position = np.zeros(3) if position is None else np.array(position)
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity)
        self.force = np.zeros(3)
        self.mass = mass
        self.cluster_id = -1
    
    def update_position(self, dt):
        """Обновление позиции на основе скорости"""
        self.position += self.velocity * dt
    
    def update_velocity(self, dt):
        """Обновление скорости на основе силы"""
        self.velocity += self.force / self.mass * dt
    
    def kinetic_energy(self):
        """Вычисление кинетической энергии нуклона"""
        return 0.5 * self.mass * np.sum(self.velocity**2)