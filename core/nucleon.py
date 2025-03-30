import torch

class Nucleon:
    """Класс, представляющий нуклон (частицу) в симуляции"""
    
    def __init__(self, position=None, velocity=None, mass=1.0, device='cuda', dtype=torch.float32):
        """
        Инициализация нуклона
        
        Args:
            position (torch.Tensor): Начальная позиция [x, y, z]
            velocity (torch.Tensor): Начальная скорость [vx, vy, vz]
            mass (float): Масса нуклона
            device: Устройство для вычислений (cpu/cuda)
            dtype: Тип данных тензоров
        """
        self.device = device
        self.dtype = dtype
        
        self.position = torch.zeros(3, device=device, dtype=dtype) if position is None else position.to(device, dtype)
        self.velocity = torch.zeros(3, device=device, dtype=dtype) if velocity is None else velocity.to(device, dtype)
        self.force = torch.zeros(3, device=device, dtype=dtype)
        self.mass = torch.tensor(mass, device=device, dtype=dtype)
        self.cluster_id = -1
    
    def update_position(self, dt):
        """Обновление позиции на основе скорости"""
        self.position += self.velocity * dt
    
    def update_velocity(self, dt):
        """Обновление скорости на основе силы"""
        self.velocity += self.force / self.mass * dt
    
    def kinetic_energy(self):
        """Вычисление кинетической энергии нуклона"""
        return 0.5 * self.mass * torch.sum(self.velocity**2)