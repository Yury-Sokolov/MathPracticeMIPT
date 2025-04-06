import torch


class Cluster:
    """Класс, представляющий кластер нуклонов (PyTorch версия)"""

    def __init__(self, nucleons=None, position=None, velocity=None,
                 radius=None, N=None, random_velocity=None,
                 device='cuda', dtype=torch.float32):
        """
        Инициализация кластера

        Args:
            nucleons (list): Список объектов Nucleon (тензоры PyTorch)
            position (torch.Tensor): Позиция центра кластера [x, y, z]
            velocity (torch.Tensor): Скорость кластера [vx, vy, vz]
            radius (float): Радиус кластера
            N (int): Количество нуклонов (если nucleons не задан)
            device: Устройство для вычислений (cpu/cuda)
            dtype: Тип данных тензоров
        """
        self.device = device
        self.dtype = dtype

        if nucleons is not None:
            self.nucleons = nucleons
        elif N is not None:
            self.nucleons = {
                'positions': None,
                'velocities': None,
                'masses': None
            }

            radius = radius if radius is not None else N ** (1 / 3)

            positions = torch.randn(N, 3, device=device, dtype=dtype)
            distances = torch.linalg.norm(positions, dim=1, keepdim=True)
            directions = positions / distances

            r = radius * torch.pow(torch.rand(N, 1, device=device), 1 / 3)
            positions = directions * r

            if position is not None:
                positions += position.to(device)

            self.nucleons = {
                'positions': positions,
                'velocities': torch.zeros((N, 3), device=device, dtype=dtype),
                'masses': torch.ones(N, device=device, dtype=dtype)
            }

            if random_velocity is not None:
                self.add_random_velocity(random_velocity)
            if velocity is not None:
                self.add_velocity(velocity)
        else:
            self.nucleons = {
                'positions': torch.empty((0, 3), device=device, dtype=dtype),
                'velocities': torch.empty((0, 3), device=device, dtype=dtype),
                'masses': torch.empty(0, device=device, dtype=dtype)
            }

        self.id = id(self)

    @classmethod
    def initialize_spherical(cls, N, radius=None, mass=1.0, **kwargs):
        """
        Создание кластера с N нуклонами в сферическом объеме

        Args:
            N (int): Количество нуклонов
            radius (float): Радиус сферы. Если None, то пропорционален N^(1/3)
            mass (float): Масса каждого нуклона

        Returns:
            Cluster: Новый кластер
        """
        device = kwargs.get('device', 'cpu')
        dtype = kwargs.get('dtype', torch.float32)

        if radius is None:
            radius = N ** (1 / 3)

        positions = torch.randn(N, 3, device=device, dtype=dtype)
        distances = torch.linalg.norm(positions, dim=1, keepdim=True)
        directions = positions / distances

        r = radius * torch.pow(torch.rand(N, 1, device=device), 1 / 3)
        positions = directions * r

        return cls(
            nucleons={
                'positions': positions,
                'velocities': torch.zeros((N, 3), device=device, dtype=dtype),
                'masses': torch.full((N,), mass, device=device, dtype=dtype)
            },
            **kwargs
        )

    def add_nucleons(self, positions, velocities, masses):
        """Добавление нуклонов в кластер"""
        self.nucleons['positions'] = torch.cat([
            self.nucleons['positions'],
            positions.to(self.device)
        ], dim=0)

        self.nucleons['velocities'] = torch.cat([
            self.nucleons['velocities'],
            velocities.to(self.device)
        ], dim=0)

        self.nucleons['masses'] = torch.cat([
            self.nucleons['masses'],
            masses.to(self.device)
        ], dim=0)

    def center_of_mass(self):
        """Вычисление центра масс кластера"""
        total_mass = self.nucleons['masses'].sum()
        com = (self.nucleons['positions'] * self.nucleons['masses'].unsqueeze(1)).sum(dim=0)
        return com / total_mass

    def total_mass(self):
        """Вычисление общей массы кластера"""
        return self.nucleons['masses'].sum()

    def add_velocity(self, velocity):
        """Добавление скорости ко всем нуклонам кластера"""
        self.nucleons['velocities'] += velocity.to(self.device)

    def add_random_velocity(self, magnitude):
        """Добавление случайной скорости ко всем нуклонам кластера"""
        random_vectors = torch.randn_like(self.nucleons['velocities'])
        unit_vectors = random_vectors / torch.linalg.norm(
            random_vectors, dim=1, keepdim=True)
        self.nucleons['velocities'] += magnitude * unit_vectors

    def add_random_rotation(self, omega_scale=0.1):
        """Добавление случайного вращения к кластеру"""
        com = self.center_of_mass()
        omega = omega_scale * torch.randn(3, device=self.device)

        r = self.nucleons['positions'] - com.unsqueeze(0)
        velocities = torch.cross(omega.expand_as(r), r)
        self.nucleons['velocities'] += velocities

    def kinetic_energy(self):
        """Вычисление общей кинетической энергии кластера"""
        speeds_sq = torch.sum(self.nucleons['velocities'] ** 2, dim=1)
        return 0.5 * torch.sum(self.nucleons['masses'] * speeds_sq)

    @property
    def positions(self):
        return self.nucleons['positions']

    @property
    def velocities(self):
        return self.nucleons['velocities']

    @property
    def masses(self):
        return self.nucleons['masses']

    def translate(self, offset):
        """Перемещение всех нуклонов на заданный вектор"""
        self.nucleons['positions'] += offset
        
    def clone(self):
        """Создает глубокую копию кластера
        
        Returns:
            Cluster: Новый кластер с теми же параметрами
        """
        new_nucleons = {
            'positions': self.nucleons['positions'].clone(),
            'velocities': self.nucleons['velocities'].clone(),
            'masses': self.nucleons['masses'].clone()
        }
        
        return Cluster(
            nucleons=new_nucleons,
            device=self.device,
            dtype=self.dtype
        )
            