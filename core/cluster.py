import numpy as np
from .nucleon import Nucleon
from scipy.spatial.distance import pdist, squareform

class Cluster:
    """Класс, представляющий кластер нуклонов"""
    
    def __init__(self, nucleons=None, position=None, velocity=None, radius=None, N=None, random_velocity=None):
        """
        Инициализация кластера
        
        Args:
            nucleons (list): Список объектов Nucleon
            position (np.ndarray): Позиция центра кластера [x, y, z]
            velocity (np.ndarray): Скорость кластера [vx, vy, vz]
            radius (float): Радиус кластера
            N (int): Количество нуклонов (если nucleons не задан)
        """
        if nucleons is not None:
            self.nucleons = nucleons
        elif N is not None:
            self.nucleons = []
            radius = radius if radius is not None else N**(1/3)
            
            positions = np.random.normal(0, 1, (N, 3))
            distances = np.sqrt(np.sum(positions**2, axis=1))
            
            for i in range(N):
                r = radius * np.random.random()**(1/3)
                pos = positions[i] / distances[i] * r
                
                if position is not None:
                    pos += position
                
                nucleon = Nucleon(position=pos, velocity=np.zeros(3), mass=1.0)
                self.nucleons.append(nucleon)

            if random_velocity is not None:
                self.add_random_velocity(random_velocity)
            if velocity is not None:
                self.add_velocity(velocity)
        else:
            self.nucleons = []
            
        self.id = id(self)
        
        for nucleon in self.nucleons:
            nucleon.cluster_id = self.id
    
    @classmethod
    def initialize_spherical(cls, N, radius=None, mass=1.0):
        """
        Создание кластера с N нуклонами в сферическом объеме
        
        Args:
            N (int): Количество нуклонов
            radius (float): Радиус сферы. Если None, то пропорционален N^(1/3)
            mass (float): Масса каждого нуклона
        
        Returns:
            Cluster: Новый кластер
        """
        if radius is None:
            radius = N**(1/3)
        
        positions = np.random.normal(0, 1, (N, 3))
        distances = np.sqrt(np.sum(positions**2, axis=1))
        
        nucleons = []
        for i in range(N):
            r = radius * np.random.random()**(1/3)
            position = positions[i] / distances[i] * r
            nucleon = Nucleon(position=position, velocity=np.zeros(3), mass=mass)
            nucleons.append(nucleon)
        
        return cls(nucleons)
    
    def add_nucleon(self, nucleon):
        """Добавление нуклона в кластер"""
        self.nucleons.append(nucleon)
        nucleon.cluster_id = self.id
    
    def center_of_mass(self):
        """Вычисление центра масс кластера"""
        total_mass = sum(n.mass for n in self.nucleons)
        com = np.zeros(3)
        for nucleon in self.nucleons:
            com += nucleon.position * nucleon.mass
        return com / total_mass
    
    def total_mass(self):
        """Вычисление общей массы кластера"""
        return sum(n.mass for n in self.nucleons)
    
    def add_velocity(self, velocity):
        """Добавление скорости ко всем нуклонам кластера"""
        for nucleon in self.nucleons:
            nucleon.velocity += velocity

    def add_random_velocity(self, velocity):
        """Добавление случайной скорости ко всем нуклонам кластера"""
        for nucleon in self.nucleons:
            random_vector = np.random.normal(size=3)
            unit_vector = random_vector / np.linalg.norm(random_vector)
            nucleon.velocity += velocity * unit_vector
    
    def add_random_rotation(self, omega_scale=0.1):
        """Добавление случайного вращения к кластеру"""
        com = self.center_of_mass()
        omega = omega_scale * np.random.normal(0, 1, 3)
        
        for nucleon in self.nucleons:
            r = nucleon.position - com
            nucleon.velocity += np.cross(omega, r)
    
    def kinetic_energy(self):
        """Вычисление общей кинетической энергии кластера"""
        return sum(n.kinetic_energy() for n in self.nucleons)

            