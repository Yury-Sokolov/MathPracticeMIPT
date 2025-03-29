import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .cluster import Cluster


class Simulation:
    """Класс для управления симуляцией столкновения кластеров"""

    def __init__(self, potential, epsilon=1e-5, t_end=200.0, convergence_threshold=1e-3, clip_force=1000):
        """
        Инициализация симуляции
        
        Args:
            potential: Объект потенциала взаимодействия
            epsilon (float): Параметр точности для адаптивного метода Рунге-Кутты
            t_end (float): Время окончания симуляции
            convergence_threshold (float): Порог для остановки симуляции при малых изменениях
        """
        self.potential = potential
        self.epsilon = epsilon
        self.t_end = t_end
        self.convergence_threshold = convergence_threshold
        self.clusters = []
        self.nucleons = []
        self.times = []
        self.trajectories = []
        self.velocities_history = []
        self.save_interval = 0
        self.clip_force = clip_force

    def add_cluster(self, cluster):
        """
        Добавление кластера в симуляцию
        
        Args:
            cluster (Cluster): Кластер для добавления
        """

        self.clusters.append(cluster)
        self.nucleons.extend(cluster.nucleons)

    def compute_forces(self):
        """Вычисление сил между всеми нуклонами"""
        positions = np.array([n.position for n in self.nucleons])

        for nucleon in self.nucleons:
            nucleon.force = np.zeros(3)

        distances = squareform(pdist(positions))
        
        mask = distances < self.potential.r1
        np.fill_diagonal(mask, False)

        interacting_pairs = np.where(mask)

        for i, j in zip(*interacting_pairs):
            if i < j:
                r = distances[i, j]
                direction = (positions[i] - positions[j]) / r

                force_magnitude = self.potential.compute_force(r)
                
                force = force_magnitude * direction
                self.nucleons[i].force += force
                self.nucleons[j].force -= force

    def run(self, save_interval, dt):
        """
        Запуск симуляции столкновения с использованием метода Leapfrog
        
        Args:
            save_interval (int): Количество шагов между сохранениями состояния
            dt (float): Шаг времени для интегрирования
        
        Returns:
            tuple: (times, trajectories, velocities_history)
        """
        N = len(self.nucleons)
        self.save_interval = save_interval
        if N == 0:
            print("Ошибка: нет нуклонов для симуляции")
            return [], [], []

        self.times = [0]
        self.trajectories = [np.array([n.position for n in self.nucleons])]
        self.velocities_history = [np.array([n.velocity for n in self.nucleons])]
        kinetic_energies, potential_energies, total_energies = self.compute_total_energy_per_nucleon()
        for i in range(N):
            nucleon =self.nucleons[i]
            print(f"Нуклон {i}, x {nucleon.position[0]:.2f}, y {nucleon.position[1]:.2f} z {nucleon.position[2]:.2f} Kinetic {kinetic_energies[i]:.2f} Potential {potential_energies[i]:.2f} Total {total_energies[i]:.2f} ")
        initial_cluster_ids = [n.cluster_id for n in self.nucleons]

        t = 0

        print(f"Начало симуляции: {N} нуклонов")
        print(f"Сохранение каждые {save_interval} шагов")
        print(f"Шаг времени: {dt}")
        start_time = time.time()

        step_count = 0

        self.compute_forces()
        with tqdm(total=int(self.t_end/dt)) as pb:
            while t < self.t_end:
                if self.clip_force > 0:
                    for nucleon in self.nucleons:
                        force_magnitude = np.linalg.norm(nucleon.force)
                        if force_magnitude > self.clip_force:
                           nucleon.force = (nucleon.force / force_magnitude) * self.clip_force

                for nucleon in self.nucleons:
                    nucleon.velocity += nucleon.force / nucleon.mass * dt / 2

                for nucleon in self.nucleons:
                    nucleon.position += nucleon.velocity * dt

                positions = np.array([n.position for n in self.nucleons])
                distances = squareform(pdist(positions))
                min_distance = 0.00001

                for i in range(N):
                    for j in range(i+1, N):
                        if distances[i, j] < min_distance:
                            direction = positions[i] - positions[j]
                            direction_norm = np.linalg.norm(direction)
                            if direction_norm > 0:
                                direction = direction / direction_norm
                                displacement = (min_distance - distances[i, j]) / 2
                                self.nucleons[i].position += displacement * direction
                                self.nucleons[j].position -= displacement * direction

                self.compute_forces()
                if self.clip_force > 0:
                    for nucleon in self.nucleons:
                        force_magnitude = np.linalg.norm(nucleon.force)
                        if force_magnitude > self.clip_force:
                            nucleon.force = (nucleon.force / force_magnitude) * self.clip_force

                for nucleon in self.nucleons:
                    nucleon.velocity += nucleon.force / nucleon.mass * dt / 2

                t += dt
                step_count += 1
                pb.update()
                if step_count % save_interval == 0:
                    self.times.append(t)
                    self.trajectories.append(np.array([n.position for n in self.nucleons]))
                    self.velocities_history.append(np.array([n.velocity for n in self.nucleons]))




        for i, nucleon in enumerate(self.nucleons):
            nucleon.cluster_id = initial_cluster_ids[i]

        end_time = time.time()
        print(f"\nСимуляция завершена за {end_time - start_time:.2f} секунд")
        print(f"Выполнено {step_count} шагов, сохранено {len(self.times)} состояний")

        return self.times, self.trajectories, self.velocities_history

    def create_animation(self, filename=None, fps=1, limit=10):
        """
        Создание анимации с индивидуальными траекториями для каждой частицы

        Args:
            filename (str): Имя файла для сохранения анимации
            fps (int): Кадров в секунду
            limit (int): Границы пространства отображения
        """
        if not self.trajectories:
            print("Ошибка: нет данных для анимации")
            return None

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        nucleons = self.nucleons
        trajectories =np.asarray(self.trajectories)
        frames_count = len(self.times)
        n_particles = len(nucleons)

        unique_cids = list({n.cluster_id for n in nucleons})
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_cids)))
        color_map = {cid: color for cid, color in zip(unique_cids, colors)}
        particle_colors = [color_map[n.cluster_id] for n in nucleons]

        scatter = ax.scatter([], [], [], s=50, alpha=0.8)
        lines = [ax.plot([], [], [], c=color, alpha=0.3)[0] for color in particle_colors]

        scatter.set_facecolors(particle_colors)
        pb = tqdm(total=len(trajectories))
        def update(frame):
            current_positions = trajectories[frame]
            scatter._offsets3d = current_positions.T
            pb.update()
            for i in range(n_particles):
                x = trajectories[:frame + 1, i, 0]
                y = trajectories[:frame + 1, i, 1]
                z = trajectories[:frame + 1, i, 2]
                lines[i].set_data(x, y)
                lines[i].set_3d_properties(z)

            ax.set_title(f't = {self.times[frame]:.5f}')
            return [scatter] + lines

        ani = animation.FuncAnimation(
            fig, update, frames=frames_count,
            init_func=lambda: [scatter] + lines,
            blit=True, interval=1000 / fps
        )

        if filename:
            try:
                writer = animation.FFMpegWriter(fps=fps)
                ani.save(filename, writer=writer)
                pb.close()
                print(f"Анимация сохранена в {filename}")
            except Exception as e:
                pb.close()
                print(f"Ошибка сохранения: {str(e)}")

        plt.close()
        return ani

    def cluster_analysis(self, clustering_algorithm, save_path=None, limit=10):
        """
        Анализ кластеров с использованием алгоритма из sklearn
        
        Args:
            clustering_algorithm: Объект кластеризации из sklearn с методом fit_predict
            save_path (str): Путь для сохранения графика кластеризации
            
        Returns:
            np.ndarray: Метки кластеров для каждого нуклона
        """
        if not self.trajectories:
            print("Ошибка: нет данных для кластеризации")
            return None

        final_positions = self.trajectories[-1]

        cluster_labels = clustering_algorithm.fit_predict(final_positions)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            ax.scatter(
                final_positions[mask, 0],
                final_positions[mask, 1],
                final_positions[mask, 2],
                c=[colors[i]],
                s=50,
                alpha=0.8,
                label=f'Кластер {label}'
            )

        ax.set_title('Результаты кластеризации')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"График кластеризации сохранен в {save_path}")

        plt.show()

        return cluster_labels

    def compute_kinetic_energy_per_nucleon(self):
        """
        Вычисление кинетической энергии для каждого нуклона
        
        Returns:
            list: Список кинетических энергий каждого нуклона
        """
        kinetic_energies = []
        for nucleon in self.nucleons:
            velocity_squared = np.sum(nucleon.velocity**2)
            kinetic_energy = 0.5 * nucleon.mass * velocity_squared
            kinetic_energies.append(kinetic_energy)
        
        return kinetic_energies
    
    def compute_potential_energy_per_nucleon(self):
        """
        Вычисление потенциальной энергии для каждого нуклона
        
        Returns:
            list: Список потенциальных энергий каждого нуклона
        """
        positions = np.array([n.position for n in self.nucleons])
        distances = squareform(pdist(positions))
        
        potential_energies = [0.0] * len(self.nucleons)
        
        mask = distances < self.potential.r1
        np.fill_diagonal(mask, False)
        
        interacting_pairs = np.where(mask)
        
        for i, j in zip(*interacting_pairs):
            if i < j:
                r = distances[i, j]
                pair_potential = self.potential.compute(r)
                
                potential_energies[i] += pair_potential / 2
                potential_energies[j] += pair_potential / 2
        
        return potential_energies
    
    def compute_total_energy_per_nucleon(self):
        """
        Вычисление полной энергии для каждого нуклона
        
        Returns:
            tuple: (список кинетических энергий, список потенциальных энергий, 
                   список полных энергий)
        """
        kinetic_energies = self.compute_kinetic_energy_per_nucleon()
        potential_energies = self.compute_potential_energy_per_nucleon()
        
        total_energies = [k + p for k, p in zip(kinetic_energies, potential_energies)]
        
        return kinetic_energies, potential_energies, total_energies
