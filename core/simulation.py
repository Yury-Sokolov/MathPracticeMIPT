import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK45
from scipy.spatial.distance import pdist, squareform

from .cluster import Cluster


class Simulation:
    """Класс для управления симуляцией столкновения кластеров"""

    def __init__(self, potential, epsilon=1e-5, t_end=200.0, convergence_threshold=1e-3, delta_time=30):
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
        self.delta_time = delta_time

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
                
                force = -1 * force_magnitude * direction
                self.nucleons[i].force += force
                self.nucleons[j].force -= force

    def system_dynamics(self, t, state):
        """
        Функция динамики системы для интегратора
        
        Args:
            t (float): Текущее время системы
            state (np.ndarray): Текущее состояние системы (позиции и скорости)
        
        Returns:
            np.ndarray: Производные состояния
        """
        N = len(self.nucleons)

        positions = state[:N * 3].reshape(N, 3)
        velocities = state[N * 3:].reshape(N, 3)

        for i, nucleon in enumerate(self.nucleons):
            nucleon.position = positions[i]
            nucleon.velocity = velocities[i]

        self.compute_forces()

        derivatives = np.zeros_like(state)
        for i, nucleon in enumerate(self.nucleons):
            derivatives[i * 3:(i + 1) * 3] = nucleon.velocity  # dx/dt = v
            derivatives[N * 3 + i * 3:N * 3 + (i + 1) * 3] = nucleon.force / nucleon.mass  # dv/dt = F/m

        return derivatives

    def run(self, save_interval):
        """
        Запуск симуляции столкновения
        
        Args:
            save_interval (int): Количество шагов между сохранениями состояния
        
        Returns:
            tuple: (times, trajectories, velocities_history)
        """
        N = len(self.nucleons)
        self.save_interval = save_interval
        if N == 0:
            print("Ошибка: нет нуклонов для симуляции")
            return [], [], []

        initial_positions = np.array([n.position for n in self.nucleons]).flatten()
        initial_velocities = np.array([n.velocity for n in self.nucleons]).flatten()
        initial_state = np.concatenate([initial_positions, initial_velocities])

        integrator = RK45(
            self.system_dynamics,
            0,
            initial_state,
            self.t_end,
            rtol=self.epsilon,
        )

        self.times = [0]
        self.trajectories = [np.array([n.position for n in self.nucleons])]
        self.velocities_history = [np.array([n.velocity for n in self.nucleons])]

        initial_cluster_ids = [n.cluster_id for n in self.nucleons]

        print(f"Начало симуляции: {N} нуклонов")
        print(f"Сохранение каждые {save_interval} шагов")
        start_time = time.time()
        
        step_count = 0
        
        while integrator.status == 'running':
            integrator.step()
            step_count += 1

            if step_count % save_interval == 0:
                current_state = integrator.y
                current_positions = current_state[:N * 3].reshape(N, 3)
                current_velocities = current_state[N * 3:].reshape(N, 3)

                for i, nucleon in enumerate(self.nucleons):
                    nucleon.position = current_positions[i]
                    nucleon.velocity = current_velocities[i]

                self.times.append(integrator.t)
                self.trajectories.append(current_positions.copy())
                self.velocities_history.append(current_velocities.copy())
                
                elapsed_time = time.time() - start_time
                print(f"Шаг {step_count}, t = {integrator.t:.4f}, прошло времени: {elapsed_time:.2f} сек", end='\r')

        # Сохраняем последнее состояние, если оно не было сохранено
        if step_count % save_interval != 0:
            current_state = integrator.y
            current_positions = current_state[:N * 3].reshape(N, 3)
            current_velocities = current_state[N * 3:].reshape(N, 3)

            for i, nucleon in enumerate(self.nucleons):
                nucleon.position = current_positions[i]
                nucleon.velocity = current_velocities[i]

            self.times.append(integrator.t)
            self.trajectories.append(current_positions.copy())
            self.velocities_history.append(current_velocities.copy())

        for i, nucleon in enumerate(self.nucleons):
            nucleon.cluster_id = initial_cluster_ids[i]

        end_time = time.time()
        print(f"Симуляция завершена за {end_time - start_time:.2f} секунд")
        print(f"Выполнено {step_count} шагов, сохранено {len(self.times)} состояний")

        return self.times, self.trajectories, self.velocities_history

    def create_animation(self, filename=None, fps=1, limit = 10):
        """
        Создание анимации столкновения с сохранением исходных цветов кластеров
        
        Args:
            filename (str): Имя файла для сохранения анимации
            fps (int): Кадров в секунду
        """
        if not self.trajectories:
            print("Ошибка: нет данных для анимации")
            return None

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        unique_cluster_ids = np.unique([n.cluster_id for n in self.nucleons])
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_cluster_ids)))
        cluster_colors = {cid: colors[i] for i, cid in enumerate(unique_cluster_ids)}

        trajectories_history = [[] for _ in range(len(self.nucleons))]

        def update(frame):
            ax.clear()

            positions = self.trajectories[frame]

            for i in range(len(self.nucleons)):
                trajectories_history[i].append(positions[i])

            for i, nucleon in enumerate(self.nucleons):
                ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2],
                           c=[cluster_colors[nucleon.cluster_id]], s=50, alpha=0.8)

                if len(trajectories_history[i]) > 1:
                    traj = np.array(trajectories_history[i])
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                            c=cluster_colors[nucleon.cluster_id], alpha=0.3)

            ax.set_title(f't = {self.times[frame]:.5f}')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(-limit, limit)

            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(self.times), interval=1000 / fps, blit=False)

        if filename:
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(filename, writer=writer)
            print(f"Анимация сохранена в {filename}")

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
