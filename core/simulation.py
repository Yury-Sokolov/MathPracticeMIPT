import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from .cluster import Cluster


class Simulation:
    def __init__(self, potential, t_end, device='cuda', Imax=0.1, tau_max=0.01,
                 dt_min=1e-10, adaptive_dt=True, clustering_algorithm=None,
                 neural_network: nn.Module | None = None):
        self.potential = potential
        self.neural_network = neural_network
        self.Imax = Imax
        self.tau_max = tau_max
        self.dt_min = dt_min
        self.t_end = t_end
        self.device = device
        self.eta = 0.1
        self.adaptive_dt = adaptive_dt
        self.clustering_algorithm = clustering_algorithm
        self.clusters = []
        self.nucleons = {
            'positions': None,
            'velocities': None,
            'masses': None,
            'cluster_ids': None
        }

        self.times = []
        self.trajectories = []
        self.velocities_history = []
        
        can_compile = True


        if can_compile:
            try:
                if self.adaptive_dt:
                    if not hasattr(self.potential, 'compute_scalar_derivatives'):
                        print("Warning: Adaptive timestep requires potential.compute_scalar_derivatives. Disabling.")
                        self.adaptive_dt = False
                        self._compute_adaptive_dt_compiled = self._compute_adaptive_dt_impl
                    else:
                        self._compute_adaptive_dt_compiled = torch.compile(self._compute_adaptive_dt_impl)
                else:
                    self._compute_adaptive_dt_compiled = self._compute_adaptive_dt_impl

                self._update_positions_velocities = torch.compile(self._update_positions_velocities_impl)
                print("Simulation steps successfully compiled.")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}). Falling back to eager execution.")
                self._compute_adaptive_dt_compiled = self._compute_adaptive_dt_impl
                self._update_positions_velocities = self._update_positions_velocities_impl
        else:
            print("Note: torch.compile skipped (compilation disabled).")
            self._compute_adaptive_dt_compiled = self._compute_adaptive_dt_impl
            self._update_positions_velocities = self._update_positions_velocities_impl

    def add_cluster(self, cluster):
        """
        Добавление кластера в симуляцию
        
        Args:
            cluster (Cluster): Объект кластера для добавления
        """
        positions = cluster.positions
        velocities = cluster.velocities
        masses = cluster.masses
        
        cluster_id = len(self.clusters)
        self.clusters.append(cluster)
        
        cluster_ids = torch.full((positions.shape[0],), cluster_id, 
                                 dtype=torch.long, device=self.device)
        
        if self.nucleons['positions'] is None:
            self.nucleons['positions'] = positions.to(self.device)
            self.nucleons['velocities'] = velocities.to(self.device)
            self.nucleons['masses'] = masses.to(self.device)
            self.nucleons['cluster_ids'] = cluster_ids
        else:
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
            self.nucleons['cluster_ids'] = torch.cat([
                self.nucleons['cluster_ids'],
                cluster_ids
            ], dim=0)

    def _compute_nn_forces(self, positions):
        """
        Computes forces between particles using the neural network.
        Handles both single instance [N, 3] and batched [B, N, 3] input.
        """
        is_batched = positions.dim() == 3
        if not is_batched:
            positions = positions.unsqueeze(0)

        batch_size, n_particles, _ = positions.shape
        total_forces = torch.zeros_like(positions)

        indices_j, indices_i = torch.triu_indices(n_particles, n_particles, offset=1, device=self.device)
        num_pairs = indices_i.shape[0]

        if num_pairs == 0:
            return total_forces.squeeze(0) if not is_batched else total_forces, None, None 

        pos_i = positions[:, indices_i, :] 
        pos_j = positions[:, indices_j, :] 
        relative_pos_ij_batch = pos_i - pos_j

        relative_pos_ij_flat = relative_pos_ij_batch.reshape(-1, 3)
        
        forces_ij_flat = self.neural_network(relative_pos_ij_flat)

        forces_ij_batch = forces_ij_flat.reshape(batch_size, num_pairs, 3)

        batch_indices_i = indices_i.unsqueeze(0).expand(batch_size, -1)
        batch_indices_j = indices_j.unsqueeze(0).expand(batch_size, -1)

        total_forces = total_forces.scatter_add_(1, batch_indices_i.unsqueeze(-1).expand(-1, -1, 3), forces_ij_batch)
        total_forces = total_forces.scatter_add_(1, batch_indices_j.unsqueeze(-1).expand(-1, -1, 3), -forces_ij_batch)
        
        if not is_batched:
            total_forces = total_forces.squeeze(0)

        return total_forces, None, None

    def compute_forces(self, positions):
        f_prime, f_double_prime = None, None

        if self.neural_network is not None:
            forces, _, _ = self._compute_nn_forces(positions)
        elif self.potential is not None:
            if self.adaptive_dt:
                forces, f_prime, f_double_prime = self.potential.compute_derivatives(positions)
            else:
                forces = self.potential.compute_force_only(positions)
        else:
            raise ValueError("No force calculation method available (potential or NN).")

        acceleration = forces / self.nucleons['masses'].unsqueeze(1)

        return acceleration, f_prime, f_double_prime

    def _compute_adaptive_dt_impl(self, velocities, f, f_prime, f_double_prime):
        """
        Оптимизированная реализация вычисления адаптивного шага времени
        """
        positions = self.nucleons['positions']
        masses = self.nucleons['masses']

        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        r = torch.norm(r_ij, dim=2)  # [N, N]
        
        mask = (r > 1e-10) & (r < self.potential.r_cutoff)
        
        valid_pairs = torch.triu(mask, diagonal=1)
        i_indices, j_indices = torch.where(valid_pairs)
        
        if len(i_indices) == 0:
            return self.tau_max
        
        v_ij = velocities[i_indices] - velocities[j_indices]  # [pairs, 3]
        v = torch.norm(v_ij, dim=1)  # [pairs]
        
        dot_rv = torch.sum(r_ij[i_indices, j_indices] * v_ij, dim=1)  # [pairs]
        
        m_i = masses[i_indices]
        m_j = masses[j_indices]
        mu = (m_i * m_j) / (m_i + m_j)  # [pairs]
        
        r_pairs = r[i_indices, j_indices]  # [pairs]
        directions = r_ij[i_indices, j_indices] / r_pairs.unsqueeze(-1)  # [pairs, 3]
        
        f_diff = f[i_indices] - f[j_indices]  # [pairs, 3]
        f_ij = torch.norm(f_diff, dim=1)  # [pairs]
        
        f_prime_r_ij = torch.zeros(len(i_indices), device=self.device)
        f_double_prime_r_ij = torch.zeros(len(i_indices), device=self.device)
        S = torch.zeros(len(i_indices), device=self.device)
        
        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx]
            direction = directions[idx]
            
            f_prime_r_ij_val = torch.tensor(0.0, device=self.device)
            f_double_prime_r_ij_val = torch.tensor(0.0, device=self.device)
            
            for a in range(3):
                for b in range(3):
                    f_prime_r_ij_val += f_prime[i, j, a, b] * direction[a] * direction[b]
                    
                    for c in range(3):
                        f_double_prime_r_ij_val += f_double_prime[i, a, b, c] * direction[a] * direction[b] * direction[c]
            
            f_prime_r_ij[idx] = f_prime_r_ij_val
            f_double_prime_r_ij[idx] = f_double_prime_r_ij_val
            
            term1 = (dot_rv[idx]**3 / r_pairs[idx]**5) * (
                r_pairs[idx] * f_prime_r_ij[idx] - f_ij[idx] - (r_pairs[idx]**2 * f_double_prime_r_ij[idx]) / 3
            )
            
            term2 = (dot_rv[idx] / (mu[idx] * r_pairs[idx]**3)) * (
                mu[idx] * v[idx]**2 * (f_ij[idx] - r_pairs[idx] * f_prime_r_ij[idx]) - 
                r_pairs[idx]**2 * f_ij[idx] * f_prime_r_ij[idx]
            )
            
            S[idx] = term1 + term2
        
        tau_ij = torch.ones_like(S) * self.tau_max
        
        nonzero_S = torch.abs(S) > 1e-10
        if torch.any(nonzero_S):
            tau_ij[nonzero_S] = 2 * torch.sqrt(self.Imax / torch.abs(S[nonzero_S]))
        
        min_tau = torch.min(tau_ij)
        
        dt = torch.clamp(min_tau, self.dt_min, self.tau_max)
        
        return dt

    def compute_adaptive_dt(self, velocities, f, f_prime, f_double_prime):
        """
        Вычисление адаптивного шага времени на основе динамики системы
        """
        if f_prime is None or f_double_prime is None:
            return self.tau_max
        
        return self._compute_adaptive_dt_compiled(velocities, f, f_prime, f_double_prime)

    def _update_positions_velocities_impl(self, positions, velocities, a_prev, dt):
        """
        Оптимизированная реализация обновления позиций и скоростей
        """
        new_positions = positions + velocities * dt + 0.5 * a_prev * dt ** 2
        forces, f_prime, f_double_prime = self.compute_forces(new_positions)
        a_new = forces
        new_velocities = velocities + 0.5 * (a_prev + a_new) * dt
        
        return new_positions, new_velocities, a_new, f_prime, f_double_prime

    def run(self, save_interval, dt_initial, max_steps):
        positions = self.nucleons['positions']
        velocities = self.nucleons['velocities']
        
        self.times = [0.0]
        self.trajectories = [positions.cpu().detach().numpy()]
        self.velocities_history = [velocities.cpu().detach().numpy()]

        t = 0.0
        dt = dt_initial
        step_count = 0

        forces, f_prime, f_double_prime = self.compute_forces(positions)
        a_prev = forces

        # with tqdm(total=max_steps) as pb:
        while t < self.t_end and step_count < max_steps:
            positions, velocities, a_new, f_prime, f_double_prime = self._update_positions_velocities(
                positions, velocities, a_prev, dt
            )

            if self.adaptive_dt:
                dt = self.compute_adaptive_dt(velocities, a_new, f_prime, f_double_prime)

            t += dt
            step_count += 1

            if step_count % save_interval == 0:
                self.times.append(t)
                self.trajectories.append(positions.cpu().detach().numpy())
                self.velocities_history.append(velocities.cpu().detach().numpy())

                # pb.update(save_interval)
                # pb.set_description(
                #     f"t={t:.3f}, dt={dt:.3e}, progress={100 * t / self.t_end:.1f}%"
                # )

            a_prev = a_new

        self.nucleons['positions'] = positions
        self.nucleons['velocities'] = velocities


        run_times = torch.tensor(self.times, dtype=torch.float32).detach()
        run_positions = torch.from_numpy(np.array(self.trajectories)).float().detach()
        run_velocities = torch.from_numpy(np.array(self.velocities_history)).float().detach()
        run_masses = self.nucleons['masses'].cpu().detach()
        
        results = {
            'times': run_times,
            'positions': run_positions,
            'velocities': run_velocities,
            'masses': run_masses
        }
        return results

    def get_cluster_data(self):
        """
        Получение данных о кластерах (положение и скорость центра масс)
        
        Returns:
            dict: Словарь с данными о кластерах
        """
        if not self.trajectories or not self.velocities_history:
            print("Ошибка: нет данных о траекториях или скоростях")
            return None
            
        final_positions = torch.tensor(self.trajectories[-1], device=self.device)
        final_velocities = torch.tensor(self.velocities_history[-1], device=self.device)
        masses = self.nucleons['masses']
        
        if self.clustering_algorithm is not None:
            positions_np = final_positions.cpu().numpy()
            
            cluster_labels = self.clustering_algorithm.fit_predict(positions_np)
            
            cluster_ids = torch.tensor(cluster_labels, device=self.device)
        else:
            cluster_ids = self.nucleons['cluster_ids']
        
        unique_cluster_ids = torch.unique(cluster_ids)
        if -1 in unique_cluster_ids:
            unique_cluster_ids = unique_cluster_ids[unique_cluster_ids != -1]
        
        cluster_data = {
            'positions': [],
            'velocities': [],
            'masses': [],
            'sizes': [],
            'nucleon_indices': [],
            'cluster_labels': cluster_ids.cpu().numpy()
        }
        
        for cluster_id in unique_cluster_ids:
            mask = cluster_ids == cluster_id
            cluster_nucleons = final_positions[mask]
            cluster_velocities = final_velocities[mask]
            cluster_masses = masses[mask]
            
            if torch.sum(mask) > 0:
                total_mass = torch.sum(cluster_masses)
                com_position = torch.sum(cluster_nucleons * cluster_masses.unsqueeze(1), dim=0) / total_mass
                com_velocity = torch.sum(cluster_velocities * cluster_masses.unsqueeze(1), dim=0) / total_mass
                
                cluster_data['positions'].append(com_position.cpu().numpy())
                cluster_data['velocities'].append(com_velocity.cpu().numpy())
                cluster_data['masses'].append(total_mass.cpu().numpy())
                cluster_data['sizes'].append(torch.sum(mask).cpu().numpy())
                cluster_data['nucleon_indices'].append(torch.where(mask)[0].cpu().numpy())
        
        for key in ['positions', 'velocities', 'masses', 'sizes']:
            cluster_data[key] = np.array(cluster_data[key])
            
        return cluster_data

    @staticmethod
    def analyze_multiple_results(results):
        """
        Анализ результатов множества симуляций
        
        Args:
            results: Список результатов симуляций
            
        Returns:
            dict: Словарь с гистограммами распределений
        """
        if not results:
            print("Ошибка: нет данных для анализа")
            return None
            
        all_masses = []
        all_momenta = []
        all_sizes = []
        
        for result in results:
            if result is None:
                continue
                
            masses = result['masses']
            velocities = result['velocities']
            
            momenta = masses[:, np.newaxis] * velocities
            momenta_magnitudes = np.linalg.norm(momenta, axis=1)
            
            all_masses.extend(masses)
            all_momenta.extend(momenta_magnitudes)
            all_sizes.extend(result['sizes'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].hist(all_masses, bins=50, alpha=0.7)
        axes[0].set_title('Распределение масс кластеров')
        axes[0].set_xlabel('Масса')
        axes[0].set_ylabel('Количество')
        
        axes[1].hist(all_momenta, bins=50, alpha=0.7)
        axes[1].set_title('Распределение импульсов кластеров')
        axes[1].set_xlabel('Импульс')
        axes[1].set_ylabel('Количество')
        
        axes[2].hist(all_sizes, bins=range(1, max(all_sizes) + 2), alpha=0.7)
        axes[2].set_title('Распределение размеров кластеров')
        axes[2].set_xlabel('Количество нуклонов')
        axes[2].set_ylabel('Количество')
        
        plt.tight_layout()
        
        return {
            'masses': np.array(all_masses),
            'momenta': np.array(all_momenta),
            'sizes': np.array(all_sizes),
            'figure': fig
        }

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

        trajectories = np.asarray(self.trajectories)
        frames_count = len(self.times)
        n_particles = trajectories[0].shape[0]
        
        cluster_ids = self.nucleons['cluster_ids'].cpu().numpy()
        unique_clusters = np.unique(cluster_ids)
        cluster_colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))

        colors = np.zeros((n_particles, 4))
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_ids == cluster_id
            colors[mask] = cluster_colors[i]
        
        scatter = ax.scatter([], [], [], s=50, alpha=0.8)
        lines = [ax.plot([], [], [], c=colors[i], alpha=0.3)[0] for i in range(n_particles)]

        scatter.set_facecolors(colors)
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

    def reset(self):
        """
        Сброс состояния симуляции для повторного использования
        
        Сохраняет настройки, потенциал и алгоритм кластеризации, 
        но очищает данные о траекториях, кластерах и нуклонах
        """
        self.clusters = []
        self.nucleons = {
            'positions': None,
            'velocities': None,
            'masses': None,
            'cluster_ids': None
        }
        
        self.times = []
        self.trajectories = []
        self.velocities_history = []

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

        if save_path:
            plt.savefig(save_path)
            print(f"График кластеризации сохранен в {save_path}")

        plt.show()

        return cluster_labels

    def compute_kinetic_energy_per_nucleon(self):
        """
        Вычисление кинетической энергии для каждого нуклона
        
        Returns:
            np.ndarray: Массив кинетических энергий каждого нуклона
        """
        if not self.velocities_history:
            print("Ошибка: нет данных о скоростях")
            return None
        
        velocities = self.velocities_history[-1]
        masses = self.nucleons['masses'].cpu().numpy()
        
        velocities_squared = np.sum(velocities**2, axis=1)
        
        kinetic_energies = 0.5 * masses * velocities_squared
        
        return kinetic_energies

    def compute_potential_energy_per_nucleon(self):
        """
        Вычисление потенциальной энергии для каждого нуклона
        
        Returns:
            np.ndarray: Массив потенциальных энергий каждого нуклона
        """
        if not self.trajectories:
            print("Ошибка: нет данных о позициях")
            return None
        
        positions = torch.tensor(self.trajectories[-1], device=self.device)
        potential_energies = self.potential.compute_energy_per_particle(positions)
        
        return potential_energies.cpu().numpy()

    def compute_total_energy_per_nucleon(self):
        """
        Вычисление полной энергии для каждого нуклона
        
        Returns:
            tuple: (массив кинетических энергий, массив потенциальных энергий, 
                   массив полных энергий)
        """
        kinetic_energies = self.compute_kinetic_energy_per_nucleon()
        potential_energies = self.compute_potential_energy_per_nucleon()
        
        if kinetic_energies is None or potential_energies is None:
            return None, None, None
        
        total_energies = kinetic_energies + potential_energies
        
        return kinetic_energies, potential_energies, total_energies

    def setup_impact_parameter(self, nucleus_count1=20, nucleus_count2=20, impact_parameter=0.0, 
                              relative_velocity=20.0, random_velocity=3.0, radius1=1.0, radius2=1.0):
        """
        Настройка симуляции для моделирования столкновения с заданным прицельным параметром
        
        Args:
            nucleus_count1: Количество нуклонов в первом кластере
            nucleus_count2: Количество нуклонов во втором кластере
            impact_parameter: Прицельный параметр (расстояние между центрами масс в плоскости xy)
            relative_velocity: Относительная скорость кластеров
            random_velocity: Величина случайной скорости для нуклонов в кластерах
            radius1: Радиус первого кластера
            radius2: Радиус второго кластера
            
        Returns:
            self: Возвращает сам объект для chain-вызовов
        """
        self.reset()
        
        device = self.device
        
        relative_velocity = max(relative_velocity, 5.0)
        
        separation = max(radius1 + radius2, 4.0)
        

        collision_point = torch.tensor([0.0, impact_parameter/2.0, 0.0], device=device)
        

        pos1 = torch.tensor([-separation, 0.0, 0.0], device=device)
        pos2 = torch.tensor([separation, impact_parameter, 0.0], device=device)
        
        cluster1 = Cluster(
            position=pos1,
            velocity=torch.tensor([0.0, 0.0, 0.0], device=device),
            radius=radius1,
            random_velocity=random_velocity,
            N=nucleus_count1,
            device=device
        )
        
        cluster2 = Cluster(
            position=pos2,
            velocity=torch.tensor([0.0, 0.0, 0.0], device=device),
            radius=radius2,
            N=nucleus_count2,
            random_velocity=random_velocity,
            device=device
        )
        
        if random_velocity > 0:
            cluster1.add_random_velocity(random_velocity)
            cluster2.add_random_velocity(random_velocity)
        
        m1 = torch.sum(cluster1.masses)
        m2 = torch.sum(cluster2.masses)
        total_mass = m1 + m2
        
        dir1 = collision_point - pos1
        dir2 = collision_point - pos2
        

        dir1_norm = torch.norm(dir1)
        dir2_norm = torch.norm(dir2)
        
        if dir1_norm > 0:
            dir1 = dir1 / dir1_norm
        
        if dir2_norm > 0:
            dir2 = dir2 / dir2_norm
        
        v1_mag = relative_velocity * (m2 / total_mass)
        v2_mag = relative_velocity * (m1 / total_mass)
        

        vel1 = dir1 * v1_mag
        vel2 = dir2 * v2_mag
        

        cluster1.add_velocity(vel1)
        cluster2.add_velocity(vel2)
        
        self.add_cluster(cluster1)
        self.add_cluster(cluster2)
        
        return self
            
    def run_multiple_collisions(self, count, nucleus_count1=20, nucleus_count2=20, velocity=20.0,
                               max_impact_parameter=5.0, save_interval=1, dt_initial=0.00001, 
                               max_steps=1000, random_velocity=3.0, radius1=1.0, radius2=1.0):
        """
        Запуск множества симуляций столкновений с различными прицельными параметрами
        
        Args:
            count: Количество симуляций
            nucleus_count1: Количество нуклонов в первом кластере
            nucleus_count2: Количество нуклонов во втором кластере
            velocity: Относительная скорость кластеров
            max_impact_parameter: Максимальный прицельный параметр
            save_interval: Интервал сохранения состояний
            dt_initial: Начальный шаг по времени
            max_steps: Максимальное количество шагов
            random_velocity: Величина случайной скорости для нуклонов в кластерах
            radius1: Радиус первого кластера
            radius2: Радиус второго кластера
            
        Returns:
            list: Список результатов симуляций
        """
        results = []
        
        random_values = np.random.random(count)
        impact_parameters = np.sqrt(random_values) * max_impact_parameter
        
        for i, b in enumerate(tqdm(impact_parameters, desc="Running simulations")):
            self.setup_impact_parameter(
                nucleus_count1=nucleus_count1, 
                nucleus_count2=nucleus_count2, 
                impact_parameter=b, 
                relative_velocity=velocity, 
                random_velocity=random_velocity,
                radius1=radius1,
                radius2=radius2
            )
            
            result = self.run(save_interval, dt_initial, max_steps)
            results.append(result)
        
        return results

    def compute_energy(self):
        """
        Вычисляет полную энергию системы (кинетическую + потенциальную)
        
        Returns:
            Тензор энергии системы
        """
        velocities = self.nucleons['velocities']
        masses = self.nucleons['masses']
        
        kinetic_energy = 0.5 * torch.sum(masses.unsqueeze(1) * torch.sum(velocities**2, dim=-1))
        
        potential_energy = torch.tensor(0.0, device=self.device)
        positions = self.nucleons['positions']
        
        if self.potential is not None and hasattr(self.potential, 'compute_energy_per_particle'):
            particle_potential = self.potential.compute_energy_per_particle(positions)
            potential_energy = torch.sum(particle_potential)
        
        elif self.neural_network is not None:
            n_particles = positions.shape[0]
            
            indices_i, indices_j = torch.triu_indices(n_particles, n_particles, offset=1)
            indices_i = indices_i.to(self.device)
            indices_j = indices_j.to(self.device)
            
            if len(indices_i) > 0:
                pos_i = positions[indices_i]
                pos_j = positions[indices_j]
                r_ij = pos_i - pos_j
                
                with torch.no_grad():
                    pair_potentials = self.neural_network.compute_potential(r_ij)
                    potential_energy = torch.sum(pair_potentials)
        
        return kinetic_energy + potential_energy
