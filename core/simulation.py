import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch


class Simulation:
    def __init__(self, potential, t_end, device='cuda', Imax=0.1, tau_max=0.01,
                 dt_min=1e-10):
        self.potential = potential
        self.Imax = Imax
        self.tau_max = tau_max
        self.dt_min = dt_min
        self.t_end = t_end
        self.device = device
        self.eta = 0.1
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
        
        self._compute_adaptive_dt_compiled = torch.compile(self._compute_adaptive_dt_impl)
        self._update_positions_velocities = torch.compile(self._update_positions_velocities_impl)

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

    def compute_forces(self, positions):
        forces, f_prime, f_double_prime = self.potential.compute_derivatives(positions)
        return forces / self.nucleons['masses'].unsqueeze(1), f_prime, f_double_prime

    def _compute_adaptive_dt_impl(self, velocities, f, f_prime, f_double_prime):
        """
        Оптимизированная реализация вычисления адаптивного шага времени
        """
        positions = self.nucleons['positions']
        masses = self.nucleons['masses']
        N = positions.shape[0]
        
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        r = torch.norm(r_ij, dim=2)  # [N, N]
        
        v_ij = velocities.unsqueeze(1) - velocities.unsqueeze(0)  # [N, N, 3]
        v = torch.norm(v_ij, dim=2)  # [N, N]
        
        dot_rv = torch.sum(r_ij * v_ij, dim=2)  # [N, N]
        
        mask = (r > 1e-10) & (r < self.potential.r1)
        
        m_i = masses.unsqueeze(1)
        m_j = masses.unsqueeze(0)
        mu = (m_i * m_j) / (m_i + m_j)
        
        nonzero_r = r.unsqueeze(-1).expand_as(r_ij)
        nonzero_r = torch.where(nonzero_r > 1e-10, nonzero_r, torch.ones_like(nonzero_r))
        directions = r_ij / nonzero_r
        
        f_diff = f.unsqueeze(1) - f.unsqueeze(0)  # [N, N, 3]
        f_ij = torch.norm(f_diff, dim=2)  # [N, N]
        
        f_prime_r_ij = torch.zeros((N, N), device=self.device)
        f_double_prime_r_ij = torch.zeros((N, N), device=self.device)
        S = torch.zeros_like(r)
        
        valid_pairs = torch.triu(mask, diagonal=1)
        i_indices, j_indices = torch.where(valid_pairs)
        
        if len(i_indices) > 0:
            for idx in range(len(i_indices)):
                i, j = i_indices[idx], j_indices[idx]
                direction = directions[i, j]
                
                f_prime_r_ij_val = 0.0
                f_double_prime_r_ij_val = 0.0
                
                for a in range(3):
                    for b in range(3):
                        f_prime_r_ij_val += f_prime[i, a, b] * direction[a] * direction[b]
                        
                        for c in range(3):
                            f_double_prime_r_ij_val += f_double_prime[i, a, b, c] * direction[a] * direction[b] * direction[c]
                
                f_prime_r_ij[i, j] = f_prime_r_ij_val
                f_prime_r_ij[j, i] = f_prime_r_ij_val
                f_double_prime_r_ij[i, j] = f_double_prime_r_ij_val
                f_double_prime_r_ij[j, i] = f_double_prime_r_ij_val
                
                term1_ij = (dot_rv[i, j]**3 / r[i, j]**5) * (
                    r[i, j] * f_prime_r_ij[i, j] - f_ij[i, j] - (r[i, j]**2 * f_double_prime_r_ij[i, j]) / 3
                )
                
                term2_ij = (dot_rv[i, j] / (mu[i, j] * r[i, j]**3)) * (
                    mu[i, j] * v[i, j]**2 * (f_ij[i, j] - r[i, j] * f_prime_r_ij[i, j]) - 
                    r[i, j]**2 * f_ij[i, j] * f_prime_r_ij[i, j]
                )
                
                S[i, j] = term1_ij + term2_ij
                S[j, i] = S[i, j]
        
        tau_ij = torch.ones_like(r) * self.tau_max
        
        nonzero_S = (torch.abs(S) > 1e-10) & mask
        if torch.any(nonzero_S):
            tau_ij[nonzero_S] = 2 * torch.sqrt(self.Imax / torch.abs(S[nonzero_S]))
        
        min_tau = torch.min(tau_ij + torch.eye(N, device=self.device) * self.tau_max)
        
        dt = torch.clamp(min_tau, self.dt_min, self.tau_max)
        
        return dt

    def compute_adaptive_dt(self, velocities, f, f_prime, f_double_prime):
        """
        Вычисление адаптивного шага времени на основе динамики системы
        """
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

        with tqdm(total=max_steps) as pb:
            while t < self.t_end and step_count < max_steps:
                positions, velocities, a_new, f_prime, f_double_prime = self._update_positions_velocities(
                    positions, velocities, a_prev, dt
                )
                
                dt = self.compute_adaptive_dt(velocities, a_new, f_prime, f_double_prime)

                t += dt
                step_count += 1

                if step_count % save_interval == 0:
                    self.times.append(t)
                    self.trajectories.append(positions.cpu().detach().numpy())
                    self.velocities_history.append(velocities.cpu().detach().numpy())

                    pb.update(save_interval)
                    pb.set_description(
                        f"t={t:.3f}, dt={dt:.3e}, progress={100 * t / self.t_end:.1f}%"
                    )

                a_prev = a_new

        self.nucleons['positions'] = positions
        self.nucleons['velocities'] = velocities

        print(f"Simulation completed at t={t:.3f}")
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
        
        velocities = self.velocities_history[-1]  # Последние скорости
        masses = self.nucleons['masses'].cpu().numpy()
        
        # Вычисление квадрата скорости для каждого нуклона
        velocities_squared = np.sum(velocities**2, axis=1)
        
        # Вычисление кинетической энергии
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
