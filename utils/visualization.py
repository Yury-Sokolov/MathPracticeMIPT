import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import tqdm


def plot_trajectory_comparison(
    predicted_positions: torch.Tensor,
    true_positions: torch.Tensor,
    noisy_positions: Optional[torch.Tensor] = None,
    times: Optional[torch.Tensor] = None,
    num_particles: int = 5,
    save_path: Optional[str] = None,
    title: str = "UDE Predicted Trajectory vs True Trajectory"
) -> plt.Figure:
    """
    Строит сравнение траекторий: предсказанные, истинные, шумные.
    
    Args:
        predicted_positions: Предсказанные позиции [n_steps, n_particles, 3]
        true_positions: Истинные позиции [n_steps, n_particles, 3]
        noisy_positions: Шумные позиции [n_steps, n_particles, 3]
        times: Временные точки [n_steps]
        num_particles: Количество частиц для отображения
        save_path: Путь для сохранения графика. Если None, график не сохраняется.
        title: Заголовок графика
        
    Returns:
        Объект Figure с построенным графиком
    """
    if predicted_positions.shape[1] < num_particles:
        num_particles = predicted_positions.shape[1]
    
    n_steps = predicted_positions.shape[0]
    time_indices = np.linspace(0, n_steps-1, num=min(n_steps, 200), dtype=int)
    
    # Преобразование в numpy
    pred_pos_np = predicted_positions.cpu().numpy()
    true_pos_np = true_positions.cpu().numpy()
    
    if times is not None:
        time_points = times[time_indices].cpu().numpy()
    else:
        time_points = np.arange(len(time_indices))
    
    fig = plt.figure(figsize=(12, 8))
    
    for i in range(num_particles):
        # X-координаты
        plt.plot(time_points, pred_pos_np[time_indices, i, 0], 'r--', alpha=0.8, 
                 label=f'UDE Pred (P{i} X)' if i==0 else None)
        plt.plot(time_points, true_pos_np[time_indices, i, 0], 'r-', alpha=0.6, 
                 label=f'True (P{i} X)' if i==0 else None)
        
        # Y-координаты
        plt.plot(time_points, pred_pos_np[time_indices, i, 1], 'b--', alpha=0.8, 
                 label=f'UDE Pred (P{i} Y)' if i==0 else None)
        plt.plot(time_points, true_pos_np[time_indices, i, 1], 'b-', alpha=0.6, 
                 label=f'True (P{i} Y)' if i==0 else None)
        
        # Шумные данные
        if noisy_positions is not None:
            noisy_pos_np = noisy_positions.cpu().numpy()
            plt.scatter(time_points, noisy_pos_np[time_indices, i, 0], c='red', marker='.', s=10, alpha=0.3, 
                        label='Noisy Data X' if i==0 else None)
            plt.scatter(time_points, noisy_pos_np[time_indices, i, 1], c='blue', marker='.', s=10, alpha=0.3, 
                        label='Noisy Data Y' if i==0 else None)

    plt.xlabel("Time")
    plt.ylabel("Position (X=Red, Y=Blue)")
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig


def plot_force_comparison(
    distances: torch.Tensor,
    learned_forces: torch.Tensor,
    true_forces: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Comparison of Learned Force vs True Potential Force"
) -> plt.Figure:
    """
    Строит сравнение моделируемых и истинных сил.
    
    Args:
        distances: Расстояния [n_points]
        learned_forces: Моделируемые силы [n_points, 3]
        true_forces: Истинные силы [n_points, 3]
        save_path: Путь для сохранения графика. Если None, график не сохраняется.
        title: Заголовок графика
        
    Returns:
        Объект Figure с построенным графиком
    """
    fig = plt.figure(figsize=(10, 6))
    
    distances_np = distances.cpu().numpy()
    learned_forces_np = learned_forces[:, 0].cpu().numpy()  # X-компонента
    true_forces_np = true_forces[:, 0].cpu().numpy()  # X-компонента
    
    plt.plot(distances_np, true_forces_np, 'k-', label='True Potential Force (Fx)')
    plt.plot(distances_np, learned_forces_np, 'r--', label='Learned Neural Force (Fx)')

    plt.xlabel("Distance (r)")
    plt.ylabel("Force component Fx")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig


def plot_potential_comparison(
    distances: torch.Tensor,
    learned_potentials: torch.Tensor,
    true_potentials: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Comparison of Learned Potential vs True Potential"
) -> plt.Figure:
    """
    Строит сравнение моделируемого и истинного потенциалов.
    
    Args:
        distances: Расстояния [n_points]
        learned_potentials: Моделируемые потенциалы [n_points]
        true_potentials: Истинные потенциалы [n_points]
        save_path: Путь для сохранения графика. Если None, график не сохраняется.
        title: Заголовок графика
        
    Returns:
        Объект Figure с построенным графиком
    """
    fig = plt.figure(figsize=(10, 6))
    
    distances_np = distances.cpu().numpy()
    learned_potentials_np = learned_potentials.cpu().numpy()
    true_potentials_np = true_potentials.cpu().numpy()
    
    plt.plot(distances_np, true_potentials_np, 'k-', label='True Potential')
    plt.plot(distances_np, learned_potentials_np, 'g--', label='Learned Neural Potential')

    plt.xlabel("Distance (r)")
    plt.ylabel("Potential Energy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training and Validation Loss"
) -> plt.Figure:
    """
    Строит график обучения (потери).
    
    Args:
        train_losses: Потери на обучающей выборке
        val_losses: Потери на валидационной выборке
        save_path: Путь для сохранения графика. Если None, график не сохраняется.
        title: Заголовок графика
        
    Returns:
        Объект Figure с построенным графиком
    """
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig


def plot_force_profile_evolution(
    distances: np.ndarray,
    force_profiles: List[np.ndarray],
    true_forces: np.ndarray,
    epochs: List[int],
    save_path: Optional[str] = None,
    title: str = "Evolution of Force Profile During Training"
) -> plt.Figure:
    """
    Строит эволюцию профиля сил во время обучения.
    
    Args:
        distances: Расстояния [n_points]
        force_profiles: Список профилей сил в разные эпохи [n_epochs, n_points]
        true_forces: Истинные силы [n_points]
        epochs: Номера эпох для профилей
        save_path: Путь для сохранения графика. Если None, график не сохраняется.
        title: Заголовок графика
        
    Returns:
        Объект Figure с построенным графиком
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Истинные силы
    plt.plot(distances, true_forces, 'k-', linewidth=3, label='True Force')
    
    # Эволюция моделируемых сил
    cmap = plt.cm.viridis
    for i, (epoch, profile) in enumerate(zip(epochs, force_profiles)):
        color = cmap(i / len(epochs))
        plt.plot(distances, profile, '--', color=color, alpha=0.7, label=f'Epoch {epoch+1}')
    
    plt.xlabel('Distance (r)')
    plt.ylabel('Force')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig


def create_collision_animation(
    positions: torch.Tensor,
    times: torch.Tensor,
    filename: str,
    fps: int = 30,
    limit: float = 10.0,
    title: str = "Collision Animation"
) -> None:
    """
    Создает анимацию столкновения.
    
    Args:
        positions: Позиции частиц в каждый момент времени [n_steps, n_particles, 3]
        times: Временные точки [n_steps]
        filename: Имя файла для сохранения анимации
        fps: Кадров в секунду
        limit: Ограничение по осям
        title: Заголовок анимации
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    positions_np = positions.cpu().numpy()
    times_np = times.cpu().numpy()
    
    n_particles = positions_np.shape[1]
    colors = [cm.tab10(i % 10) for i in range(n_particles)]
    
    # Инициализация частиц
    particles = []
    for i in range(n_particles):
        particle, = ax.plot([], [], 'o', markersize=5, color=colors[i])
        particles.append(particle)
    
    # Текст для отображения времени
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        for particle in particles:
            particle.set_data([], [])
        time_text.set_text('')
        return particles + [time_text]
    
    def animate(i):
        for j, particle in enumerate(particles):
            particle.set_data(positions_np[i, j, 0], positions_np[i, j, 1])
        time_text.set_text(f'Time: {times_np[i]:.2f}')
        return particles + [time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions_np),
                         interval=1000/fps, blit=True)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)


def create_trajectory_animation(
    trajectories: np.ndarray,
    times: List[float],
    filename: Optional[str] = None,
    fps: int = 30,
    limit: float = 10.0,
    cluster_ids: Optional[np.ndarray] = None,
    title: str = "Trajectory Animation"
) -> animation.FuncAnimation:
    """
    Создает анимацию движения частиц на основе траекторий
    
    Args:
        trajectories: Массив траекторий формы [frames, n_particles, 3]
        times: Список временных точек для каждого кадра
        filename: Путь для сохранения анимации (если None, то не сохраняется)
        fps: Количество кадров в секунду
        limit: Границы пространства отображения
        cluster_ids: Массив с идентификаторами кластеров для каждой частицы
        title: Заголовок анимации
        
    Returns:
        Объект анимации
        
    Raises:
        ValueError: Если trajectories пустой массив или имеет неправильную форму
        IOError: Если не удалось сохранить анимацию
    """
    if len(trajectories) == 0:
        raise ValueError("Ошибка: нет данных для анимации")
    
    if len(trajectories) != len(times):
        raise ValueError(f"Несоответствие длин: trajectories ({len(trajectories)}) и times ({len(times)})")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    frames_count = len(times)
    n_particles = trajectories[0].shape[0]
    
    # Определение цветов для частиц на основе кластеров
    if cluster_ids is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_particles))
    else:
        unique_clusters = np.unique(cluster_ids)
        cluster_colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))
        colors = np.zeros((n_particles, 4))
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_ids == cluster_id
            colors[mask] = cluster_colors[i]
    
    scatter = ax.scatter([], [], [], s=50, alpha=0.8)
    lines = [ax.plot([], [], [], c=colors[i], alpha=0.3)[0] for i in range(n_particles)]

    scatter.set_facecolors(colors)
    pbar = tqdm(total=frames_count, desc="Creating animation")
    
    def update(frame):
        current_positions = trajectories[frame]
        scatter._offsets3d = current_positions.T
        pbar.update(1)
        
        for i in range(n_particles):
            x = trajectories[:frame + 1, i, 0]
            y = trajectories[:frame + 1, i, 1]
            z = trajectories[:frame + 1, i, 2]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)

        ax.set_title(f'{title} (t = {times[frame]:.5f})')
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
            print(f"Анимация сохранена в {filename}")
        except Exception as e:
            pbar.close()
            plt.close()
            raise IOError(f"Ошибка сохранения анимации: {str(e)}") from e
    
    pbar.close()
    plt.close()
    return ani


def visualize_cluster_analysis(
    positions: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Optional[str] = None,
    limit: float = 10.0,
    title: str = "Кластерный анализ"
) -> np.ndarray:
    """
    Визуализирует результаты кластерного анализа
    
    Args:
        positions: Позиции частиц [n_particles, 3]
        cluster_labels: Метки кластеров для каждой частицы
        save_path: Путь для сохранения графика
        limit: Границы отображения
        title: Заголовок графика
        
    Returns:
        Метки кластеров
        
    Raises:
        ValueError: Если positions или cluster_labels имеют неправильную форму
        IOError: Если не удалось сохранить изображение
    """
    if len(positions) == 0:
        raise ValueError("Ошибка: нет данных для кластеризации")
    
    if len(positions) != len(cluster_labels):
        raise ValueError(f"Несоответствие длин: positions ({len(positions)}) и cluster_labels ({len(cluster_labels)})")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            positions[mask, 2],
            c=[colors[i]],
            s=50,
            alpha=0.8,
            label=f'Кластер {label}'
        )

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"График кластеризации сохранен в {save_path}")
        except Exception as e:
            plt.close()
            raise IOError(f"Ошибка сохранения графика: {str(e)}") from e

    return cluster_labels


def analyze_multiple_collision_results(
    results: List[Dict],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Анализ результатов множества симуляций столкновений
    
    Args:
        results: Список результатов симуляций
        save_path: Путь для сохранения графика
        
    Returns:
        Словарь с статистиками и графиком
        
    Raises:
        ValueError: Если список results пуст
        IOError: Если не удалось сохранить график
    """
    if not results:
        raise ValueError("Ошибка: нет данных для анализа")
        
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
        all_sizes.extend(result.get('sizes', [1] * len(masses)))
    
    if not all_masses:
        raise ValueError("Ошибка: нет данных для анализа после фильтрации пустых результатов")
    
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
    axes[2].set_xlabel('Количество частиц')
    axes[2].set_ylabel('Количество')
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Анализ сохранен в {save_path}")
        except Exception as e:
            plt.close()
            raise IOError(f"Ошибка сохранения анализа: {str(e)}") from e
    
    return {
        'masses': np.array(all_masses),
        'momenta': np.array(all_momenta),
        'sizes': np.array(all_sizes),
        'figure': fig
    } 