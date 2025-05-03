import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# Добавляем корневую директорию в путь импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import Simulation
from potential.potential import MesonExchangePotential
from ude.ude_network import PotentialNN, KANPotentialModel
from utils.config import load_config, create_arg_parser
from utils.visualization import create_trajectory_animation, visualize_cluster_analysis, analyze_multiple_collision_results


def run_simulation(config):
    """
    Запускает симуляцию с обученной моделью UDE или аналитическим потенциалом.
    
    Args:
        config: Конфигурация с параметрами для симуляции
    """
    print("\n--- Запуск симуляции ---")
    start_time = time.time()
    
    output_dir = config.get('evaluation.analysis_output_dir', './analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Используется устройство: {device}")
    
    model_load_path = config.get('evaluation.model_load_path') or config.get('training.model_save_path')
    
    sim_params = {
        't_end': config.get('simulation.t_end', 15.0),
        'device': device,
        'dt_min': config.get('simulation.dt_min', 1e-16),
        'tau_max': config.get('simulation.tau_max', 0.01),
        'Imax': config.get('simulation.Imax', 0.1),
        'adaptive_dt': True
    }
    
    setup_params = {
        'nucleus_count1': config.get('simulation.n_particles', 4) // 2,
        'nucleus_count2': config.get('simulation.n_particles', 4) - config.get('simulation.n_particles', 4) // 2,
        'impact_parameter': config.get('simulation.max_impact_parameter', 3.0) * 0.5,
        'relative_velocity': config.get('simulation.relative_velocity', 1.0),
        'random_velocity': config.get('simulation.random_velocity', 0.1),
        'radius1': config.get('simulation.radius', 1.5),
        'radius2': config.get('simulation.radius', 1.5)
    }
    
    run_params = {
        'save_interval': config.get('data.save_interval', 1),
        'dt_initial': config.get('data.dt_initial', 0.001),
        'max_steps': config.get('evaluation.prediction_steps', 500)
    }
    
    clustering_algorithm = DBSCAN(eps=1.5, min_samples=2)
    
    use_ude = bool(model_load_path and os.path.exists(model_load_path))
    
    if use_ude:
        print(f"Используем обученную UDE модель из {model_load_path}")
        
        model_type = config.get('model.type', 'potential_nn')
        hidden_dim = config.get('model.hidden_dim', 16)
        num_blocks = config.get('model.num_residual_blocks', 4)
        max_potential = config.get('model.max_potential', 50.0)
        dropout_rate = config.get('model.dropout_rate', 0.1)
        
        if model_type == 'kan':
            print("Загружаем KAN модель...")
            try:
                model = KANPotentialModel(
                    hidden_dim=hidden_dim,
                    num_layers=num_blocks,
                    max_potential=max_potential,
                    device=device
                )
            except ImportError:
                print("Ошибка: Модель KAN требует библиотеку pykan. Используем PotentialNN.")
                model = PotentialNN(
                    hidden_dim=hidden_dim, 
                    num_blocks=num_blocks,
                    max_potential=max_potential,
                    dropout_rate=dropout_rate
                ).to(device)
        else:
            print("Загружаем PotentialNN модель...")
            model = PotentialNN(
                hidden_dim=hidden_dim, 
                num_blocks=num_blocks,
                max_potential=max_potential,
                dropout_rate=dropout_rate
            ).to(device)
        
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.eval()
        
        sim = Simulation(
            potential=None,
            neural_network=model,
            clustering_algorithm=clustering_algorithm,
            **sim_params
        )
        
        sim_type = "UDE"
    else:
        print("Используем аналитический потенциал")
        
        potential_params = {
            'g_att': config.get('potential.g_att', 7.0),
            'g_rep': config.get('potential.g_rep', 10.0),
            'm_pi': config.get('potential.m_pi', 0.7),
            'm_rho': config.get('potential.m_rho', 3.5),
            'r_cutoff': config.get('potential.r_cutoff', 5.0),
            'r_core': config.get('potential.r_core', 0.5),
            'device': device
        }
        
        potential = MesonExchangePotential(**potential_params)
        
        sim = Simulation(
            potential=potential,
            clustering_algorithm=clustering_algorithm,
            **sim_params
        )
        
        sim_type = "Analytical"
    
    print(f"Настраиваем симуляцию: {setup_params}")
    sim.setup_impact_parameter(**setup_params)
    
    print(f"Запускаем симуляцию с параметрами: {run_params}")
    result = sim.run(**run_params)
    
    run_output_dir = os.path.join(output_dir, f"{sim_type}_simulation_{int(time.time())}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    print("Создаем анимацию столкновения...")
    anim_path = os.path.join(run_output_dir, "collision_animation.mp4")
    
    trajectories = np.array(sim.trajectories)
    times = sim.times
    
    cluster_ids = None
    if sim.particles['cluster_ids'] is not None:
        cluster_ids = sim.particles['cluster_ids'].cpu().numpy()
    
    create_trajectory_animation(
        trajectories=trajectories,
        times=times,
        filename=anim_path,
        fps=30,
        limit=10,
        cluster_ids=cluster_ids,
        title=f"{sim_type} Simulation"
    )
    
    print("Анализируем кластеризацию...")
    cluster_path = os.path.join(run_output_dir, "cluster_analysis.png")
    
    final_positions = trajectories[-1]
    cluster_labels = clustering_algorithm.fit_predict(final_positions)
    
    visualize_cluster_analysis(
        positions=final_positions,
        cluster_labels=cluster_labels,
        save_path=cluster_path,
        limit=10,
        title=f"Кластерный анализ - {sim_type} Simulation"
    )
    
    if 'energies' in result:
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        plt.plot(result['energies']['kinetic'], label='Kinetic Energy')
        plt.title("Кинетическая энергия")
        plt.xlabel("Шаг симуляции")
        plt.ylabel("Энергия")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(result['energies']['potential'], label='Potential Energy')
        plt.plot(result['energies']['total'], label='Total Energy')
        plt.title("Энергии системы")
        plt.xlabel("Шаг симуляции")
        plt.ylabel("Энергия")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_output_dir, "energy_plot.png"))
        plt.close()
    
    plt.figure(figsize=(12, 10))
    positions = np.array(result['positions'])
    
    for i in range(min(8, positions.shape[1])):
        plt.plot(positions[:, i, 0], positions[:, i, 1], '-', label=f'Particle {i+1}')
    
    plt.title("Траектории частиц")
    plt.xlabel("X позиция")
    plt.ylabel("Y позиция")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(run_output_dir, "trajectories.png"))
    plt.close()
    
    n_collisions = config.get('simulation.multiple_collisions', 0)
    if n_collisions > 0:
        print(f"\nЗапуск {n_collisions} симуляций для статистического анализа...")
        
        results = sim.run_multiple_collisions(
            count=n_collisions,
            nucleus_count1=setup_params['nucleus_count1'],
            nucleus_count2=setup_params['nucleus_count2'],
            velocity=setup_params['relative_velocity'],
            max_impact_parameter=config.get('simulation.max_impact_parameter', 3.0),
            save_interval=run_params['save_interval'],
            dt_initial=run_params['dt_initial'],
            max_steps=run_params['max_steps'] // 2,
            random_velocity=setup_params['random_velocity'],
            radius1=setup_params['radius1'],
            radius2=setup_params['radius2']
        )
        
        print("Анализ результатов множественных симуляций...")
        stats = analyze_multiple_collision_results(results, os.path.join(run_output_dir, "statistics.png"))
        
        if stats is not None and 'figure' in stats:
            plt.close(stats['figure'])
            print(f"Статистика сохранена в {os.path.join(run_output_dir, 'statistics.png')}")
    
    end_time = time.time()
    print(f"--- Симуляция завершена ({end_time - start_time:.2f}с) ---")
    print(f"Результаты сохранены в {run_output_dir}")


if __name__ == "__main__":
    parser = create_arg_parser('run_simulation')
    args = parser.parse_args()
    
    config = load_config(args)
    
    run_simulation(config) 