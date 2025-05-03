import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import Simulation
from potential.potential import MesonExchangePotential
from utils.config import load_config, create_arg_parser
from utils.data_manager import DataManager


def generate_data(config):
    """
    Генерирует данные для обучения UDE.
    
    Args:
        config: Конфигурация с параметрами для генерации данных
    """
    print("\n--- Начинаем генерацию данных ---")
    start_time = time.time()
    
    output_file = config.get('data.output_file')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Используется устройство: {device}")
    
    potential_params = {
        'g_att': config.get('potential.g_att'),
        'g_rep': config.get('potential.g_rep'),
        'm_pi': config.get('potential.m_pi'),
        'm_rho': config.get('potential.m_rho'),
        'r_cutoff': config.get('potential.r_cutoff'),
        'r_core': config.get('potential.r_core'),
        'device': device
    }
    
    sim_params = {
        't_end': config.get('simulation.t_end'),
        'device': device,
        'dt_min': config.get('simulation.dt_min'),
        'tau_max': config.get('simulation.tau_max'),
        'Imax': config.get('simulation.Imax'),
    }
    
    n_particles = config.get('simulation.n_particles')
    setup_base_params = {
        'nucleus_count1': n_particles // 2,
        'nucleus_count2': n_particles - (n_particles // 2),
        'relative_velocity': config.get('simulation.relative_velocity'),
        'random_velocity': config.get('simulation.random_velocity'),
        'radius1': config.get('simulation.radius'),
        'radius2': config.get('simulation.radius')
    }
    
    run_params = {
        'save_interval': config.get('data.save_interval'),
        'dt_initial': config.get('data.dt_initial'),
        'max_steps': config.get('data.max_steps')
    }
    
    potential = MesonExchangePotential(**potential_params)
    sim = Simulation(potential, **sim_params)
    sim.clustering_algorithm = None
    
    num_collisions = config.get('data.num_collisions')
    max_impact_parameter = config.get('simulation.max_impact_parameter')
    
    print(f"Генерируем данные из {num_collisions} столкновений...")
    
    all_times = []
    all_positions = []
    all_velocities = []
    all_masses = None
    
    random_impact_params = np.sqrt(np.random.random(num_collisions)) * max_impact_parameter
    
    if config.get('wandb.use_wandb', False):
        try:
            import wandb
            wandb.init(
                project=config.get('wandb.project'),
                entity=config.get('wandb.entity'),
                name=config.get('wandb.run_name'),
                config={k: v for k, v in config.config.items()},
                tags=config.get('wandb.tags', []) + ["data_generation"]
            )
            wandb.log({"impact_parameters": wandb.Histogram(random_impact_params)})
        except ImportError:
            print("Предупреждение: Не удалось импортировать wandb. Логирование будет отключено.")
            config.set('wandb.use_wandb', False)
    
    for b in tqdm(random_impact_params, desc="Симулируем столкновения"):
        current_setup_params = setup_base_params.copy()
        current_setup_params['impact_parameter'] = b
        sim.setup_impact_parameter(**current_setup_params)
        
        result = sim.run(**run_params)
        
        times_coll = torch.tensor(result['times'], dtype=torch.float32).cpu()
        pos_coll = torch.from_numpy(np.array(result['positions'])).float().cpu()
        vel_coll = torch.from_numpy(np.array(result['velocities'])).float().cpu()
        
        all_times.append(times_coll)
        all_positions.append(pos_coll)
        all_velocities.append(vel_coll)
        
        if all_masses is None:
            all_masses = sim.nucleons['masses'].cpu()
        
        if config.get('wandb.use_wandb', False):
            max_vel = torch.norm(vel_coll[-1], dim=-1).max().item()
            final_spread = torch.std(pos_coll[-1], dim=0).mean().item()
            
            wandb.log({
                f"collision_{len(all_times)}/impact_parameter": b,
                f"collision_{len(all_times)}/max_velocity": max_vel,
                f"collision_{len(all_times)}/final_spatial_spread": final_spread,
                f"collision_{len(all_times)}/trajectory_length": len(times_coll)
            })
    
    try:
        stacked_positions = torch.stack(all_positions, dim=0)
        stacked_velocities = torch.stack(all_velocities, dim=0)
        times = all_times[0]
        
        n_collisions, n_steps, n_particles, _ = stacked_positions.shape
        true_positions_flat = stacked_positions.reshape(n_collisions * n_steps, n_particles, 3)
        true_velocities_flat = stacked_velocities.reshape(n_collisions * n_steps, n_particles, 3)
        times_flat = times.repeat(n_collisions)
    except RuntimeError as e:
        print(f"Ошибка при объединении траекторий: {e}")
        print("Траектории могут иметь разную длину. Используем данные только из первого столкновения.")
        true_positions_flat = all_positions[0]
        true_velocities_flat = all_velocities[0]
        times_flat = all_times[0]
    
    print(f"Объединенные данные: Позиции={true_positions_flat.shape}, Скорости={true_velocities_flat.shape}, Времена={times_flat.shape}")
    
    noise_level = config.get('data.noise_level')
    print(f"Добавляем шум (уровень={noise_level})...")
    
    pos_mean_norm = torch.mean(torch.norm(true_positions_flat[0], dim=1))
    vel_mean_norm = torch.mean(torch.norm(true_velocities_flat[0], dim=1))
    
    pos_noise_std = noise_level * pos_mean_norm if pos_mean_norm > 0 else 1e-6
    vel_noise_std = noise_level * vel_mean_norm if vel_mean_norm > 0 else 1e-6
    
    noisy_positions_flat = true_positions_flat + torch.randn_like(true_positions_flat) * pos_noise_std
    noisy_velocities_flat = true_velocities_flat + torch.randn_like(true_velocities_flat) * vel_noise_std
    
    if config.get('wandb.use_wandb', False):
        wandb.log({
            "dataset/total_timesteps": len(times_flat),
            "dataset/num_particles": true_positions_flat.shape[1],
            "dataset/pos_noise_std": pos_noise_std,
            "dataset/vel_noise_std": vel_noise_std,
            "dataset/pos_mean_norm": pos_mean_norm.item(),
            "dataset/vel_mean_norm": vel_mean_norm.item(),
        })
        
        if true_positions_flat.shape[0] > 0 and true_positions_flat.shape[1] > 0:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 8))
            sample_particles = min(5, true_positions_flat.shape[1])
            
            for i in range(sample_particles):
                plt.plot(true_positions_flat[:100, i, 0].cpu().numpy(), 
                       true_positions_flat[:100, i, 1].cpu().numpy(), 
                       '-', alpha=0.7, label=f'True P{i}')
                plt.plot(noisy_positions_flat[:100, i, 0].cpu().numpy(), 
                       noisy_positions_flat[:100, i, 1].cpu().numpy(), 
                       '.', markersize=2, alpha=0.5, label=f'Noisy P{i}')
            
            plt.title("Пример траектории (первые 100 шагов)")
            plt.xlabel("X позиция")
            plt.ylabel("Y позиция")
            plt.legend()
            wandb.log({"dataset/sample_trajectory": wandb.Image(fig)})
            plt.close(fig)
    
    data_to_save = {
        'potential_params': potential_params,
        'sim_params': sim_params,
        'setup_base_params': setup_base_params,
        'run_params': run_params,
        'generation_params': {
            'num_collisions': num_collisions,
            'max_impact_parameter': max_impact_parameter,
        },
        'noise_level': noise_level,
        'times': times_flat,
        'true_positions': true_positions_flat,
        'true_velocities': true_velocities_flat,
        'noisy_positions': noisy_positions_flat,
        'noisy_velocities': noisy_velocities_flat,
        'masses': all_masses
    }
    
    for key, value in data_to_save.items():
        if isinstance(value, dict) and 'device' in value:
            value['device'] = str(value['device'])
    
    print(f"Сохраняем данные в {output_file}...")
    DataManager.save_data(data_to_save, output_file)
    
    end_time = time.time()
    print(f"--- Генерация данных завершена ({end_time - start_time:.2f}с) ---")
    
    if config.get('wandb.use_wandb', False):
        wandb.finish()


if __name__ == "__main__":
    parser = create_arg_parser('generate_data')
    args = parser.parse_args()
    
    config = load_config(args)
    
    generate_data(config) 