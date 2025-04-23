import torch
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Simulation
from potential import MesonExchangePotential

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "training_data.pt")
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    potential_params = {
        'g_att': 13.5,
        'g_rep': 20.0,
        'm_pi': 0.70,
        'm_rho': 3.93,
        'r_cutoff': 5.0,
        'r_core': 0.3,
        'device': device
    }
    sim_params = {
        't_end': 1.0,
        'device': device,
        'dt_min': 1e-16,
        'tau_max': 0.01,
        'Imax': 0.1,
    }
    setup_params = {
        'nucleus_count1': 10,
        'nucleus_count2': 10,
        'impact_parameter': 1.0,
        'relative_velocity': 20.0,
        'random_velocity': 1.0,
        'radius1': 0.8,
        'radius2': 0.8
    }
    run_params = {
        'save_interval': 1,
        'dt_initial': 0.001,
        'max_steps': 500
    }
    noise_level = 5e-3
    potential = MesonExchangePotential(**potential_params)
    sim = Simulation(potential, **sim_params)
    print("Setting up initial conditions...")
    sim.setup_impact_parameter(**setup_params)
    print(f"Running simulation to generate ground truth data (saving every {run_params['save_interval']} step(s))...")
    sim.clustering_algorithm = None
    result = sim.run(**run_params)
    times = torch.tensor(result['times'], device=device, dtype=torch.float32)
    positions = torch.stack(result['positions']).to(device).float()
    velocities = torch.stack(result['velocities']).to(device).float()
    print(f"Generated trajectory shapes: times={times.shape}, positions={positions.shape}, velocities={velocities.shape}")
    print(f"Adding noise (level={noise_level})...")
    pos_mean_norm = torch.mean(torch.norm(positions[0], dim=1))
    vel_mean_norm = torch.mean(torch.norm(velocities[0], dim=1))
    pos_noise_std = noise_level * pos_mean_norm
    vel_noise_std = noise_level * vel_mean_norm
    if pos_noise_std == 0: pos_noise_std = 1e-6
    if vel_noise_std == 0: vel_noise_std = 1e-6
    noisy_positions = positions + torch.randn_like(positions) * pos_noise_std
    noisy_velocities = velocities + torch.randn_like(velocities) * vel_noise_std
    data_to_save = {
        'potential_params': potential_params,
        'sim_params': sim_params,
        'setup_params': setup_params,
        'run_params': run_params,
        'noise_level': noise_level,
        'times': times,
        'true_positions': positions,
        'true_velocities': velocities,
        'noisy_positions': noisy_positions,
        'noisy_velocities': noisy_velocities,
        'masses': sim.nucleons['masses']
    }
    for key, value in data_to_save.items():
        if isinstance(value, torch.Tensor):
            data_to_save[key] = value.cpu()
        if isinstance(value, dict) and 'device' in value:
             value['device'] = str(value['device'])
    print(f"Saving data to {output_file}...")
    torch.save(data_to_save, output_file)
    print("Data generation complete.") 