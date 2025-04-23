import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import time

try:
    from core import Simulation
    from potential import MesonExchangePotential
    from ude.ude_network import PotentialNN
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core import Simulation
    from potential import MesonExchangePotential
    from ude.ude_network import PotentialNN

def generate_data(args):
    print("\n--- Starting Data Generation ---")
    start_time = time.time()
    output_dir = os.path.dirname(args.data_file)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    potential_params = {
        'g_att': args.g_att, 'g_rep': args.g_rep, 'm_pi': args.m_pi,
        'm_rho': args.m_rho, 'r_cutoff': args.r_cutoff, 'r_core': args.r_core,
        'device': device
    }
    sim_params = {
        't_end': args.t_end, 'device': device, 'dt_min': args.dt_min,
        'tau_max': args.tau_max, 'Imax': args.Imax,
    }
    setup_params = {
        'nucleus_count1': args.n_particles // 2, 'nucleus_count2': args.n_particles - (args.n_particles // 2),
        'impact_parameter': args.impact_parameter, 'relative_velocity': args.relative_velocity,
        'random_velocity': args.random_velocity, 'radius1': args.radius, 'radius2': args.radius
    }
    run_params = {
        'save_interval': args.save_interval, 'dt_initial': args.dt_initial, 'max_steps': args.max_steps
    }

    potential = MesonExchangePotential(**potential_params)
    sim = Simulation(potential, **sim_params)
    sim.setup_impact_parameter(**setup_params)
    sim.clustering_algorithm = None

    print(f"Running simulation (max_steps={args.max_steps}, dt_initial={args.dt_initial})...")
    result = sim.run(**run_params)

    times = torch.tensor(result['times'], device=device, dtype=torch.float32)
    positions = torch.stack(result['positions']).to(device).float()
    velocities = torch.stack(result['velocities']).to(device).float()

    print(f"Adding noise (level={args.noise_level})...")
    pos_mean_norm = torch.mean(torch.norm(positions[0], dim=1))
    vel_mean_norm = torch.mean(torch.norm(velocities[0], dim=1))
    pos_noise_std = args.noise_level * pos_mean_norm if pos_mean_norm > 0 else 1e-6
    vel_noise_std = args.noise_level * vel_mean_norm if vel_mean_norm > 0 else 1e-6
    noisy_positions = positions + torch.randn_like(positions) * pos_noise_std
    noisy_velocities = velocities + torch.randn_like(velocities) * vel_noise_std

    data_to_save = {
        'potential_params': potential_params, 'sim_params': sim_params,
        'setup_params': setup_params, 'run_params': run_params,
        'noise_level': args.noise_level, 'times': times,
        'true_positions': positions, 'true_velocities': velocities,
        'noisy_positions': noisy_positions, 'noisy_velocities': noisy_velocities,
        'masses': sim.nucleons['masses']
    }

    for key, value in data_to_save.items():
        if isinstance(value, torch.Tensor):
            data_to_save[key] = value.cpu()
        if isinstance(value, dict) and 'device' in value:
            value['device'] = str(value['device'])

    print(f"Saving data to {args.data_file}...")
    torch.save(data_to_save, args.data_file)
    end_time = time.time()
    print(f"--- Data Generation Finished ({end_time - start_time:.2f}s) ---")

def train_ude(args):
    print("\n--- Starting UDE Training ---")
    start_time = time.time()
    output_dir = os.path.dirname(args.model_save_path)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}. Run with --generate_data first or provide existing file.")

    print(f"Loading training data from {args.data_file}...")
    data = torch.load(args.data_file, map_location='cpu')
    noisy_positions = data['noisy_positions'].to(device)
    noisy_velocities = data['noisy_velocities'].to(device)
    times = data['times'].to(device)
    masses = data['masses'].to(device)
    sim_params_loaded = data['sim_params']
    run_params_loaded = data['run_params']
    print(f"Data loaded: {noisy_positions.shape[0]} time steps, {noisy_positions.shape[1]} particles.")

    print("Initializing neural network and simulation...")
    nn_model = PotentialNN(hidden_dim=args.nn_hidden_dim).to(device)
    sim_model = Simulation(
        potential=None, neural_network=nn_model,
        t_end=sim_params_loaded['t_end'], device=device,
        dt_min=sim_params_loaded['dt_min'], tau_max=sim_params_loaded['tau_max'],
        Imax=sim_params_loaded['Imax'], adaptive_dt=False
    )
    sim_model.nucleons['masses'] = masses

    optimizer = optim.Adam(nn_model.parameters(), lr=args.learning_rate)
    losses = []
    initial_pos_noisy = noisy_positions[0]
    initial_vel_noisy = noisy_velocities[0]
    target_positions_noisy = noisy_positions

    def predict_local(sim, init_pos, init_vel, ts, dt_init, max_steps):
        initial_state = {'positions': init_pos, 'velocities': init_vel}
        results = sim.run(save_interval=1, dt_initial=dt_init, max_steps=max_steps,
                          initial_state=initial_state, disable_pbar=True)
        return results['positions'][:len(ts)]

    def loss_fn_local(predicted, target):
         min_len = min(predicted.shape[0], target.shape[0])
         return torch.mean((predicted[:min_len] - target[:min_len])**2)

    print(f"Starting training for {args.epochs} epochs...")
    pbar = tqdm(range(args.epochs), desc="Training UDE")
    for epoch in pbar:
        optimizer.zero_grad()
        nn_model.train()
        predicted_positions = predict_local(
            sim_model, initial_pos_noisy, initial_vel_noisy, times,
            run_params_loaded['dt_initial'], run_params_loaded['max_steps']
        )
        loss = loss_fn_local(predicted_positions.to(device), target_positions_noisy)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        losses.append(current_loss)
        pbar.set_postfix({"Loss": f"{current_loss:.4e}"})

    print("Training complete.")
    print(f"Saving trained model to {args.model_save_path}...")
    torch.save(nn_model.state_dict(), args.model_save_path)

    plot_loss_path = os.path.join(output_dir, "ude_training_loss.png")
    print(f"Saving training loss plot to {plot_loss_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("UDE Training Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(plot_loss_path)
    plt.close()

    end_time = time.time()
    print(f"--- UDE Training Finished ({end_time - start_time:.2f}s) ---")

def analyze_results(args):
    print("\n--- Starting Analysis ---")
    start_time = time.time()
    output_dir = args.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_trajectory_path = os.path.join(output_dir, "ude_trajectory_comparison.png")
    plot_force_path = os.path.join(output_dir, "ude_force_comparison.png")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if not os.path.exists(args.model_load_path):
         raise FileNotFoundError(f"Model file not found: {args.model_load_path}. Run training first.")

    print(f"Loading data from {args.data_file}...")
    data = torch.load(args.data_file, map_location='cpu')
    noisy_positions = data['noisy_positions'].to(device)
    true_positions = data['true_positions'].to(device)
    true_velocities = data['true_velocities'].to(device)
    times = data['times'].to(device)
    masses = data['masses'].to(device)
    sim_params_loaded = data['sim_params']
    run_params_loaded = data['run_params']
    potential_params_loaded = data['potential_params']

    potential_params_loaded['device'] = device
    print("Data loaded.")

    print(f"Loading trained model from {args.model_load_path}...")
    nn_model = PotentialNN(hidden_dim=args.nn_hidden_dim).to(device)
    try:
        nn_model.load_state_dict(torch.load(args.model_load_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure --nn_hidden_dim used for analysis matches the saved model.")
        sys.exit(1)
    nn_model.eval()
    print("Trained model loaded.")

    sim_model_ude = Simulation(
        potential=None, neural_network=nn_model,
        t_end=sim_params_loaded['t_end'], device=device,
        dt_min=sim_params_loaded['dt_min'], tau_max=sim_params_loaded['tau_max'],
        Imax=sim_params_loaded['Imax'], adaptive_dt=False
    )
    sim_model_ude.nucleons['masses'] = masses

    def predict_local(sim, init_pos, init_vel, ts, dt_init, max_steps):
        initial_state = {'positions': init_pos, 'velocities': init_vel}
        results = sim.run(save_interval=1, dt_initial=dt_init, max_steps=max_steps,
                          initial_state=initial_state, disable_pbar=True)
        return results['positions'][:len(ts)]

    print("Running prediction with trained UDE model (using clean initial state)...")
    initial_pos_true = true_positions[0]
    initial_vel_true = true_velocities[0]
    ude_predicted_positions = predict_local(
        sim_model_ude, initial_pos_true, initial_vel_true, times,
        run_params_loaded['dt_initial'], run_params_loaded['max_steps']
    )
    ude_predicted_positions = ude_predicted_positions.detach().cpu()
    print("Prediction complete.")

    print(f"Plotting trajectory comparison to {plot_trajectory_path}...")
    num_particles_to_plot = min(5, true_positions.shape[1])
    time_indices = np.linspace(0, len(times)-1, num=min(len(times), 200), dtype=int)
    time_points_plot = times[time_indices].cpu().numpy()

    plt.figure(figsize=(12, 8))
    for i in range(num_particles_to_plot):
        plt.plot(time_points_plot, ude_predicted_positions[time_indices, i, 0].numpy(), 'r--', alpha=0.8, label=f'UDE Pred (P{i} X)' if i==0 else None)
        plt.plot(time_points_plot, ude_predicted_positions[time_indices, i, 1].numpy(), 'b--', alpha=0.8, label=f'UDE Pred (P{i} Y)' if i==0 else None)
        plt.plot(time_points_plot, true_positions[time_indices, i, 0].cpu().numpy(), 'r-', alpha=0.6, label='True (Clean)' if i==0 else None)
        plt.plot(time_points_plot, true_positions[time_indices, i, 1].cpu().numpy(), 'b-', alpha=0.6, label=None)
        plt.scatter(time_points_plot, noisy_positions[time_indices, i, 0].cpu().numpy(), c='red', marker='.', s=10, alpha=0.3, label='Noisy Data' if i==0 else None)
        plt.scatter(time_points_plot, noisy_positions[time_indices, i, 1].cpu().numpy(), c='blue', marker='.', s=10, alpha=0.3, label=None)

    plt.xlabel("Time")
    plt.ylabel("Position (X=Red, Y=Blue)")
    plt.title("UDE Predicted Trajectory vs True Trajectory & Noisy Data")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_trajectory_path)
    plt.close()
    print("Trajectory plot saved.")

    print(f"Plotting force comparison to {plot_force_path}...")
    true_potential = MesonExchangePotential(**potential_params_loaded)

    min_dist = 0.1
    max_dist = potential_params_loaded['r_cutoff'] * 0.95
    distances = torch.linspace(min_dist, max_dist, 200, device=device)
    relative_vectors = torch.zeros((len(distances), 3), device=device)
    relative_vectors[:, 0] = distances
    r = distances

    m_rho = potential_params_loaded['m_rho']
    m_pi = potential_params_loaded['m_pi']
    g_rep = potential_params_loaded['g_rep']
    g_att = potential_params_loaded['g_att']
    term1 = g_rep * torch.exp(-m_rho*r) * (m_rho/r + 1/(r**2))
    term2 = g_att * torch.exp(-m_pi*r) * (m_pi/r + 1/(r**2))
    true_force_magnitudes = term1 - term2
    true_force_vectors = true_force_magnitudes.unsqueeze(1) * (relative_vectors / r.unsqueeze(1))

    relative_vectors_ji = -relative_vectors
    with torch.no_grad():
        force_ij_pred = nn_model(relative_vectors)
        force_ji_pred = nn_model(relative_vectors_ji)
    learned_force_vectors = 0.5 * (force_ij_pred - force_ji_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(distances.cpu().numpy(), true_force_vectors[:, 0].cpu().numpy(), 'k-', label='True Potential Force (Fx)')
    plt.plot(distances.cpu().numpy(), learned_force_vectors[:, 0].cpu().numpy(), 'r--', label='Learned UDE Force (Fx)')

    plt.xlabel("Distance (r)")
    plt.ylabel("Force component Fx")
    plt.title("Comparison of Learned Force vs True Potential Force")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.ylim(auto=True)
    plt.savefig(plot_force_path)
    plt.close()
    print("Force comparison plot saved.")

    end_time = time.time()
    print(f"--- Analysis Finished ({end_time - start_time:.2f}s) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full UDE pipeline: Data Generation -> Training -> Analysis")

    parser.add_argument('--data_file', type=str, default="data/training_data.pt", help='Path to save/load the training data file.')
    parser.add_argument('--model_save_path', type=str, default="ude_output/ude_potential_nn.pt", help='Path to save the trained UDE model.')
    parser.add_argument('--model_load_path', type=str, default="ude_output/ude_potential_nn.pt", help='Path to load the trained UDE model for analysis.')
    parser.add_argument('--analysis_output_dir', type=str, default="ude_output", help='Directory to save analysis plots.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    parser.add_argument('--skip_data_gen', action='store_true', help='Skip data generation if data file exists.')
    parser.add_argument('--skip_training', action='store_true', help='Skip training if model file exists.')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip the final analysis step.')

    parser.add_argument('--g_att', type=float, default=13.5, help='Attractive potential strength.')
    parser.add_argument('--g_rep', type=float, default=20.0, help='Repulsive potential strength.')
    parser.add_argument('--m_pi', type=float, default=0.70, help='Pion mass (range parameter).')
    parser.add_argument('--m_rho', type=float, default=3.93, help='Rho meson mass (range parameter).')
    parser.add_argument('--r_cutoff', type=float, default=5.0, help='Potential cutoff radius.')
    parser.add_argument('--r_core', type=float, default=0.3, help='Potential core radius.')
    parser.add_argument('--t_end', type=float, default=1.0, help='Simulation end time.')
    parser.add_argument('--dt_min', type=float, default=1e-16, help='Minimum adaptive timestep.')
    parser.add_argument('--tau_max', type=float, default=0.01, help='Maximum error tolerance for adaptive timestep.')
    parser.add_argument('--Imax', type=float, default=0.1, help='Maximum impulse for adaptive timestep.')
    parser.add_argument('--n_particles', type=int, default=20, help='Total number of particles in simulation.')
    parser.add_argument('--impact_parameter', type=float, default=1.0, help='Impact parameter for collision setup.')
    parser.add_argument('--relative_velocity', type=float, default=20.0, help='Initial relative velocity of nuclei.')
    parser.add_argument('--random_velocity', type=float, default=1.0, help='Magnitude of initial random velocities.')
    parser.add_argument('--radius', type=float, default=0.8, help='Radius for initial particle distribution.')
    parser.add_argument('--save_interval', type=int, default=1, help='Simulation save interval.')
    parser.add_argument('--dt_initial', type=float, default=0.001, help='Initial timestep for simulation.')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum simulation steps.')
    parser.add_argument('--noise_level', type=float, default=5e-3, help='Noise level to add to generated data.')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--nn_hidden_dim', type=int, default=64, help='Hidden dimension size for the PotentialNN.')

    args = parser.parse_args()

    overall_start_time = time.time()

    if not args.skip_data_gen or not os.path.exists(args.data_file):
        if not args.skip_data_gen and os.path.exists(args.data_file):
             print(f"Data file {args.data_file} exists, but --skip_data_gen not specified. Regenerating data.")
        generate_data(args)
    else:
        print(f"Skipping data generation, using existing file: {args.data_file}")

    if not args.skip_training or not os.path.exists(args.model_save_path):
        if not args.skip_training and os.path.exists(args.model_save_path):
            print(f"Model file {args.model_save_path} exists, but --skip_training not specified. Retraining model.")
        train_ude(args)
    else:
         print(f"Skipping training, using existing model: {args.model_save_path}")
         args.model_load_path = args.model_save_path

    if not args.skip_analysis:
        args.model_load_path = args.model_save_path if not args.skip_training else args.model_load_path
        analyze_results(args)
    else:
        print("Skipping analysis.")

    overall_end_time = time.time()
    print(f"\n--- Pipeline Finished ({overall_end_time - overall_start_time:.2f}s) ---")