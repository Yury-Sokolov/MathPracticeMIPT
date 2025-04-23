import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Simulation
from potential import MesonExchangePotential
from ude.ude_network import PotentialNN
from ude.train_ude import predict

if __name__ == "__main__":
    data_file = "data/training_data.pt"
    model_load_path = "ude_output/ude_potential_nn.pt"
    output_dir = "ude_output"

    plot_trajectory_path = os.path.join(output_dir, "ude_trajectory_comparison.png")
    plot_force_path = os.path.join(output_dir, "ude_force_comparison.png")

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nn_hidden_dim = 64
    print(f"Loading data from {data_file}...")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}. Run generate_data.py first.")
    
    data = torch.load(data_file, map_location='cpu')
    noisy_positions = data['noisy_positions'].to(device)
    true_positions = data['true_positions'].to(device)

    times = data['times'].to(device)
    masses = data['masses'].to(device)

    sim_params_loaded = data['sim_params']
    run_params_loaded = data['run_params']
    potential_params_loaded = data['potential_params']

    if isinstance(potential_params_loaded['device'], str):
         potential_params_loaded['device'] = device

    print("Data loaded.")
    print(f"Loading trained model from {model_load_path}...")

    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model file not found: {model_load_path}. Run train_ude.py first.")
    
    nn_model = PotentialNN(hidden_dim=nn_hidden_dim).to(device)
    
    try:
         nn_model.load_state_dict(torch.load(model_load_path, map_location=device))
    except Exception as e:
         print(f"Error loading model state_dict: {e}")
         print("Ensure nn_hidden_dim matches the saved model.")
         sys.exit(1)

    nn_model.eval()

    print("Trained model loaded.")

    sim_model_ude = Simulation(
        potential=None,
        neural_network=nn_model,
        t_end=sim_params_loaded['t_end'],
        device=device,
        dt_min=sim_params_loaded['dt_min'],
        tau_max=sim_params_loaded['tau_max'],
        Imax=sim_params_loaded['Imax'],
        adaptive_dt=False
    )

    sim_model_ude.nucleons['masses'] = masses

    print("Running prediction with trained UDE model...")

    initial_pos_true = true_positions[0]

    initial_vel_true = data['true_velocities'][0].to(device)

    ude_predicted_positions = predict(
        sim_model_ude,
        initial_pos_true,
        initial_vel_true,
        times,
        dt_initial=run_params_loaded['dt_initial'],
        max_steps=run_params_loaded['max_steps']
    )

    ude_predicted_positions = ude_predicted_positions.detach().cpu()

    print("Prediction complete.")
    print(f"Plotting trajectory comparison to {plot_trajectory_path}...")

    num_particles_to_plot = min(5, true_positions.shape[1])
    time_indices = np.linspace(0, len(times)-1, num=min(len(times), 100), dtype=int)
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
    plt.title("UDE Predicted Trajectory vs True Trajectory")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_trajectory_path)
    plt.close()

    print("Trajectory plot saved.")
    print(f"Plotting force comparison to {plot_force_path}...")

    true_potential = MesonExchangePotential(**potential_params_loaded)
    min_dist = 0.1
    max_dist = potential_params_loaded['r_cutoff'] * 0.95
    distances = torch.linspace(min_dist, max_dist, 100, device=device)
    relative_vectors = torch.zeros((len(distances), 3), device=device)
    relative_vectors[:, 0] = distances
    r = distances
    m_rho = potential_params_loaded['m_rho']
    m_pi = potential_params_loaded['m_pi']
    g_rep = potential_params_loaded['g_rep']
    g_att = potential_params_loaded['g_att']
    true_force_magnitudes = g_rep * torch.exp(-m_rho*r) * (m_rho/r + 1/(r**2)) - g_att * torch.exp(-m_pi*r) * (m_pi/r + 1/(r**2))
    true_force_vectors = true_force_magnitudes.unsqueeze(1) * (relative_vectors / r.unsqueeze(1))
    relative_vectors_ij = relative_vectors
    relative_vectors_ji = -relative_vectors

    with torch.no_grad():
        force_ij_pred = nn_model(relative_vectors_ij)
        force_ji_pred = nn_model(relative_vectors_ji)
    learned_force_vectors = 0.5 * (force_ij_pred - force_ji_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(distances.cpu().numpy(), true_force_vectors[:, 0].cpu().numpy(), 'k-', label='True Potential Force (Fx)')
    plt.plot(distances.cpu().numpy(), learned_force_vectors[:, 0].cpu().numpy(), 'r--', label='Learned UDE Force (Fx)')
    plt.xlabel("Distance (r)")
    plt.ylabel("Force component Fx (or Magnitude)")
    plt.title("Comparison of Learned Force vs True Potential Force")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.ylim(auto=True)
    plt.savefig(plot_force_path)
    plt.close()
    
    print("Force comparison plot saved.")
    print("Analysis script finished.")