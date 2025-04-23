import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Simulation
from potential import MesonExchangePotential
from ude.ude_network import PotentialNN

def predict(sim_model: Simulation, initial_pos, initial_vel, times, dt_initial, max_steps):
    initial_state = {
        'positions': initial_pos,
        'velocities': initial_vel
    }
    save_interval = 1
    results = sim_model.run(
        save_interval=save_interval,
        dt_initial=dt_initial,
        max_steps=max_steps,
        initial_state=initial_state,
        disable_pbar=True
    )
    predicted_positions = results['positions']
    return predicted_positions[:len(times)]

def loss_fn(predicted_positions, target_positions):
    len_pred = predicted_positions.shape[0]
    len_target = target_positions.shape[0]
    min_len = min(len_pred, len_target)
    if len_pred != len_target:
         print(f"Warning: Length mismatch in loss calculation. Pred: {len_pred}, Target: {len_target}. Using min length {min_len}.")
    loss = torch.mean((predicted_positions[:min_len] - target_positions[:min_len])**2)
    return loss

if __name__ == "__main__":
    data_file = "data/training_data.pt"
    output_dir = "ude_output"
    model_save_path = os.path.join(output_dir, "ude_potential_nn.pt")
    plot_save_path = os.path.join(output_dir, "ude_training_loss.png")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    learning_rate = 1e-3
    epochs = 500
    nn_hidden_dim = 64
    print(f"Loading training data from {data_file}...")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}. Run generate_data.py first.")
    data = torch.load(data_file, map_location='cpu')
    noisy_positions = data['noisy_positions'].to(device)
    noisy_velocities = data['noisy_velocities'].to(device)
    true_positions = data['true_positions'].to(device)
    times = data['times'].to(device)
    masses = data['masses'].to(device)
    sim_params_loaded = data['sim_params']
    run_params_loaded = data['run_params']
    potential_params_loaded = data['potential_params']
    print(f"Data loaded: {noisy_positions.shape[0]} time steps, {noisy_positions.shape[1]} particles.")
    print("Initializing neural network and simulation...")
    nn_model = PotentialNN(hidden_dim=nn_hidden_dim).to(device)
    sim_model = Simulation(
        potential=None,
        neural_network=nn_model,
        t_end=sim_params_loaded['t_end'],
        device=device,
        dt_min=sim_params_loaded['dt_min'],
        tau_max=sim_params_loaded['tau_max'],
        Imax=sim_params_loaded['Imax'],
        adaptive_dt=False
    )
    sim_model.nucleons['masses'] = masses
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)
    print("Starting training...")
    losses = []
    initial_pos_noisy = noisy_positions[0]
    initial_vel_noisy = noisy_velocities[0]
    target_positions_noisy = noisy_positions
    pbar = tqdm(range(epochs), desc="Training UDE")
    for epoch in pbar:
        optimizer.zero_grad()
        nn_model.train()
        predicted_positions = predict(
            sim_model,
            initial_pos_noisy,
            initial_vel_noisy,
            times,
            dt_initial=run_params_loaded['dt_initial'],
            max_steps=run_params_loaded['max_steps']
        )
        loss = loss_fn(predicted_positions.to(device), target_positions_noisy)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        losses.append(current_loss)
        pbar.set_postfix({"Loss": f"{current_loss:.4e}"})
    print("Training complete.")
    print(f"Saving trained model to {model_save_path}...")
    torch.save(nn_model.state_dict(), model_save_path)
    print(f"Saving training loss plot to {plot_save_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("UDE Training Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(plot_save_path)
    plt.close()
    print("Training script finished.")
