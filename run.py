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
import wandb

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

    if args.use_wandb:
        wandb_config = {
            'phase': 'data_generation',
            'g_att': args.g_att,
            'g_rep': args.g_rep,
            'm_pi': args.m_pi,
            'm_rho': args.m_rho,
            'r_cutoff': args.r_cutoff,
            'r_core': args.r_core,
            't_end': args.t_end,
            'dt_min': args.dt_min,
            'tau_max': args.tau_max,
            'Imax': args.Imax,
            'n_particles': args.n_particles,
            'relative_velocity': args.relative_velocity,
            'random_velocity': args.random_velocity,
            'radius': args.radius,
            'num_collisions': args.num_collisions,
            'max_impact_parameter': args.max_impact_parameter,
            'noise_level': args.noise_level
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"data-gen-{args.n_particles}p-{args.num_collisions}c", 
            config=wandb_config,
            tags=args.wandb_tags + ["data_generation"],
            resume="allow"
        )
        
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
    setup_base_params = {
        'nucleus_count1': args.n_particles // 2,
        'nucleus_count2': args.n_particles - (args.n_particles // 2),
        'relative_velocity': args.relative_velocity,
        'random_velocity': args.random_velocity,
        'radius1': args.radius,
        'radius2': args.radius
    }
    run_params = {
        'save_interval': args.save_interval,
        'dt_initial': args.dt_initial,
        'max_steps': args.max_steps
    }

    potential = MesonExchangePotential(**potential_params)
    sim = Simulation(potential, **sim_params)
    sim.clustering_algorithm = None

    print(f"Generating data from {args.num_collisions} collisions...")
    all_times = []
    all_positions = []
    all_velocities = []
    all_masses = None

    random_impact_params = np.sqrt(np.random.random(args.num_collisions)) * args.max_impact_parameter
    
    if args.use_wandb:
        wandb.log({"impact_parameters": wandb.Histogram(random_impact_params)})

    for b in tqdm(random_impact_params, desc="Simulating Collisions"):
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
            
        if args.use_wandb:
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
        print(f"Error stacking trajectories: {e}")
        print("Trajectories might have different lengths. Implement padding or adjust simulation parameters.")
        print("Falling back to using data from the first collision only.")
        true_positions_flat = all_positions[0]
        true_velocities_flat = all_velocities[0]
        times_flat = all_times[0]

    print(f"Combined data shape: Positions={true_positions_flat.shape}, Velocities={true_velocities_flat.shape}, Times={times_flat.shape}")

    print(f"Adding noise (level={args.noise_level})...")
    pos_mean_norm = torch.mean(torch.norm(true_positions_flat[0], dim=0))
    vel_mean_norm = torch.mean(torch.norm(true_velocities_flat[0], dim=0))
    pos_noise_std = args.noise_level * pos_mean_norm if pos_mean_norm > 0 else 1e-6
    vel_noise_std = args.noise_level * vel_mean_norm if vel_mean_norm > 0 else 1e-6
    noisy_positions_flat = true_positions_flat + torch.randn_like(true_positions_flat) * pos_noise_std
    noisy_velocities_flat = true_velocities_flat + torch.randn_like(true_velocities_flat) * vel_noise_std
    
    if args.use_wandb:
        wandb.log({
            "dataset/total_timesteps": len(times_flat),
            "dataset/num_particles": true_positions_flat.shape[1],
            "dataset/pos_noise_std": pos_noise_std,
            "dataset/vel_noise_std": vel_noise_std,
            "dataset/pos_mean_norm": pos_mean_norm.item(),
            "dataset/vel_mean_norm": vel_mean_norm.item(),
        })
        
        if true_positions_flat.shape[0] > 0 and true_positions_flat.shape[1] > 0:
            fig = plt.figure(figsize=(10, 8))
            sample_particles = min(5, true_positions_flat.shape[1])
            
            for i in range(sample_particles):
                plt.plot(true_positions_flat[:100, i, 0].cpu().numpy(), 
                        true_positions_flat[:100, i, 1].cpu().numpy(), 
                        '-', alpha=0.7, label=f'True P{i}')
                plt.plot(noisy_positions_flat[:100, i, 0].cpu().numpy(), 
                        noisy_positions_flat[:100, i, 1].cpu().numpy(), 
                        '.', markersize=2, alpha=0.5, label=f'Noisy P{i}')
            
            plt.title("Sample Trajectory (First 100 steps)")
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.legend()
            wandb.log({"dataset/sample_trajectory": wandb.Image(fig)})
            plt.close(fig)

    data_to_save = {
        'potential_params': potential_params,
        'sim_params': sim_params,
        'setup_base_params': setup_base_params,
        'run_params': run_params,
        'generation_params': {
            'num_collisions': args.num_collisions,
            'max_impact_parameter': args.max_impact_parameter,
        },
        'noise_level': args.noise_level,
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

    print(f"Saving data to {args.data_file}...")
    torch.save(data_to_save, args.data_file)
    end_time = time.time()
    print(f"--- Data Generation Finished ({end_time - start_time:.2f}s) ---")
    
    if args.use_wandb:
        wandb.finish()

def train_ude(args):
    print("\n--- Starting UDE Training ---")
    start_time = time.time()
    output_dir = os.path.dirname(args.model_save_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.use_wandb:
        wandb_config = {
            'phase': 'training',
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'nn_hidden_dim': args.nn_hidden_dim,
            'batch_size': args.batch_size,
            'use_scheduler': args.use_scheduler,
            'weight_decay': args.weight_decay,
            'num_residual_blocks': args.num_residual_blocks,
            'symmetry_weight': args.symmetry_weight,
            'clip_grad': args.clip_grad,
            'patience': args.patience,
            'device': args.device
        }
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"train-ude-{args.nn_hidden_dim}d-{args.epochs}e", 
            config=wandb_config,
            tags=args.wandb_tags + ["training"],
            resume="allow"
        )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}. Run with --generate_data first or provide existing file.")

    print(f"Loading training data from {args.data_file}...")
    data = torch.load(args.data_file, map_location='cpu')
    
    if args.use_wandb:
        wandb.config.update({
            'data_file': args.data_file,
            'num_particles': data['noisy_positions'].shape[1],
            'noise_level': data.get('noise_level', 'unknown'),
            'g_att': data['potential_params']['g_att'],
            'g_rep': data['potential_params']['g_rep'],
            'm_pi': data['potential_params']['m_pi'],
            'm_rho': data['potential_params']['m_rho'],
        })

    times = data['times']
    masses = data['masses'].to(device)
    sim_params_loaded = data['sim_params']
    print(f"Data loaded: {data['noisy_positions'].shape[0]} total time steps, {data['noisy_positions'].shape[1]} particles.")

    if len(data['times']) > 1:
        dt = (data['times'][1] - data['times'][0]).item()
    else:
        dt = 0.001
        print("Warning: Could not robustly determine original dt. Using default 0.001")

    print("Computing normalization statistics...")
    pos_mean = torch.mean(data['noisy_positions'], dim=(0, 1))
    pos_std = torch.std(data['noisy_positions'], dim=(0, 1))
    vel_mean = torch.mean(data['noisy_velocities'], dim=(0, 1))
    vel_std = torch.std(data['noisy_velocities'], dim=(0, 1))
    
    print("Calculating target accelerations...")
    chunk_size = min(1000, data['noisy_velocities'].shape[0] - 1)
    num_chunks = (data['noisy_velocities'].shape[0] - 1 + chunk_size - 1) // chunk_size
    
    normalized_accel_list = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, data['noisy_velocities'].shape[0] - 1)
        
        norm_vel_chunk_1 = (data['noisy_velocities'][start_idx:end_idx] - vel_mean) / (vel_std + 1e-8)
        norm_vel_chunk_2 = (data['noisy_velocities'][start_idx+1:end_idx+1] - vel_mean) / (vel_std + 1e-8)
        
        accel_chunk = (norm_vel_chunk_2 - norm_vel_chunk_1) / dt
        normalized_accel_list.append(accel_chunk)
        
        del norm_vel_chunk_1, norm_vel_chunk_2, accel_chunk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    normalized_accel = torch.cat(normalized_accel_list, dim=0)
    del normalized_accel_list
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accel_scale = vel_std / (dt * (pos_std + 1e-8))
    
    normalized_positions = (data['noisy_positions'] - pos_mean) / (pos_std + 1e-8)
    normalized_positions_for_loss = normalized_positions[:-1]
    
    accel_magnitudes = torch.norm(normalized_accel, dim=-1)
    max_accel = torch.max(accel_magnitudes).item()
    mean_accel = torch.mean(accel_magnitudes).item()
    print(f"Target acceleration stats: Max={max_accel:.4f}, Mean={mean_accel:.4f}")
    
    potential_params = data['potential_params'].copy()
    potential_params['device'] = device
    true_potential = MesonExchangePotential(**potential_params)
    
    print("Initializing neural network and simulation...")
    print("Using enhanced PotentialNN model with physics-informed architecture.")
    nn_model = PotentialNN(
        hidden_dim=args.nn_hidden_dim, 
        num_blocks=args.num_residual_blocks,
        max_potential=args.max_potential
    ).to(device)

    print("Pre-initializing weights to approximate true potential...")
    with torch.enable_grad():
        test_dists = torch.linspace(0.1, potential_params['r_cutoff'], 50, device=device)
        test_vectors = torch.zeros((len(test_dists), 3), device=device)
        test_vectors[:, 0] = test_dists
        
        true_forces = []
        true_potentials = []
        
        for r_idx, r_vec in enumerate(test_vectors):
            r_tensor = r_vec.reshape(1, 3)
            pos_pair = torch.stack([r_tensor[0], -r_tensor[0]])
            
            true_force = true_potential.compute_force_only(pos_pair)[0]
            true_forces.append(true_force)
            
            if r_idx > 0:
                dr = test_dists[r_idx] - test_dists[r_idx-1]
                if r_idx == 1:
                    prev_potential = torch.tensor(0.0, device=device)
                else:
                    prev_potential = true_potentials[-1]
                    
                current_potential = prev_potential - torch.norm(true_force) * dr
                true_potentials.append(current_potential)
            else:
                true_potentials.append(torch.tensor(0.0, device=device))
        
        true_forces = torch.stack(true_forces)
        true_potentials = torch.stack(true_potentials)
        
        min_potential = torch.min(true_potentials)
        true_potentials = true_potentials - min_potential

    init_model = PotentialNN(
        hidden_dim=args.nn_hidden_dim, 
        num_blocks=args.num_residual_blocks,
        max_potential=args.max_potential
    ).to(device)
    
    init_optimizer = optim.Adam(init_model.parameters(), lr=0.01)
    
    for _ in range(100):
        init_optimizer.zero_grad()
        
        with torch.enable_grad():
            test_vectors_clone = test_vectors.clone().requires_grad_(True)
            
            pred_potentials = init_model.compute_potential(test_vectors_clone)
            loss = nn.MSELoss()(pred_potentials, true_potentials)
            loss.backward()
        
        init_optimizer.step()
    
    with torch.enable_grad():
        test_vectors_grad = test_vectors.clone().requires_grad_(True)
        pred_potentials = init_model.compute_potential(test_vectors_grad)
        total_potential = torch.sum(pred_potentials)
        
        pred_forces = -torch.autograd.grad(
            total_potential, test_vectors_grad, 
            create_graph=False, retain_graph=True
        )[0]
        
        force_error = nn.MSELoss()(pred_forces, true_forces).item()
        print(f"Pre-initialization force error: {force_error:.6f}")
    
    nn_model.load_state_dict(init_model.state_dict())
    
    del init_model, init_optimizer, test_vectors_clone, pred_potentials, loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    sim_model = Simulation(
        potential=None, neural_network=nn_model,
        t_end=1.0,
        device=device,
        dt_min=1e-16, tau_max=0.01, Imax=0.1,
        adaptive_dt=False
    )
    sim_model.nucleons['masses'] = masses
    
    optimizer = optim.AdamW(nn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.use_wandb:
        wandb.watch(nn_model, log="all", log_freq=10)
    
    if args.use_scheduler:
        print(f"Using cosine scheduler with warmup and restart")
        warmup_epochs = max(3, int(args.epochs * 0.1))
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            cycle_epochs = args.epochs - warmup_epochs
            cycle_length = cycle_epochs // 3 
            cycle_epoch = (epoch - warmup_epochs) % cycle_length
            return 0.5 * (1 + np.cos(np.pi * cycle_epoch / cycle_length))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.learning_rate
        param_group['lr'] = args.learning_rate * 2

    losses = []
    force_profile_history = []
    

    print(f"Starting training for {args.epochs} epochs...")
    num_steps_loss = normalized_accel.shape[0]
    
    if args.batch_size <= 0:
        batch_size = min(128, num_steps_loss)
        print(f"Using default batch size of {batch_size} to prevent memory issues")
    else:
        batch_size = min(args.batch_size, num_steps_loss)
        
    num_batches = (num_steps_loss + batch_size - 1) // batch_size
    print(f"Total steps for loss: {num_steps_loss}, Batch size: {batch_size}, Batches per epoch: {num_batches}")

    best_loss = float('inf')
    patience_counter = 0
    zero_force_counter = 0
    
    def check_force_profile(model, device):
        """Проверяет, не стремится ли модель к нулевому решению"""
        
        distances = torch.linspace(0.2, 4.0, 20, device=device)
        test_vectors = torch.zeros((len(distances), 3), device=device)
        test_vectors[:, 0] = distances
        test_vectors.requires_grad_(True)
        
        potentials = model.compute_potential(test_vectors)
        total_potential = potentials.sum()
        
        forces = -torch.autograd.grad(
            total_potential, test_vectors, 
            create_graph=False, retain_graph=True
        )[0]
        
        force_x = forces[:, 0]
        
        max_force = torch.max(torch.abs(force_x)).item()
        mean_force = torch.mean(torch.abs(force_x)).item()
        
        is_zero_like = max_force < 0.01
        
        return max_force, mean_force, is_zero_like, force_x.cpu().numpy()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_symmetry_loss = 0.0
        epoch_force_magnitude_loss = 0.0
        
        permuted_indices = torch.randperm(num_steps_loss).tolist()

        nn_model.train()

        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for i in batch_pbar:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            batch_loss = 0.0
            batch_mse_loss = 0.0
            batch_symmetry_loss = 0.0
            batch_force_magnitude_loss = 0.0
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_steps_loss)
            batch_indices = permuted_indices[start_idx:end_idx]
            current_batch_actual_size = len(batch_indices)
            
            for step_idx in batch_indices:
                current_positions = normalized_positions_for_loss[step_idx].to(device)
                current_target_accel = normalized_accel[step_idx].to(device)

                with torch.set_grad_enabled(True):
                    predicted_accels_step, _, _ = sim_model.compute_forces(
                        current_positions * pos_std.to(device) + pos_mean.to(device)
                    )
                    
                    predicted_accels_norm = predicted_accels_step / accel_scale.to(device)
                    
                    mse_loss_step = torch.mean((predicted_accels_norm - current_target_accel)**2)
                    
                    symmetry_loss = 0.0
                    n_particles = current_positions.shape[0]
                    
                    if n_particles > 1 and args.symmetry_weight > 0:
                        num_pairs = min(10, n_particles * (n_particles - 1) // 2)
                        pair_indices = torch.randperm(n_particles, device=device)[:min(num_pairs * 2, n_particles)]
                        
                        for p_idx in range(0, len(pair_indices) - 1, 2):
                            if p_idx + 1 >= len(pair_indices):
                                break
                                
                            i, j = pair_indices[p_idx], pair_indices[p_idx + 1]
                            pos_i = current_positions[i]
                            pos_j = current_positions[j]
                            rel_pos_ij = pos_i - pos_j
                            rel_pos_ji = pos_j - pos_i
                            
                            rel_pos_ij.requires_grad_(True)
                            
                            pot_ij = nn_model.compute_potential(rel_pos_ij.unsqueeze(0)).squeeze(0)
                            
                            pot_ji = nn_model.compute_potential(rel_pos_ji.unsqueeze(0)).squeeze(0)
                            
                            symmetry_loss += torch.mean((pot_ij - pot_ji)**2)
                            
                            total_pot_ij = pot_ij.sum()
                            force_ij = -torch.autograd.grad(
                                total_pot_ij, rel_pos_ij,
                                create_graph=False, retain_graph=True
                            )[0]
                            
                            direction_ij = rel_pos_ij.detach() / (torch.norm(rel_pos_ij.detach()) + 1e-8)
                            projection = torch.sum(force_ij * direction_ij)
                            perpendicular = force_ij - projection * direction_ij
                            symmetry_loss += 0.1 * torch.sum(perpendicular**2)
                        
                        symmetry_loss /= max(1, num_pairs)
                    

                    force_magnitude_loss = 0.0
                    if epoch >= 1:
                        r_cutoff = potential_params.get('r_cutoff', 5.0)
                        
                        test_dists = torch.rand(5, device=device) * (r_cutoff - 0.1) + 0.1
                        test_vectors = torch.zeros((len(test_dists), 3), device=device)
                        test_vectors[:, 0] = test_dists
                        
                        true_test_positions = torch.cat([test_vectors, -test_vectors])
                        true_forces = true_potential.compute_force_only(true_test_positions)[:len(test_vectors)]
                        
                        test_vectors.requires_grad_(True)
                        
                        potentials = nn_model.compute_potential(test_vectors)
                        total_pot = potentials.sum()
                        
                        pred_forces = -torch.autograd.grad(
                            total_pot, test_vectors, 
                            create_graph=False, retain_graph=True
                        )[0]
                        
                        pred_force_magnitudes = torch.norm(pred_forces, dim=1)
                        
                        if zero_force_counter > 2:
                            force_magnitude_loss += torch.mean(torch.exp(-10 * pred_force_magnitudes))
                            
                        force_magnitude_loss += 0.1 * torch.mean((pred_forces - true_forces.to(device))**2)
                    
                    current_sym_weight = args.symmetry_weight
                    if epoch < args.epochs // 4:
                        current_sym_weight *= 0.5
                    elif epoch > args.epochs * 3 // 4:
                        current_sym_weight *= 2.0
                    
                    magnitude_weight = 0.0
                    if zero_force_counter > 2:
                        magnitude_weight = 2.0
                    elif epoch >= 5:
                        magnitude_weight = 0.2
                    
                    mse_loss_clamped = mse_loss_step
                    symmetry_loss_clamped = symmetry_loss
                    
                    loss_scale = 1.0
                    if epoch < 5 and mse_loss_step > 1.0:
                        loss_scale = 0.5
                    
                    combined_loss = loss_scale * (
                        mse_loss_clamped + 
                        current_sym_weight * symmetry_loss_clamped
                    )
                    
                    if magnitude_weight > 0 and force_magnitude_loss > 0:
                        combined_loss = combined_loss + loss_scale * magnitude_weight * force_magnitude_loss
                    
                    if epoch < 10 and (mse_loss_step > 1.0 or symmetry_loss > 1.0):
                        all_potentials = []
                        force_magnitude_penalty = 0.0
                        
                        for i_idx in range(n_particles):
                            for j_idx in range(i_idx+1, n_particles):
                                rel_pos = current_positions[i_idx] - current_positions[j_idx]
                                rel_pos.requires_grad_(True)
                                
                                pot_pred = nn_model.compute_potential(rel_pos.unsqueeze(0)).squeeze(0)
                                all_potentials.append(pot_pred)
                        
                        if all_potentials:
                            potentials_tensor = torch.stack(all_potentials, dim=0)
                            if epoch > 3:
                                force_magnitude_penalty = 0.1 * torch.mean(torch.exp(-5 * torch.abs(potentials_tensor)))
                        
                        if force_magnitude_penalty > 0:
                            combined_loss = combined_loss + force_magnitude_penalty
                
                    step_loss = combined_loss / current_batch_actual_size
                    step_loss.backward()
                    
                batch_mse_loss += mse_loss_step.item()
                batch_symmetry_loss += symmetry_loss if isinstance(symmetry_loss, float) else symmetry_loss.item()
                batch_force_magnitude_loss += force_magnitude_loss if isinstance(force_magnitude_loss, float) else force_magnitude_loss.item()
                batch_loss += combined_loss.item()
                
                del current_positions, current_target_accel, predicted_accels_step, predicted_accels_norm
                if args.symmetry_weight > 0 and n_particles > 1:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), args.clip_grad)

            optimizer.step()

            avg_batch_loss = batch_loss / current_batch_actual_size
            avg_batch_mse = batch_mse_loss / current_batch_actual_size
            avg_batch_sym = batch_symmetry_loss / current_batch_actual_size
            avg_batch_mag = batch_force_magnitude_loss / current_batch_actual_size
            
            if torch.isnan(torch.tensor(avg_batch_loss)) or torch.isinf(torch.tensor(avg_batch_loss)):
                had_nan_loss = True
                print(f"\nWarning: NaN/Inf loss detected in epoch {epoch+1}, batch {i+1}")
                
                if not had_nan_loss:
                    print("Reducing learning rate by 10x and reloading last checkpoint")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    
                    continue
            
            epoch_loss += batch_loss
            epoch_mse_loss += batch_mse_loss
            epoch_symmetry_loss += batch_symmetry_loss
            epoch_force_magnitude_loss += batch_force_magnitude_loss

            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({
                "Loss": f"{avg_batch_loss:.4e}", 
                "MSE": f"{avg_batch_mse:.4e}", 
                "Sym": f"{avg_batch_sym:.4e}",
                "Mag": f"{avg_batch_mag:.4e}",
                "LR": f"{current_lr:.3e}",
                "Mem": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
            })
            
            if args.use_wandb:
                wandb.log({
                    "batch/loss": avg_batch_loss,
                    "batch/mse_loss": avg_batch_mse,
                    "batch/symmetry_loss": avg_batch_sym,
                    "batch/force_magnitude_loss": avg_batch_mag,
                    "batch/learning_rate": current_lr,
                    "batch/memory_usage_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "batch/batch_size": current_batch_actual_size,
                    "batch/global_step": epoch * num_batches + i
                })
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_pbar.close()
        
        avg_epoch_loss = epoch_loss / num_steps_loss
        avg_epoch_mse = epoch_mse_loss / num_steps_loss
        avg_epoch_sym = epoch_symmetry_loss / num_steps_loss
        avg_epoch_mag = epoch_force_magnitude_loss / num_steps_loss

        max_force, mean_force, is_zero_like, force_profile = check_force_profile(nn_model, device)
        force_profile_history.append(force_profile)
        
        if is_zero_like:
            zero_force_counter += 1
            if zero_force_counter >= 3:
                print(f"\nWarning: Model is converging to zero solution for {zero_force_counter} epochs!")
                
                if os.path.exists(args.model_save_path.replace('.pt', '_best.pt')):
                    print("Loading best model and increasing learning rate...")
                    del nn_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    nn_model = PotentialNN(
                        hidden_dim=args.nn_hidden_dim, 
                        num_blocks=args.num_residual_blocks,
                        max_potential=args.max_potential
                    ).to(device)
                    nn_model.load_state_dict(torch.load(args.model_save_path.replace('.pt', '_best.pt')))
                    
                    sim_model.neural_network = nn_model
                    
                    optimizer = optim.AdamW(nn_model.parameters(), lr=args.learning_rate * 5.0, weight_decay=args.weight_decay)
                else:
                    print("Re-initializing network weights...")
                    with torch.no_grad():
                        for name, param in nn_model.named_parameters():
                            if 'output_layer3.weight' in name:
                                param.data.normal_(0, 0.1)
                            elif 'output_layer3.bias' in name:
                                param.data.fill_(-0.1)
                        
                        optimizer = optim.AdamW(nn_model.parameters(), lr=args.learning_rate * 2.0, weight_decay=args.weight_decay)
                
                zero_force_counter = 0
        else:
            zero_force_counter = 0
        
        if scheduler:
            scheduler.step()

        losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4e}, MSE: {avg_epoch_mse:.4e}, Sym: {avg_epoch_sym:.4e}, Mag: {avg_epoch_mag:.4e}, MaxForce: {max_force:.4e}, LR: {current_lr:.3e}")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_epoch_loss,
                "train/mse_loss": avg_epoch_mse,
                "train/symmetry_loss": avg_epoch_sym,
                "train/force_magnitude_loss": avg_epoch_mag,
                "train/learning_rate": current_lr,
                "train/patience_counter": patience_counter,
                "train/max_force": max_force,
                "train/mean_force": mean_force,
                "train/is_zero_solution": is_zero_like,
                "train/time_elapsed": time.time() - start_time
            })

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            best_model_path = args.model_save_path.replace('.pt', '_best.pt')
            print(f"New best model! Saving to {best_model_path}")
            torch.save(nn_model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {args.patience} epochs)")
                break

        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.dirname(args.model_save_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            base, ext = os.path.splitext(args.model_save_path)
            checkpoint_path = f"{base}_epoch{epoch+1}{ext}"
            print(f"\nSaving checkpoint to {checkpoint_path}...")
            torch.save(nn_model.state_dict(), checkpoint_path)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training complete.")
    
    print(f"Saving trained MLP model to {args.model_save_path}...")
    torch.save(nn_model.state_dict(), args.model_save_path)

    if best_loss < avg_epoch_loss:
        print(f"Loading best model from {best_model_path}...")
        nn_model.load_state_dict(torch.load(best_model_path))

    plot_loss_path = os.path.join(output_dir, "ude_training_loss.png")
    print(f"Saving training loss plot to {plot_loss_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Combined)")
    plt.title("UDE Training Loss (Physics-Informed MLP)")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(plot_loss_path)
    plt.close()
    
    plot_force_evolution_path = os.path.join(output_dir, "force_profile_evolution.png")
    print(f"Saving force profile evolution plot to {plot_force_evolution_path}...")
    distances = np.linspace(0.2, 4.0, 20)
    
    plt.figure(figsize=(12, 8))
    
    with torch.no_grad():
        test_dist_tensor = torch.tensor(distances, device=device).float()
        test_vectors = torch.zeros((len(distances), 3), device=device)
        test_vectors[:, 0] = test_dist_tensor
        
        true_forces_plot = []
        for r_vec in test_vectors:
            r_tensor = r_vec.reshape(1, 3)
            true_force = true_potential.compute_force_only(torch.stack([r_tensor[0], -r_tensor[0]]))[0][0].cpu().numpy()
            true_forces_plot.append(true_force)
    
    plt.plot(distances, true_forces_plot, 'k-', linewidth=3, label='True Force')
    
    num_profiles = 5
    indices = np.linspace(0, len(force_profile_history)-1, num_profiles, dtype=int)
    
    cmap = plt.cm.viridis
    for i, idx in enumerate(indices):
        if idx < len(force_profile_history):
            epoch_num = int(idx * args.epochs / (len(force_profile_history)-1))
            color = cmap(i / (num_profiles - 1))
            plt.plot(distances, force_profile_history[idx], '--', color=color, alpha=0.7, 
                     label=f'Epoch {epoch_num+1}')
    
    plt.xlabel('Distance (r)')
    plt.ylabel('Force')
    plt.title('Evolution of Force Profile During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_force_evolution_path)
    plt.close()

    if args.use_wandb:
        wandb.log({
            "train/final_loss": avg_epoch_loss,
            "train/best_loss": best_loss,
            "train/total_epochs": epoch + 1,
            "train/training_time": time.time() - start_time,
            "train/loss_plot": wandb.Image(plot_loss_path),
            "train/force_profile_evolution": wandb.Image(plot_force_evolution_path)
        })
        
        model_artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", 
            type="model",
            description="Trained UDE neural network model"
        )
        model_artifact.add_file(args.model_save_path)
        wandb.log_artifact(model_artifact)

    end_time = time.time()
    print(f"--- UDE Training Finished ({end_time - start_time:.2f}s) ---")

def analyze_results(args):
    print("\n--- Starting Analysis ---")
    start_time = time.time()
    output_dir = args.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_trajectory_path = os.path.join(output_dir, "ude_trajectory_comparison.png")
    plot_force_path = os.path.join(output_dir, "ude_force_comparison.png")
    plot_potential_path = os.path.join(output_dir, "ude_potential_comparison.png")

    if args.use_wandb:
        wandb_config = {
            'phase': 'analysis',
            'model_load_path': args.model_load_path,
            'nn_hidden_dim': args.nn_hidden_dim,
            'device': args.device
        }
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"analyze-ude-{os.path.basename(args.model_load_path)}", 
            config=wandb_config,
            tags=args.wandb_tags + ["analysis"],
            resume="allow"
        )

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
    run_params_loaded = data.get('run_params', {})
    if not run_params_loaded:
        print("Warning: 'run_params' not found in data file. Using defaults for analysis prediction.")
        run_params_loaded = {'dt_initial': 0.001, 'max_steps': 500}
    potential_params_loaded = data['potential_params']

    potential_params_loaded['device'] = device
    print("Data loaded.")

    print(f"Loading trained MLP model from {args.model_load_path}...")
    print("Instantiating MLP model for loading.")
    nn_model = PotentialNN(
        hidden_dim=args.nn_hidden_dim,
        num_blocks=args.num_residual_blocks,
        max_potential=args.max_potential
    ).to(device)

    try:
        nn_model.load_state_dict(torch.load(args.model_load_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print(f"Ensure the --nn_hidden_dim used for analysis matches the saved model.")
        sys.exit(1)
    nn_model.eval()
    print("Trained model loaded.")

    sim_model_ude = Simulation(
        potential=None, neural_network=nn_model,
        t_end=sim_params_loaded.get('t_end', 1.0),
        device=device,
        dt_min=sim_params_loaded.get('dt_min', 1e-16),
        tau_max=sim_params_loaded.get('tau_max', 0.01),
        Imax=sim_params_loaded.get('Imax', 0.1),
        adaptive_dt=False
    )
    sim_model_ude.nucleons['masses'] = masses

    def predict_local(sim, init_pos, init_vel, num_steps, dt_init):
        sim.nucleons['positions'] = init_pos
        sim.nucleons['velocities'] = init_vel
        results = sim.run(save_interval=1, dt_initial=dt_init, max_steps=num_steps)
        return results['positions'][:num_steps]

    print("Running prediction with trained UDE model (using clean initial state from first collision)...")
    num_collisions = data['generation_params'].get('num_collisions', 1)
    num_steps_per_traj = len(data['times']) // num_collisions
    initial_pos_true = true_positions[0]
    initial_vel_true = true_velocities[0]
    times_single_traj = times[:num_steps_per_traj]

    ude_predicted_positions = predict_local(
        sim_model_ude, initial_pos_true, initial_vel_true,
        num_steps=num_steps_per_traj,
        dt_init=run_params_loaded['dt_initial']
    )
    ude_predicted_positions = ude_predicted_positions.detach().cpu()
    print("Prediction complete.")
    
    true_positions_first = true_positions[:num_steps_per_traj]
    trajectory_mse = torch.mean((ude_predicted_positions - true_positions_first.cpu())**2).item()
    trajectory_rmse = trajectory_mse**0.5
    
    if args.use_wandb:
        wandb.log({
            "prediction/trajectory_mse": trajectory_mse,
            "prediction/trajectory_rmse": trajectory_rmse,
        })

    print(f"Plotting trajectory comparison (first collision) to {plot_trajectory_path}...")
    true_positions_first = true_positions[:num_steps_per_traj]
    noisy_positions_first = noisy_positions[:num_steps_per_traj]

    num_particles_to_plot = min(5, true_positions_first.shape[1])
    time_indices = np.linspace(0, num_steps_per_traj-1, num=min(num_steps_per_traj, 200), dtype=int)
    time_points_plot = times_single_traj[time_indices].cpu().numpy()

    plt.figure(figsize=(12, 8))
    for i in range(num_particles_to_plot):
        plt.plot(time_points_plot, ude_predicted_positions[time_indices, i, 0].numpy(), 'r--', alpha=0.8, label=f'UDE Pred (P{i} X)' if i==0 else None)
        plt.plot(time_points_plot, ude_predicted_positions[time_indices, i, 1].numpy(), 'b--', alpha=0.8, label=f'UDE Pred (P{i} Y)' if i==0 else None)
        plt.plot(time_points_plot, true_positions_first[time_indices, i, 0].cpu().numpy(), 'r-', alpha=0.6, label='True (Clean)' if i==0 else None)
        plt.plot(time_points_plot, true_positions_first[time_indices, i, 1].cpu().numpy(), 'b-', alpha=0.6, label=None)
        plt.scatter(time_points_plot, noisy_positions_first[time_indices, i, 0].cpu().numpy(), c='red', marker='.', s=10, alpha=0.3, label='Noisy Data' if i==0 else None)
        plt.scatter(time_points_plot, noisy_positions_first[time_indices, i, 1].cpu().numpy(), c='blue', marker='.', s=10, alpha=0.3, label=None)

    plt.xlabel("Time")
    plt.ylabel("Position (X=Red, Y=Blue)")
    plt.title("UDE Predicted Trajectory vs True Trajectory & Noisy Data (MLP - First Collision)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_trajectory_path)
    
    if args.use_wandb:
        wandb.log({"prediction/trajectory_comparison": wandb.Image(plt)})
    
    plt.close()
    print("Trajectory plot saved.")

    print(f"Plotting force and potential comparison...")
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
    
    true_potential_values = g_rep * torch.exp(-m_rho*r) / r - g_att * torch.exp(-m_pi*r) / r
    
    relative_vectors.requires_grad_(True) 
    
    predicted_potentials = nn_model.compute_potential(relative_vectors)
    
    total_potential = predicted_potentials.sum()
    
    learned_force_vectors = -torch.autograd.grad(
        total_potential, relative_vectors, 
        create_graph=False, retain_graph=True
    )[0]
    
    force_mse = torch.mean((learned_force_vectors - true_force_vectors)**2).item()
    force_rmse = force_mse**0.5
    
    predicted_potentials_norm = predicted_potentials - torch.min(predicted_potentials)
    true_potential_values_norm = true_potential_values - torch.min(true_potential_values)
    
    scaling_factor = torch.max(true_potential_values_norm) / torch.max(predicted_potentials_norm)
    predicted_potentials_scaled = predicted_potentials_norm * scaling_factor
    
    potential_mse = torch.mean((predicted_potentials_scaled - true_potential_values_norm)**2).item()
    potential_rmse = potential_mse**0.5
    
    if args.use_wandb:
        wandb.log({
            "prediction/force_mse": force_mse,
            "prediction/force_rmse": force_rmse,
            "prediction/max_force_diff": torch.max(torch.abs(learned_force_vectors - true_force_vectors)).item(),
            "prediction/potential_mse": potential_mse,
            "prediction/potential_rmse": potential_rmse,
        })
        
        distance_numpy = distances.cpu().numpy()
        true_force_numpy = true_force_vectors[:, 0].cpu().numpy()
        learned_force_numpy = learned_force_vectors[:, 0].cpu().numpy()
        
        force_data = [[d, t, l, abs(t-l)] for d, t, l in zip(
            distance_numpy[::10], 
            true_force_numpy[::10],
            learned_force_numpy[::10]
        )]
        
        force_table = wandb.Table(
            columns=["Distance", "True Force", "Learned Force", "Absolute Error"],
            data=force_data
        )
        
        wandb.log({"prediction/force_comparison_table": force_table})
        
        potential_data = [[d, t, l, abs(t-l)] for d, t, l in zip(
            distance_numpy[::10], 
            true_potential_values_norm.cpu().numpy()[::10],
            predicted_potentials_scaled.cpu().numpy()[::10]
        )]
        
        potential_table = wandb.Table(
            columns=["Distance", "True Potential", "Learned Potential", "Absolute Error"],
            data=potential_data
        )
        
        wandb.log({"prediction/potential_comparison_table": potential_table})

    plt.figure(figsize=(10, 6))
    plt.plot(distances.cpu().numpy(), true_force_vectors[:, 0].cpu().numpy(), 'k-', label='True Potential Force (Fx)')
    plt.plot(distances.cpu().numpy(), learned_force_vectors[:, 0].cpu().numpy(), 'r--', label='Learned MLP Force (Fx)')

    plt.xlabel("Distance (r)")
    plt.ylabel("Force component Fx")
    plt.title("Comparison of Learned Force (MLP) vs True Potential Force")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.ylim(auto=True)
    plt.savefig(plot_force_path)
    
    if args.use_wandb:
        wandb.log({"prediction/force_comparison": wandb.Image(plt)})
    
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances.cpu().numpy(), true_potential_values_norm.cpu().numpy(), 'k-', label='True Potential')
    plt.plot(distances.cpu().numpy(), predicted_potentials_scaled.cpu().numpy(), 'g--', label='Learned MLP Potential')

    plt.xlabel("Distance (r)")
    plt.ylabel("Potential Energy")
    plt.title("Comparison of Learned Potential (MLP) vs True Potential")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_potential_path)
    
    if args.use_wandb:
        wandb.log({"prediction/potential_comparison": wandb.Image(plt)})
    
    plt.close()
    print("Force and potential comparison plots saved.")

    end_time = time.time()
    print(f"--- Analysis Finished ({end_time - start_time:.2f}s) ---")
    
    if args.use_wandb:
        wandb.run.summary["analysis_time"] = end_time - start_time
        wandb.run.summary["trajectory_rmse"] = trajectory_rmse
        wandb.run.summary["force_rmse"] = force_rmse
        wandb.run.summary["potential_rmse"] = potential_rmse
        
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full UDE pipeline: Data Generation -> Training -> Analysis")

    parser.add_argument('--data_file', type=str, default="data/training_data.pt", help='Path to save/load the training data file.')
    parser.add_argument('--model_save_path', type=str, default="ude_output/ude_potential_mlp.pt", help='Path to save the trained UDE MLP model.')
    parser.add_argument('--model_load_path', type=str, default="ude_output/ude_potential_mlp.pt", help='Path to load the trained UDE MLP model for analysis.')
    parser.add_argument('--analysis_output_dir', type=str, default="ude_output", help='Directory to save analysis plots.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    parser.add_argument('--skip_data_gen', action='store_true', help='Skip data generation if data file exists.')
    parser.add_argument('--skip_training', action='store_true', help='Skip training if model file exists.')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip the final analysis step.')
    
    parser.add_argument('--use_wandb', action='store_true', help='Enable logging with Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='ude-nuclear', help='W&B project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name. If not provided, will be auto-generated.')
    parser.add_argument('--wandb_tags', nargs='+', default=[], help='Tags for the W&B run.')

    parser.add_argument('--num_collisions', type=int, default=10, help='Number of collisions to simulate for data generation.')
    parser.add_argument('--max_impact_parameter', type=float, default=3.0, help='Maximum impact parameter for collisions.')
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
    parser.add_argument('--relative_velocity', type=float, default=20.0, help='Initial relative velocity of nuclei.')
    parser.add_argument('--random_velocity', type=float, default=1.0, help='Magnitude of initial random velocities.')
    parser.add_argument('--radius', type=float, default=0.8, help='Radius for initial particle distribution.')
    parser.add_argument('--save_interval', type=int, default=1, help='Simulation save interval.')
    parser.add_argument('--dt_initial', type=float, default=0.001, help='Initial timestep for simulation.')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum simulation steps.')
    parser.add_argument('--noise_level', type=float, default=5e-3, help='Noise level to add to generated data.')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--nn_hidden_dim', type=int, default=64, help='Hidden dimension size for the PotentialNN (MLP).')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size for time steps during training (0 means full batch).')
    parser.add_argument('--use_scheduler', action='store_true', help='Use CosineAnnealingLR learning rate scheduler.')
    parser.add_argument('--checkpoint_interval', type=int, default=0, help='Save checkpoint every N epochs (0 to disable).')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='Max gradient norm for clipping (0 to disable).')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer.')
    parser.add_argument('--num_residual_blocks', type=int, default=2, help='Number of residual blocks in the PotentialNN.')
    parser.add_argument('--symmetry_weight', type=float, default=0.1, help='Weight for symmetry loss in the physics-informed loss function.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--max_potential', type=float, default=100.0, help='Maximum potential value for scaling in the PotentialNN model.')

    args = parser.parse_args()

    overall_start_time = time.time()
    
    if args.use_wandb:
        os.environ["WANDB_SILENT"] = "true"
        print(f"Weights & Biases logging enabled. Project: {args.wandb_project}")
        print(f"Note: Using enhanced PotentialNN model that predicts scalar potential rather than forces directly.")
        print(f"      Forces are computed as the gradient of the potential, ensuring conservation laws.")
    
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

    if not args.skip_analysis:
        analyze_results(args)
    else:
        print("Skipping analysis.")

    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    print(f"\n--- Pipeline Finished ({total_runtime:.2f}s) ---")
    
    if args.use_wandb:
        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"pipeline-summary", 
            config=vars(args),
            tags=args.wandb_tags + ["pipeline_summary"],
            resume="allow"
        ) as run:
            wandb.summary["total_runtime"] = total_runtime
            wandb.summary["data_file"] = args.data_file
            wandb.summary["model_file"] = args.model_save_path
            wandb.summary["skipped_data_gen"] = args.skip_data_gen
            wandb.summary["skipped_training"] = args.skip_training
            wandb.summary["skipped_analysis"] = args.skip_analysis