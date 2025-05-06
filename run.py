import torch
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import numpy
import argparse
import time
import wandb
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

try:
    from ude.ude_network import KANPotentialModel, LossManager, PotentialNN
    from core import Simulation
    from potential import MesonExchangePotential
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ude.ude_network import KANPotentialModel, LossManager, PotentialNN
    from core import Simulation
    from potential import MesonExchangePotential

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

    random_impact_params = numpy.sqrt(numpy.random.random(args.num_collisions)) * args.max_impact_parameter
    
    if args.use_wandb:
        wandb.log({"impact_parameters": wandb.Histogram(random_impact_params)})

    for b in tqdm(random_impact_params, desc="Simulating Collisions"):
        current_setup_params = setup_base_params.copy()
        current_setup_params['impact_parameter'] = b
        sim.setup_impact_parameter(**current_setup_params)

        result = sim.run(**run_params)

        times_coll = torch.tensor(result['times'], dtype=torch.float32).cpu()
        pos_coll = torch.from_numpy(numpy.array(result['positions'])).float().cpu()
        vel_coll = torch.from_numpy(numpy.array(result['velocities'])).float().cpu()

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
            'batch_accumulation_steps': args.batch_accumulation_steps,
            'use_scheduler': args.use_scheduler,
            'weight_decay': args.weight_decay,
            'num_residual_blocks': args.num_residual_blocks,
            'symmetry_weight': args.symmetry_weight,
            'clip_grad': args.clip_grad,
            'patience': args.patience,
            'model_type': args.model_type,
            'device': args.device
        }
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"train-ude-{args.model_type}-{args.nn_hidden_dim}d-{args.epochs}e", 
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
    
    train_indices, val_indices = train_test_split(
        numpy.arange(data['noisy_positions'].shape[0]), 
        test_size=0.1, 
        random_state=42
    )
    
    print(f"Training on {len(train_indices)} samples, validating on {len(val_indices)} samples")

    print("Computing normalization statistics...")
    pos_mean = torch.mean(data['noisy_positions'][train_indices], dim=(0, 1))
    pos_std = torch.std(data['noisy_positions'][train_indices], dim=(0, 1))
    vel_mean = torch.mean(data['noisy_velocities'][train_indices], dim=(0, 1))
    vel_std = torch.std(data['noisy_velocities'][train_indices], dim=(0, 1))
    
    pos_std = torch.max(pos_std, torch.ones_like(pos_std) * 1e-8)
    vel_std = torch.max(vel_std, torch.ones_like(vel_std) * 1e-8)
    
    print("Calculating target accelerations...")
    chunk_size = min(1000, data['noisy_velocities'].shape[0] - 1)
    num_chunks = (data['noisy_velocities'].shape[0] - 1 + chunk_size - 1) // chunk_size
    
    normalized_accel_list = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, data['noisy_velocities'].shape[0] - 1)
        
        norm_vel_chunk_1 = (data['noisy_velocities'][start_idx:end_idx] - vel_mean) / vel_std
        norm_vel_chunk_2 = (data['noisy_velocities'][start_idx+1:end_idx+1] - vel_mean) / vel_std
        
        accel_chunk = (norm_vel_chunk_2 - norm_vel_chunk_1) / dt
        normalized_accel_list.append(accel_chunk)
        
        del norm_vel_chunk_1, norm_vel_chunk_2, accel_chunk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    normalized_accel = torch.cat(normalized_accel_list, dim=0)
    del normalized_accel_list
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accel_scale = vel_std / (dt * pos_std)
    
    normalized_positions = (data['noisy_positions'] - pos_mean) / pos_std
    normalized_positions_for_loss = normalized_positions[:-1]
    
    accel_magnitudes = torch.norm(normalized_accel, dim=-1)
    max_accel = torch.max(accel_magnitudes).item()
    mean_accel = torch.mean(accel_magnitudes).item()
    print(f"Target acceleration stats: Max={max_accel:.4f}, Mean={mean_accel:.4f}")
    
    potential_params = data['potential_params'].copy()
    potential_params['device'] = device
    true_potential = MesonExchangePotential(**potential_params)
    
    print("Initializing neural network and simulation...")
    if args.model_type == 'potential_nn':
        print("Using enhanced PotentialNN model with physics-informed architecture.")
        nn_model = PotentialNN(
            hidden_dim=args.nn_hidden_dim, 
            num_blocks=args.num_residual_blocks,
            max_potential=args.max_potential,
            dropout_rate=args.dropout_rate
        ).to(device)
    elif args.model_type == 'kan':
        print("Using Kolmogorov-Arnold Network (KAN) model for potential learning.")
        try:
            nn_model = KANPotentialModel(
                hidden_dim=args.nn_hidden_dim,
                num_layers=args.num_residual_blocks,
                max_potential=args.max_potential,
                device=device
            )
            
            init_model = KANPotentialModel(
                hidden_dim=args.nn_hidden_dim,
                num_layers=args.num_residual_blocks,
                max_potential=args.max_potential,
                device=device
            )
        except ImportError as e:
            print(f"Error: KAN model requires pykan library. Falling back to PotentialNN: {e}")
            nn_model = PotentialNN(
                hidden_dim=args.nn_hidden_dim, 
                num_blocks=args.num_residual_blocks,
                max_potential=args.max_potential,
                dropout_rate=args.dropout_rate
            ).to(device)
            
            init_model = PotentialNN(
                hidden_dim=args.nn_hidden_dim,
                num_blocks=args.num_residual_blocks,
                max_potential=args.max_potential,
                dropout_rate=args.dropout_rate
            ).to(device)
            
            args.model_type = 'potential_nn'
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    loss_manager = LossManager(
        potential_weight=0.5,
        force_weight=1.0,
        symmetric_weight=args.symmetry_weight
    )
    
    print("Pre-initializing weights to match true potential curve...")
    r_cutoff = potential_params.get('r_cutoff', 5.0)
    
    test_dists_close = torch.linspace(0.01, 1.0, 40, device=device)
    test_dists_far = torch.linspace(1.0, r_cutoff, 40, device=device)
    test_dists = torch.cat([test_dists_close, test_dists_far[1:]])
    
    test_vectors = torch.zeros((len(test_dists), 3), device=device)
    test_vectors[:, 0] = test_dists
    
    g_att = potential_params['g_att']
    g_rep = potential_params['g_rep']
    m_pi = potential_params['m_pi']
    m_rho = potential_params['m_rho']
    
    true_potential_values = g_rep * torch.exp(-m_rho*test_dists) / test_dists - g_att * torch.exp(-m_pi*test_dists) / test_dists
    true_potential_values -= torch.min(true_potential_values)

    term1 = g_rep * torch.exp(-m_rho*test_dists) * (m_rho/test_dists + 1/(test_dists**2))
    term2 = g_att * torch.exp(-m_pi*test_dists) * (m_pi/test_dists + 1/(test_dists**2))
    true_force_magnitudes = term1 - term2
    
    true_forces = torch.zeros_like(test_vectors)
    true_forces[:, 0] = true_force_magnitudes
    
    init_optimizer = optim.Adam(init_model.parameters(), lr=0.01)
    
    print("Pre-training neural network...")
    for pre_epoch in tqdm(range(30), desc="Pre-training"):
        init_optimizer.zero_grad()
        
        with torch.enable_grad():
            test_vectors_clone = test_vectors.clone().requires_grad_(True)
            
            if args.model_type == 'kan':
                pred_potentials = init_model.compute_potential(test_vectors_clone)
                pred_min = torch.min(pred_potentials)
                pred_potentials = pred_potentials - pred_min
                
                max_true = torch.max(true_potential_values)
                max_pred = torch.max(pred_potentials.detach())
                
                if max_pred > 1e-6:
                    pred_scale = max_true / max_pred
                else:
                    pred_scale = 1.0
                    
                pred_potentials_scaled = pred_potentials * pred_scale
                pot_loss = F.mse_loss(pred_potentials_scaled, true_potential_values)

                total_potential_sum = torch.sum(pred_potentials_scaled)
                pred_forces = -torch.autograd.grad(
                    total_potential_sum, test_vectors_clone,
                    create_graph=True, retain_graph=True
                )[0]
                
                weights = 1.0 / (test_dists.detach() + 0.5)
                weights = weights / weights.sum() 
                force_diff = (pred_forces - true_forces) ** 2
                force_loss = torch.sum(weights.unsqueeze(1) * force_diff) 

                combined_loss = pot_loss + 5.0 * force_loss 
                combined_loss.backward()
                
                total_pot_loss = pot_loss.item()
                total_force_loss = force_loss.item()
            else:
                pred_potentials = init_model.compute_potential(test_vectors_clone)
                
                pred_min = torch.min(pred_potentials)
                pred_potentials = pred_potentials - pred_min
                pred_scale = torch.max(true_potential_values) / torch.max(pred_potentials.detach())
                pred_potentials_scaled = pred_potentials * pred_scale
                
                pot_loss = F.mse_loss(pred_potentials_scaled, true_potential_values)
                
                total_potential = torch.sum(pred_potentials_scaled)
                pred_forces = -torch.autograd.grad(
                    total_potential, test_vectors_clone, 
                    create_graph=True, retain_graph=True
                )[0]
                
                weights = 1.0 / (test_dists.detach() + 0.5)
                weights = weights / weights.sum()
                force_diff = (pred_forces - true_forces) ** 2
                force_loss = torch.sum(weights.unsqueeze(1) * force_diff)
                
                combined_loss = pot_loss + 5.0 * force_loss
                combined_loss.backward()
                
                total_pot_loss = pot_loss.item()
                total_force_loss = force_loss.item()
        
        torch.nn.utils.clip_grad_norm_(init_model.parameters(), 1.0)
        init_optimizer.step()
        
        if args.model_type == 'kan':
            print(f"  Pre-train epoch {pre_epoch+1}: Pot Loss={total_pot_loss:.6f}, Force Loss={total_force_loss:.6f}")
    
    with torch.no_grad():
        if args.model_type == 'kan':
            pred_potentials = []
            batch_size = 8
            for i in range(0, len(test_vectors), batch_size):
                batch_vectors = test_vectors[i:i+batch_size]
                batch_potentials = init_model.compute_potential(batch_vectors)
                pred_potentials.append(batch_potentials)
            pred_potentials = torch.cat(pred_potentials, dim=0)
        else:
            pred_potentials = init_model.compute_potential(test_vectors)
            
        pred_min = torch.min(pred_potentials)
        pred_potentials = pred_potentials - pred_min
        pred_scale = torch.max(true_potential_values) / torch.max(pred_potentials)
        pred_potentials_scaled = pred_potentials * pred_scale
        
        pot_error = F.mse_loss(pred_potentials_scaled, true_potential_values).item()
        print(f"Pre-trained potential error: {pot_error:.6f}")
    
    nn_model.load_state_dict(init_model.state_dict())
    
    del init_model, init_optimizer, test_vectors_clone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    sim_model = Simulation(
        potential=None, neural_network=nn_model,
        t_end=1.0,
        device=device,
        dt_min=1e-16, tau_max=0.01, Imax=0.1,
        adaptive_dt=False
    )
    sim_model.nucleons['masses'] = masses
    
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(nn_model.parameters(), 
                              lr=args.learning_rate, 
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(nn_model.parameters(),
                              lr=args.learning_rate,
                              betas=(0.9, 0.999))
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(nn_model.parameters(),
                             lr=args.learning_rate,
                             momentum=0.9,
                             nesterov=True)
    else:
        print(f"Unknown optimizer: {args.optimizer}, falling back to AdamW")
        optimizer = optim.AdamW(nn_model.parameters(), 
                              lr=args.learning_rate, 
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))
    
    if args.use_wandb:
        wandb.watch(nn_model, log="all", log_freq=10)
    
    if args.use_scheduler:
        if args.scheduler_type == 'one_cycle':
            print(f"Using OneCycleLR scheduler for faster convergence")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=args.learning_rate * 10,
                total_steps=args.epochs,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=100.0
            )
        elif args.scheduler_type == 'cosine':
            print(f"Using CosineAnnealingWarmRestarts scheduler")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.epochs // 10 + 5,
                T_mult=2,
                eta_min=args.learning_rate * 0.01
            )
        elif args.scheduler_type == 'reduce_on_plateau':
            print(f"Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                threshold=0.0001,
                verbose=True
            )
        else:
            print(f"Unknown scheduler type: {args.scheduler_type}, using OneCycleLR")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=args.learning_rate * 10,
                total_steps=args.epochs,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=100.0
            )
    else:
        scheduler = None
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.learning_rate
        param_group['lr'] = args.learning_rate * 5

    losses = []
    val_losses = []
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
    best_val_loss = float('inf')
    patience_counter = 0
    zero_force_counter = 0
    
    def check_force_profile(model, device):
        """Check if model is trending toward zero solution and get force profile"""
        
        distances = torch.linspace(0.2, 4.0, 40, device=device)
        test_vectors = torch.zeros((len(distances), 3), device=device)
        test_vectors[:, 0] = distances
        
        test_vectors_grad = test_vectors.clone().requires_grad_(True)
        
        if isinstance(model, (KANPotentialModel, PotentialNN)):
            potentials = model.compute_potential(test_vectors_grad)
            total_potential = potentials.sum()
            forces = -torch.autograd.grad(
                total_potential, test_vectors_grad, 
                create_graph=False,
                retain_graph=False
            )[0]
            force_x = forces[:, 0]
        else:
             print("Warning: Unknown model type in check_force_profile")
             potentials = torch.zeros_like(distances, device=device)
             force_x = torch.zeros_like(distances, device=device)

        with torch.no_grad():
            g_att = potential_params['g_att']
            g_rep = potential_params['g_rep']
            m_pi = potential_params['m_pi']
            m_rho = potential_params['m_rho']
            term1 = g_rep * torch.exp(-m_rho*distances) * (m_rho/distances + 1/(distances**2))
            term2 = g_att * torch.exp(-m_pi*distances) * (m_pi/distances + 1/(distances**2))
            true_force_x = term1 - term2
            
            max_force = torch.max(torch.abs(force_x)).item()
            mean_force = torch.mean(torch.abs(force_x)).item()
            force_error = torch.mean((force_x - true_force_x) ** 2).item()
            is_zero_like = max_force < 0.1
            
            potentials_np = potentials.cpu().numpy()
            force_x_np = force_x.cpu().numpy()
            
        return max_force, mean_force, is_zero_like, force_x_np, force_error, potentials_np
    
    potential_params = data['potential_params']
    g_att = potential_params['g_att']
    g_rep = potential_params['g_rep']
    m_pi = potential_params['m_pi']
    m_rho = potential_params['m_rho']

    initial_max_force, initial_mean_force, _, _, initial_force_error, initial_potentials = check_force_profile(nn_model, device)
    print(f"Initial force profile: Max={initial_max_force:.4f}, Mean={initial_mean_force:.4f}, Error={initial_force_error:.4f}")
    
    train_losses_epoch = []
    val_losses_epoch = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_symmetry_loss = 0.0
        
        permuted_indices = torch.randperm(num_steps_loss).tolist()

        nn_model.train()

        effective_batch_size = max(1, args.batch_size // (4 if args.model_type == 'kan' else 1))
        accumulation_steps = args.batch_accumulation_steps
        num_batches = (num_steps_loss + effective_batch_size - 1) // effective_batch_size
        
        print(f"Using effective batch size: {effective_batch_size}, accumulation steps: {accumulation_steps}")
        batch_pbar = tqdm(range(0, num_batches, accumulation_steps), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for i in batch_pbar:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            batch_loss = 0.0
            batch_mse_loss = 0.0
            batch_symmetry_loss = 0.0
            
            actual_accumulation_steps = min(accumulation_steps, num_batches - i)
            
            for acc_step in range(actual_accumulation_steps):
                batch_idx = i + acc_step
                start_idx = batch_idx * effective_batch_size
                end_idx = min((batch_idx + 1) * effective_batch_size, num_steps_loss)
                batch_indices = permuted_indices[start_idx:end_idx]
                current_batch_actual_size = len(batch_indices)
                
                if current_batch_actual_size == 0:
                    continue
                
                current_positions_batch = normalized_positions_for_loss[batch_indices].to(device)
                current_target_accel_batch = normalized_accel[batch_indices].to(device)

                with torch.set_grad_enabled(True):
                    unnorm_positions_batch = current_positions_batch * pos_std.to(device) + pos_mean.to(device)
                    predicted_accels_batch, _, _ = sim_model.compute_forces(unnorm_positions_batch)
                    
                    predicted_accels_norm_batch = predicted_accels_batch / accel_scale.to(device)
                    
                    n_particles = current_positions_batch.shape[1]
                    batch_actual_size = current_positions_batch.shape[0]
                    
                    mse_loss_batch = F.huber_loss(
                        predicted_accels_norm_batch.reshape(-1, 3),
                        current_target_accel_batch.reshape(-1, 3),
                        delta=1.0,
                        reduction='mean' 
                    )
                    
                    symmetry_loss_tensor = torch.tensor(0.0, device=device)
                    
                    if n_particles > 1:
                        num_pairs_per_instance = min(3 if args.model_type == 'kan' else 5, n_particles * (n_particles - 1) // 2) 
                        
                        pair_indices_batch = torch.randint(0, n_particles, 
                                                          (batch_actual_size, num_pairs_per_instance * 2), 
                                                          device=device)
                        
                        all_rel_pos_ij = []
                        
                        for b_idx in range(batch_actual_size):
                            instance_positions = current_positions_batch[b_idx]
                            instance_pairs = pair_indices_batch[b_idx]
                            
                            for p_idx in range(0, len(instance_pairs), 2):
                                if p_idx + 1 >= len(instance_pairs):
                                     break
                                i_idx, j_idx = instance_pairs[p_idx], instance_pairs[p_idx + 1]
                                if i_idx == j_idx: continue
                                
                                pos_i = instance_positions[i_idx]
                                pos_j = instance_positions[j_idx]
                                rel_pos_ij = pos_i - pos_j
                                r_norm = torch.norm(rel_pos_ij)

                                if r_norm > 1e-6:
                                    all_rel_pos_ij.append(rel_pos_ij.unsqueeze(0))

                        if len(all_rel_pos_ij) > 0:
                            all_rel_pos_ij_tensor = torch.cat(all_rel_pos_ij, dim=0).requires_grad_(True)
                            
                            symmetry_loss_tensor = loss_manager.symmetry_loss(nn_model, all_rel_pos_ij_tensor)
                        else: 
                            symmetry_loss_tensor = torch.tensor(0.0, device=device)
                             
                    else: 
                         symmetry_loss_tensor = torch.tensor(0.0, device=device)
                         
                    if epoch < args.epochs // 10:
                        mse_weight = 1.0
                        symmetry_weight = args.symmetry_weight * 0.2
                    elif epoch > args.epochs * 0.8:
                        mse_weight = 0.8
                        symmetry_weight = args.symmetry_weight * 1.5
                    else:
                        mse_weight = 1.0
                        symmetry_weight = args.symmetry_weight
                    
                    if args.model_type == 'kan':
                        symmetry_weight *= 0.3 
                        
                    combined_loss_batch = (
                        mse_weight * mse_loss_batch + 
                        symmetry_weight * symmetry_loss_tensor
                    )
                
                accumulation_scale = 1.0 / actual_accumulation_steps
                scaled_loss = combined_loss_batch * accumulation_scale
                scaled_loss.backward() 
                
                batch_mse_loss += mse_loss_batch.item() * accumulation_scale
                batch_symmetry_loss += symmetry_loss_tensor.item() * accumulation_scale 
                batch_loss += combined_loss_batch.item() * accumulation_scale
                
                del current_positions_batch, current_target_accel_batch, predicted_accels_batch, predicted_accels_norm_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), args.clip_grad)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 

            avg_batch_loss = batch_loss 
            avg_batch_mse = batch_mse_loss
            avg_batch_sym = batch_symmetry_loss
            
            if torch.isnan(torch.tensor(avg_batch_loss)) or torch.isinf(torch.tensor(avg_batch_loss)):
                print(f"\nWarning: NaN/Inf loss detected in epoch {epoch+1}, batch {i+1}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                continue
            
            epoch_loss += batch_loss
            epoch_mse_loss += batch_mse_loss
            epoch_symmetry_loss += batch_symmetry_loss

            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({
                "Loss": f"{avg_batch_loss:.4e}", 
                "MSE": f"{avg_batch_mse:.4e}", 
                "Sym": f"{avg_batch_sym:.4e}",
                "LR": f"{current_lr:.3e}",
                "Mem": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
            })
            
            if args.use_wandb:
                wandb.log({
                    "batch/loss": avg_batch_loss,
                    "batch/mse_loss": avg_batch_mse,
                    "batch/symmetry_loss": avg_batch_sym,
                    "batch/learning_rate": current_lr,
                    "batch/memory_usage_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "batch/batch_size": current_batch_actual_size,
                    "batch/global_step": epoch * num_batches + i
                })
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        batch_pbar.close()
        
        nn_model.eval()
        val_loss = 0.0
        val_batches = 0
        
        val_batch_size = min(16, len(val_indices)) if args.model_type == 'kan' else len(val_indices)
        
        for i in range(0, len(val_indices), val_batch_size):
            batch_val_indices = val_indices[i:i+val_batch_size]
            val_loss_batch = 0.0
            
            for val_idx in batch_val_indices:
                if val_idx >= len(normalized_positions_for_loss):
                    continue
                    
                val_pos = normalized_positions_for_loss[val_idx].to(device)
                val_target_accel = normalized_accel[val_idx].to(device)
                
                val_pred_accels, _, _ = sim_model.compute_forces(
                    val_pos * pos_std.to(device) + pos_mean.to(device)
                )
                
                val_pred_accels_norm = val_pred_accels / accel_scale.to(device)
                
                val_loss_step = F.huber_loss(
                    val_pred_accels_norm, 
                    val_target_accel,
                    delta=1.0
                )
                
                val_loss_batch += val_loss_step.item()
                val_batches += 1
                
                del val_pos, val_target_accel, val_pred_accels, val_pred_accels_norm
                
            val_loss += val_loss_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_val_loss = val_loss / max(1, val_batches)
        
        avg_epoch_loss = epoch_loss / num_steps_loss
        avg_epoch_mse = epoch_mse_loss / num_steps_loss
        avg_epoch_sym = epoch_symmetry_loss / num_steps_loss

        max_force, mean_force, is_zero_like, force_profile, force_error, potentials = check_force_profile(nn_model, device)
        force_profile_history.append(force_profile)
        
        train_losses_epoch.append(avg_epoch_loss)
        val_losses_epoch.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, " 
              f"MSE: {avg_epoch_mse:.6f}, Sym: {avg_epoch_sym:.6f}, " 
              f"Force Error: {force_error:.6f}, Max Force: {max_force:.6f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": avg_val_loss,
                "epoch/mse_loss": avg_epoch_mse,
                "epoch/symmetry_loss": avg_epoch_sym,
                "epoch/force_error": force_error,
                "epoch/max_force": max_force,
                "epoch/mean_force": mean_force,
                "epoch/is_zero_like": int(is_zero_like),
                "epoch/learning_rate": optimizer.param_groups[0]['lr'],
                "epoch/force_profile": wandb.plot.line_series(
                    xs=numpy.linspace(0.2, 4.0, 40),
                    ys=[force_profile, 
                        term1.cpu().numpy() - term2.cpu().numpy()],
                    keys=["Predicted", "True"],
                    title="Force Profile",
                    xname="Distance"
                ),
                "epoch/potential_profile": wandb.plot.line_series(
                    xs=numpy.linspace(0.2, 4.0, 40),
                    ys=[potentials],
                    keys=["Potential"],
                    title="Potential Profile",
                    xname="Distance"
                )
            })
        
        if is_zero_like:
            zero_force_counter += 1
            print(f"\nWarning: Model is converging to zero solution for {zero_force_counter} epochs! Max force: {max_force:.6f}")
            
            if zero_force_counter >= 3:
                print("Applying corrective action for zero solution...")
                
                if os.path.exists(args.model_save_path.replace('.pt', '_best.pt')):
                    print("Loading best model and applying reinitialization...")
                    nn_model.load_state_dict(torch.load(args.model_save_path.replace('.pt', '_best.pt')))
                    
                    with torch.no_grad():
                        for name, param in nn_model.named_parameters():
                            if 'output_layer3.weight' in name or 'scaling_factor' in name:
                                param.data *= 10.0
                                
                    optimizer = optim.AdamW(nn_model.parameters(), 
                                          lr=args.learning_rate * 10.0, 
                                          weight_decay=args.weight_decay * 0.1,
                                          betas=(0.9, 0.999))
                else:
                    print("No best model found. Reinitializing weights...")
                    with torch.no_grad():
                        for name, param in nn_model.named_parameters():
                            if 'output_layer3.weight' in name or 'scaling_factor' in name:
                                param.data *= 10.0
                    
                    zero_force_counter = 0
        else:
            zero_force_counter = 0
                
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"New best validation loss: {best_val_loss:.6f}. Saving model...")
            torch.save(nn_model.state_dict(), args.model_save_path.replace('.pt', '_best.pt'))
            
            if force_error < best_loss:
                best_loss = force_error
                print(f"New best force error: {best_loss:.6f}. Saving model...")
                torch.save(nn_model.state_dict(), args.model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement")
                break
                
        if args.use_scheduler:
            if args.scheduler_type == 'reduce_on_plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    print("Training complete.")
    
    print(f"Saving trained MLP model to {args.model_save_path}...")
    torch.save(nn_model.state_dict(), args.model_save_path)

    if args.model_type == 'kan':
        kan_metadata = {
            'hidden_dim': args.nn_hidden_dim,
            'num_layers': args.num_residual_blocks,
            'max_potential': args.max_potential
        }
        torch.save(kan_metadata, args.model_save_path.replace('.pt', '_metadata.pt'))

    if best_loss < avg_epoch_loss:
        print(f"Loading best model from {args.model_save_path}...")
        nn_model.load_state_dict(torch.load(args.model_save_path))

    plot_loss_path = os.path.join(output_dir, "ude_training_loss.png")
    print(f"Saving training loss plot to {plot_loss_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_epoch, label="Training Loss")
    plt.plot(val_losses_epoch, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UDE Training and Validation Loss")
    plt.legend()
    plt.savefig(plot_loss_path)
    plt.close()
    
    plot_force_evolution_path = os.path.join(output_dir, "force_profile_evolution.png")
    print(f"Saving force profile evolution plot to {plot_force_evolution_path}...")
    distances = numpy.linspace(0.2, 4.0, 40)
    
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
    indices = numpy.linspace(0, len(force_profile_history)-1, num_profiles, dtype=int)
    
    cmap = plt.cm.viridis
    for i, idx in enumerate(indices):
        if idx < len(force_profile_history):
            epoch_num = int(idx * args.epochs / (len(force_profile_history)))
            color = cmap(i / (num_profiles))
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
            "train/total_epochs": args.epochs,
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

    if args.model_type == 'kan':
        print("\n--- Extracting Symbolic Formula from KAN Model ---")
        try:
            kan_formula = nn_model.get_symbolic_formula(precision=4, simplify=True)
            print(f"\nСимволическая формула потенциала (KAN):")
            print(f"{kan_formula}")
            
            formula_file = args.model_save_path.replace('.pt', '_formula.txt')
            with open(formula_file, 'w') as f:
                f.write(f"KAN Symbolic Formula:\n{kan_formula}\n\n")
                f.write(f"Model Parameters:\n")
                f.write(f"hidden_dim: {args.nn_hidden_dim}\n")
                f.write(f"num_residual_blocks: {args.num_residual_blocks}\n")
                f.write(f"epochs: {args.epochs}\n")
                f.write(f"learning_rate: {args.learning_rate}\n")
                f.write(f"max_potential: {args.max_potential}\n")
            
            print(f"Формула сохранена в {formula_file}")
            
            try:
                from sympy import symbols, sympify, lambdify
                
                plt.figure(figsize=(12, 6))
                
                g_att = potential_params['g_att']
                g_rep = potential_params['g_rep']
                m_pi = potential_params['m_pi']
                m_rho = potential_params['m_rho']
                
                r = numpy.linspace(0.2, r_cutoff, 200)
                true_potential = g_rep * numpy.exp(-m_rho*r) / r - g_att * numpy.exp(-m_pi*r) / r
                plt.plot(r, true_potential, 'k-', label='True Potential', linewidth=2)
                
                try:
                    r_sym = symbols('r_norm')
                    can_be_evaluated = False
                    
                    kan_formula_clean = kan_formula
                    for term in ["torch.", "tanh(", ")", "["]:
                        kan_formula_clean = kan_formula_clean.replace(term, "")
                    
                    kan_expr = sympify(kan_formula_clean)
                    kan_func = lambdify(r_sym, kan_expr, 'numpy')
                    
                    can_be_evaluated = True
                except Exception as eval_err:
                    print(f"Не удалось оценить формулу как выражение sympy: {eval_err}")
                    can_be_evaluated = False
                
                if can_be_evaluated:
                    try:
                        kan_potential = kan_func(r)
                        plt.plot(r, kan_potential, 'r--', label='KAN Formula', linewidth=2)
                    except Exception as e:
                        print(f"Ошибка при вычислении значений KAN формулы: {e}")
                
                with torch.no_grad():
                    r_tensor = torch.from_numpy(r).float().to(device)
                    zeros = torch.zeros_like(r_tensor)
                    
                    r_vectors = torch.stack([r_tensor, zeros, zeros], dim=-1)
                    
                    kan_potentials = nn_model.compute_potential(r_vectors).cpu().numpy()
                    plt.plot(r, kan_potentials, 'b--', label='KAN Model', linewidth=2)
                
                plt.xlabel('Distance (r)')
                plt.ylabel('Potential')
                plt.title('Comparison of True Potential vs KAN Symbolic Formula vs KAN Model')
                plt.legend()
                plt.grid(True)
                
                formula_plot_file = args.model_save_path.replace('.pt', '_formula_comparison.png')
                plt.savefig(formula_plot_file)
                plt.close()
                
                print(f"Сравнительный график сохранен в {formula_plot_file}")
                
                if args.use_wandb:
                    wandb.log({
                        "kan_symbolic_formula": kan_formula,
                        "kan_formula_comparison": wandb.Image(formula_plot_file)
                    })
            except ImportError as e:
                print(f"Не удалось создать сравнительный график: {e}")
                
        except Exception as e:
            print(f"Ошибка при извлечении символической формулы: {e}")
            
def analyze_results(args):
    print("\n--- Starting Analysis ---")
    start_time = time.time()
    output_dir = args.analysis_output_dir if hasattr(args, 'analysis_output_dir') else os.path.dirname(args.model_save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    model_dir = os.path.dirname(args.model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
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
            name=args.wandb_run_name or f"analyze-ude-{os.path.basename(args.model_load_path or args.model_save_path)}", 
            config=wandb_config,
            tags=args.wandb_tags + ["analysis"],
            resume="allow"
        )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    model_path = args.model_load_path if args.model_load_path else args.model_save_path
    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        if args.mode == "analyze_results":
            print("Пожалуйста, сначала запустите обучение модели или укажите правильный путь.")
            return
        else:
            print("Режим анализа будет пропущен, т.к. модель еще не обучена.")
            return

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

    print(f"Loading trained MLP model from {model_path}...")
    print("Instantiating MLP model for loading.")
    nn_model = PotentialNN(
        hidden_dim=args.nn_hidden_dim,
        num_blocks=args.num_residual_blocks,
        max_potential=args.max_potential
    ).to(device)

    try:
        nn_model.load_state_dict(torch.load(model_path, map_location=device))
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
    time_indices = numpy.linspace(0, num_steps_per_traj-1, num=min(num_steps_per_traj, 200), dtype=int)
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
    parser = argparse.ArgumentParser(description='Universal Differential Equation (UDE) for nucleon simulation')
    parser.add_argument('--mode', type=str, required=True, choices=['generate_data', 'train_ude', 'analyze_results'], help='Program operation mode')
    parser.add_argument('--model_save_path', type=str, default='./models/ude_model.pt', help='Path to save/load the trained model')
    parser.add_argument('--model_load_path', type=str, default=None, help='Path to load the model for analysis (defaults to model_save_path if not provided)')
    parser.add_argument('--data_file', type=str, default='./data/ude_data.pt', help='Path to save/load the generated data')
    parser.add_argument('--analysis_output_dir', type=str, default='./analysis', help='Directory to save analysis output files')
    parser.add_argument('--g_att', type=float, default=13.5, help='Attractive coupling constant')
    parser.add_argument('--g_rep', type=float, default=20.0, help='Repulsive coupling constant')
    parser.add_argument('--m_pi', type=float, default=0.7, help='Pion mass parameter')
    parser.add_argument('--m_rho', type=float, default=3.93, help='Rho mass parameter')
    parser.add_argument('--r_cutoff', type=float, default=5.0, help='Cutoff radius for potential')
    parser.add_argument('--r_core', type=float, default=0.3, help='Repulsive core radius')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for NN training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training. Set to 0 for auto-determination.')
    parser.add_argument('--batch_accumulation_steps', type=int, default=4, help='Number of batches to accumulate gradients for (reduces memory usage)')
    parser.add_argument('--nn_hidden_dim', type=int, default=32, help='Hidden dimension of the neural network')
    parser.add_argument('--num_residual_blocks', type=int, default=2, help='Number of residual blocks in the neural network')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='physics-ude', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom name for the W&B run')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Tags for wandb run')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--n_particles', type=int, default=4, help='Number of particles in the simulation')
    parser.add_argument('--relative_velocity', type=float, default=1.0, help='Relative velocity of the nuclei in the x direction')
    parser.add_argument('--random_velocity', type=float, default=0.1, help='Random velocity components')
    parser.add_argument('--radius', type=float, default=1.5, help='Nucleus radius')
    parser.add_argument('--t_end', type=float, default=15.0, help='End time of the simulation')
    parser.add_argument('--dt_min', type=float, default=1e-16, help='Minimum time step in the adaptive scheme')
    parser.add_argument('--tau_max', type=float, default=0.01, help='Maximum difference in the RKQ4 error control')
    parser.add_argument('--Imax', type=float, default=0.1, help='Safety factor in the adaptive time step')
    parser.add_argument('--num_collisions', type=int, default=1, help='Number of collision simulations for data generation')
    parser.add_argument('--max_impact_parameter', type=float, default=3.0, help='Maximum impact parameter for collisions (actual value sampled stochastically)')
    parser.add_argument('--noise_level', type=float, default=0.02, help='Level of Gaussian noise to add to the data')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving simulation data')
    parser.add_argument('--dt_initial', type=float, default=0.001, help='Initial time step')
    parser.add_argument('--max_steps', type=int, default=20000, help='Maximum number of simulation steps')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save model checkpoints every N epochs. Set to 0 to disable.')
    parser.add_argument('--symmetry_weight', type=float, default=0.5, help='Weight for symmetry loss component')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping (increased).')
    parser.add_argument('--max_potential', type=float, default=50.0, help='Maximum potential value for scaling in the PotentialNN model (reduced).')
    parser.add_argument('--model_type', type=str, default='potential_nn', choices=['potential_nn', 'kan'], help='Type of neural network model to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for the PotentialNN model.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'], help='Optimizer to use.')
    parser.add_argument('--scheduler_type', type=str, default='one_cycle', choices=['one_cycle', 'cosine', 'reduce_on_plateau'], help='Scheduler type to use.')
    parser.add_argument('--plot_results', action='store_true', help='Create plots of results')
    parser.add_argument('--prediction_steps', type=int, default=500, help='Number of steps for prediction trajectory')
    parser.add_argument('--use_true_potential', action='store_true', help='Use the true potential for analysis comparison')
    
    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        wandb.login()

    if args.mode == 'generate_data':
        generate_data(args)
    elif args.mode == 'train_ude':
        train_ude(args)
    elif args.mode == 'analyze_results':
        analyze_results(args)
    else:
        print(f"Unknown mode: {args.mode}")