import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import kan
except ImportError:
    print("Warning: pykan not installed. KAN models will not be available.")
    kan = None

class ScaleSILU(nn.Module):
    """Scaled SiLU activation for better gradient flow"""
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return F.silu(x) * self.scale

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = ScaleSILU(scale=1.414)
        
    def forward(self, x):
        identity = x
        out = self.lin1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.norm2(out)
        out = self.activation(out + identity)
        return out

class PotentialNN(nn.Module):
    def __init__(self, hidden_dim=16, num_blocks=4, input_dim=3, max_potential=100.0, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.max_potential = max_potential
        
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            ScaleSILU()
        )
        
        self.inverse_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            ScaleSILU()
        )
        
        self.inverse_squared_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            ScaleSILU()
        )
        
        self.direction_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            ScaleSILU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        self.output_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm1 = nn.LayerNorm(hidden_dim)
        self.output_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.output_layer3 = nn.Linear(hidden_dim // 2, 1)
        
        self.scaling_factor = nn.Parameter(torch.ones(1))
        self.r_cutoff = nn.Parameter(torch.tensor(5.0))
        
        self._init_weights()
        
    def _init_weights(self):
        """Improved weight initialization for stable training"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.normal_(self.output_layer3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer3.bias)

    def compute_potential(self, r_vectors):
        """
        Computes scalar potential for given distance vectors with improved features
        
        Args:
            r_vectors: Tensor of distance vectors [batch_size, 3]
            
        Returns:
            Tensor of scalar potentials [batch_size]
        """
        distances = torch.norm(r_vectors, dim=-1, keepdim=True)
        inverse_distances = 1.0 / (distances + 1e-6)
        inverse_squared = 1.0 / (distances**2 + 1e-6)
        
        directions = r_vectors / (distances + 1e-6)
        
        d_embedding = self.distance_embedding(distances)
        inv_embedding = self.inverse_embedding(inverse_distances)
        inv2_embedding = self.inverse_squared_embedding(inverse_squared)
        dir_embedding = self.direction_embedding(directions)
        
        x = torch.cat([d_embedding, inv_embedding, inv2_embedding, dir_embedding], dim=-1)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.output_layer1(x)
        x = self.output_norm1(x)
        x = F.silu(x)
        
        x = self.output_layer2(x)
        x = self.output_norm2(x)
        x = F.silu(x)
        
        potential = self.output_layer3(x)
        
        scaled_potential = self.scaling_factor * potential * inverse_distances
        
        cutoff_factor = torch.exp(-(distances / self.r_cutoff)**2)
        
        return scaled_potential.squeeze(-1) * cutoff_factor.squeeze(-1)
    
    def compute_force(self, r_vectors):
        """
        Computes vector force as negative gradient of potential
        
        Args:
            r_vectors: Tensor of distance vectors [batch_size, 3]
            
        Returns:
            Tensor of vector forces [batch_size, 3]
        """
        r_vectors_grad = r_vectors.clone().requires_grad_(True)
        
        potential = self.compute_potential(r_vectors_grad)
        
        total_potential = torch.sum(potential)
        
        forces = -torch.autograd.grad(
            total_potential, r_vectors_grad, 
            create_graph=True, retain_graph=True
        )[0]
        
        return forces
    
    def forward(self, r_vectors):
        """
        Forward pass computing forces from distance vectors
        
        Args:
            r_vectors: Tensor of distance vectors [batch_size, 3]
            
        Returns:
            Tensor of vector forces [batch_size, 3]
        """
        return self.compute_force(r_vectors.clone())


class KANPotentialModel(nn.Module):
    """Potential model based on Kolmogorov-Arnold Network (KAN)"""
    def __init__(self, hidden_dim=64, num_layers=3, max_potential=100.0, device='cuda'):
        super().__init__()
        
        if kan is None:
            raise ImportError("pykan is not installed. Please install with: pip install pykan")
            
        self.max_potential = max_potential
        self.hidden_dim = hidden_dim
        self.grid_size = 16
        self.device = device
        
        try:
            width_list = [4] + [hidden_dim] * num_layers + [1]
            self.kan_network = kan.MultKAN(width_list, self.grid_size, device=self.device)
            print(f"KAN успешно инициализирован с размерностями: {width_list} на устройстве {self.device}")
        except Exception as e:
            print(f"Ошибка инициализации KAN: {e}")
            print(f"Пробуем альтернативную инициализацию...")
            
            try:
                self.kan_network = kan.KAN(
                    [4, hidden_dim, hidden_dim, hidden_dim, 1], 
                    grid=self.grid_size,
                    device=self.device
                )
                print(f"Успешно инициализирован KAN на устройстве {self.device}")
            except Exception as e2:
                print(f"Вторая попытка инициализации KAN также не удалась: {e2}")
                print("Используем очень простую инициализацию...")
                
                args = dir(kan.MultKAN.__init__)
                print(f"Доступные аргументы KAN: {args}")
                self.kan_network = kan.MultKAN([4, hidden_dim, 1], 10, device=self.device)
                print(f"Успешно инициализирован KAN (простой) на устройстве {self.device}")
        self.kan_network.speed()
        self.scaling_factor = nn.Parameter(torch.ones(1, device=self.device) * 0.1)
    
    def compute_potential(self, r_vectors):
        """Calculate potential energy using KAN"""
        is_batched = len(r_vectors.shape) > 1
        if not is_batched:
            r_vectors = r_vectors.unsqueeze(0) 
        r_norm = torch.norm(r_vectors, dim=1, keepdim=True)
        
        safe_r_norm = torch.clamp(r_norm, min=1e-6)
        normalized_r = r_vectors / safe_r_norm
        
        r_scaled = torch.clamp(r_norm / 5.0, 0.0, 1.0) 
        kan_input = torch.cat([r_scaled, normalized_r], dim=1) 

        if r_vectors.requires_grad:
            kan_input.requires_grad_(True)
            
        potential_raw = self.kan_network(kan_input)

        potential = torch.tanh(potential_raw) * torch.abs(self.scaling_factor) * self.max_potential
        
        decaying_factor = 1.0 / (1.0 + safe_r_norm)
        scaled_potential = potential * decaying_factor
        
        if not is_batched:
            scaled_potential = scaled_potential.squeeze(0)
            
        return scaled_potential.squeeze(-1)
    
    def compute_force(self, r_vectors):
        """Compute forces from potential using automatic differentiation"""
        r_vectors_grad = r_vectors.clone().requires_grad_(True)
        
        potential = self.compute_potential(r_vectors_grad)
        
        total_potential = torch.sum(potential)
        
        forces = -torch.autograd.grad(
            total_potential, r_vectors_grad, 
            create_graph=True, retain_graph=True
        )[0]
        
        return forces
    
    def forward(self, r_vectors):
        """Forward pass computing forces from distance vectors"""
        return self.compute_force(r_vectors.clone())

    def get_symbolic_formula(self, precision=3, simplify=True):
        """
        Извлекает символическую формулу из обученной KAN сети
        
        Args:
            precision (int): Количество знаков после запятой в коэффициентах
            simplify (bool): Нужно ли упрощать формулу
            
        Returns:
            str: Символическая формула потенциала
        """
        try:
            if hasattr(self.kan_network, 'get_formula'):
                formula = self.kan_network.get_formula(precision=precision, simplify=simplify)
                
                scale_factor = self.scaling_factor.item()
                formula = f"({formula}) * {scale_factor:.{precision}f} * (1.0 / (1.0 + r_norm))"
                
                return formula
            elif hasattr(self.kan_network, 'get_expression'):
                formula = self.kan_network.get_expression(precision=precision, simplify=simplify)
                
                scale_factor = self.scaling_factor.item()
                formula = f"({formula}) * {scale_factor:.{precision}f} * (1.0 / (1.0 + r_norm))"
                
                return formula
            elif hasattr(self.kan_network, 'get_symbolic_expression'):
                formula = self.kan_network.get_symbolic_expression(precision=precision, simplify=simplify)
                
                scale_factor = self.scaling_factor.item()
                formula = f"({formula}) * {scale_factor:.{precision}f} * (1.0 / (1.0 + r_norm))"
                
                return formula
            else:
                return "Функция извлечения формулы не найдена в реализации KAN"
        except Exception as e:
            return f"Ошибка при извлечении формулы: {str(e)}"


class LossManager:
    """Class to handle different loss functions and combinations"""
    def __init__(self, potential_weight=1.0, force_weight=1.0, symmetric_weight=0.5, 
                 shape_weight=0.3, conservation_weight=0.2, true_potential_weight=0.5,
                 use_true_potential_only_for_init=True):
        self.potential_weight = potential_weight
        self.force_weight = force_weight
        self.symmetric_weight = symmetric_weight
        self.shape_weight = shape_weight
        self.conservation_weight = conservation_weight
        self.true_potential_weight = true_potential_weight
        self.true_potential_model = None 
        self.use_true_potential_only_for_init = use_true_potential_only_for_init
        
    def set_true_potential(self, true_potential_model):
        """Устанавливает истинную модель потенциала для использования в обучении"""
        self.true_potential_model = true_potential_model
        
    def enable_initialization_mode(self):
        """Временно разрешает использование истинного потенциала для инициализации.
        Вызовите перед началом предварительного обучения."""
        self._original_use_true_potential_only_for_init = self.use_true_potential_only_for_init
        self._in_initialization_mode = True
        
    def disable_initialization_mode(self):
        """Возвращает настройки к исходному состоянию.
        Вызовите после завершения предварительного обучения."""
        if hasattr(self, '_original_use_true_potential_only_for_init'):
            self.use_true_potential_only_for_init = self._original_use_true_potential_only_for_init
        self._in_initialization_mode = False
        
    def potential_loss(self, pred_potential, true_potential):
        """MSE loss for potential values"""
        return F.mse_loss(pred_potential, true_potential)
    
    def true_potential_loss(self, model, r_vectors, potential_params=None, initialization_only=False):
        """Loss на основе истинного потенциала из MesonExchangePotential
        
        Args:
            model: Модель, для которой вычисляется потеря
            r_vectors: Входные векторы расстояний
            potential_params: Параметры истинного потенциала (опционально)
            initialization_only: Если True, игнорирует флаг use_true_potential_only_for_init и всегда вычисляет потерю
        """
        if not (initialization_only or not self.use_true_potential_only_for_init):
            return torch.tensor(0.0, device=r_vectors.device)
        
        if self.true_potential_model is None and potential_params is None:
            return torch.tensor(0.0, device=r_vectors.device)
            
        r_norms = torch.norm(r_vectors, dim=1)
        device = r_vectors.device
        
        if potential_params is not None:
            r_core = potential_params.get('r_core', 0.3)
            r_cutoff = potential_params.get('r_cutoff', 5.0)
        else:
            r_core = getattr(self.true_potential_model, 'r_core', 0.3)
            r_cutoff = getattr(self.true_potential_model, 'r_cutoff', 5.0)
            
        valid_mask = (r_norms > r_core) & (r_norms < r_cutoff)
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=device)
            
        valid_r = r_vectors[valid_mask]
        pred_potential = model.compute_potential(valid_r)
        
        if potential_params is not None:
            g_att = potential_params['g_att']
            g_rep = potential_params['g_rep']
            m_pi = potential_params['m_pi']
            m_rho = potential_params['m_rho']
            
            r_safe = torch.clamp(r_norms[valid_mask], min=1e-10)
            
            true_potential = -g_att**2 * torch.exp(-m_pi * r_safe) / r_safe + \
                             g_rep**2 * torch.exp(-m_rho * r_safe) / r_safe
        else:
            r_tensor = r_norms[valid_mask].unsqueeze(1)
            true_potential = self.true_potential_model.compute(r_tensor).squeeze()
        
        pred_min = torch.min(pred_potential)
        true_min = torch.min(true_potential)
        pred_potential_norm = pred_potential - pred_min
        true_potential_norm = true_potential - true_min
        
        max_true = torch.max(true_potential_norm)
        max_pred = torch.max(pred_potential_norm)
        if max_pred > 1e-6:
            scale_factor = max_true / max_pred
            pred_potential_scaled = pred_potential_norm * scale_factor
        else:
            pred_potential_scaled = pred_potential_norm
        
        mse_loss = F.mse_loss(pred_potential_scaled, true_potential_norm)
        huber_loss = F.smooth_l1_loss(pred_potential_scaled, true_potential_norm)
        
        if len(valid_r) > 5: 
            valid_r_grad = valid_r.clone().requires_grad_(True)
            
            pred_pot_grad = model.compute_potential(valid_r_grad)
            pred_sum = torch.sum(pred_pot_grad)
            pred_force = -torch.autograd.grad(pred_sum, valid_r_grad, create_graph=True)[0]
            
            if potential_params is not None:
                r_norms_valid = r_norms[valid_mask]
                r_safe = torch.clamp(r_norms_valid, min=1e-10)
                
                term1 = g_rep**2 * torch.exp(-m_rho * r_safe) * (m_rho/r_safe + 1/(r_safe**2))
                term2 = g_att**2 * torch.exp(-m_pi * r_safe) * (m_pi/r_safe + 1/(r_safe**2))
                true_force_magnitudes = term1 - term2
                
                true_force = valid_r / r_norms_valid.unsqueeze(1) * true_force_magnitudes.unsqueeze(1)
            else:
                true_force = self.true_potential_model.compute_force(r_tensor)
            
            weights = 1.0 / (r_norms[valid_mask] + 0.5)
            weights = weights / weights.sum()
            
            force_diff = (pred_force - true_force) ** 2
            force_loss = torch.sum(weights.unsqueeze(1) * force_diff)
            
            combined_loss = 0.5 * mse_loss + 0.3 * huber_loss + 0.2 * force_loss
        else:
            combined_loss = 0.7 * mse_loss + 0.3 * huber_loss
        
        return combined_loss
    
    def force_loss(self, pred_force, true_force):
        """Combined L1 and L2 loss for forces"""
        if pred_force.shape != true_force.shape:
            min_size = min(pred_force.shape[0], true_force.shape[0])
            pred_force = pred_force[:min_size]
            true_force = true_force[:min_size]
        
        pred_force = torch.clamp(pred_force, min=-1e6, max=1e6)
        true_force = torch.clamp(true_force, min=-1e6, max=1e6)
        
        mse_loss = F.smooth_l1_loss(pred_force, true_force, beta=0.1)
        mae_loss = F.l1_loss(pred_force, true_force)
        
        pred_norm = torch.norm(pred_force, dim=-1, keepdim=True)
        true_norm = torch.norm(true_force, dim=-1, keepdim=True)
        
        epsilon = 1e-6
        valid_mask = (pred_norm > epsilon) & (true_norm > epsilon)
        
        direction_loss = torch.tensor(0.0, device=pred_force.device)
        
        if torch.any(valid_mask):
            pred_dir = torch.zeros_like(pred_force)
            true_dir = torch.zeros_like(true_force)
            
            pred_dir[valid_mask.squeeze(-1)] = pred_force[valid_mask.squeeze(-1)] / pred_norm[valid_mask]
            true_dir[valid_mask.squeeze(-1)] = true_force[valid_mask.squeeze(-1)] / true_norm[valid_mask]
            
            cos_sim = F.cosine_similarity(pred_dir, true_dir, dim=-1)
            cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)  
            direction_loss = 1.0 - cos_sim[valid_mask.squeeze(-1)].mean()
        
        total_loss = mse_loss + 0.2 * mae_loss + 0.3 * direction_loss
        
        return total_loss
    
    def symmetry_loss(self, model, r_vectors):
        """Enforce rotational symmetry"""
        if r_vectors.shape[0] == 0:
            return torch.tensor(0.0, device=r_vectors.device)
            
        batch_size = r_vectors.shape[0]
        
        axis = F.normalize(torch.randn(batch_size, 3, device=r_vectors.device), dim=1)
        angle = torch.rand(batch_size, 1, device=r_vectors.device) * 2 * np.pi
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        R = (
            cos_a.unsqueeze(-1) * torch.eye(3, device=r_vectors.device).unsqueeze(0) +
            sin_a.unsqueeze(-1) * torch.cross(axis.unsqueeze(2), torch.eye(3, device=r_vectors.device).unsqueeze(0).repeat(batch_size, 1, 1), dim=1) +
            (1 - cos_a).unsqueeze(-1) * (axis.unsqueeze(2) @ axis.unsqueeze(1))
        )
        
        r_rotated = torch.bmm(r_vectors.unsqueeze(1), R).squeeze(1)
        
        f_original = model.compute_force(r_vectors)
        f_rotated = model.compute_force(r_rotated)
        
        f_rotated_inv = torch.bmm(f_rotated.unsqueeze(1), R.transpose(1, 2)).squeeze(1)
        
        return F.mse_loss(f_original, f_rotated_inv)
    
    def shape_constraints_loss(self, model, r_vectors):
        """Enforce physical shape constraints for nuclear potentials"""
        r_norms = torch.norm(r_vectors, dim=1)
        device = r_vectors.device
        
        total_shape_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_components_count = 0
        
        far_mask = r_norms > 3.0
        if torch.any(far_mask):
            far_vectors = r_vectors[far_mask]
            far_potentials = model.compute_potential(far_vectors)
            far_potentials = torch.clamp(far_potentials, min=-10.0, max=10.0)
            far_loss = torch.mean(far_potentials**2)
            
            if torch.isfinite(far_loss):
                total_shape_loss = total_shape_loss + 0.5 * far_loss
                loss_components_count += 1
        
        close_mask = (r_norms > 0.1) & (r_norms < 0.3)
        if torch.any(close_mask):
            close_vectors = r_vectors[close_mask].clone().requires_grad_(True)
            close_potentials = model.compute_potential(close_vectors)
            close_potentials = torch.clamp(close_potentials, min=-10.0, max=10.0)
            
            grad_sum = torch.sum(close_potentials)
            grads = torch.autograd.grad(grad_sum, close_vectors, create_graph=True)[0]
            
            if grads is not None:
                grads = torch.clamp(grads, min=-100.0, max=100.0)
                
                safe_norms = torch.clamp(r_norms[close_mask].unsqueeze(1), min=1e-6)
                dir_vectors = close_vectors / safe_norms
                
                grad_radial = torch.sum(grads * dir_vectors, dim=1)
                
                repulsive_loss = torch.mean(F.relu(grad_radial))
                
                positive_pot_loss = torch.mean(F.relu(-close_potentials))
                
                if torch.isfinite(repulsive_loss) and torch.isfinite(positive_pot_loss):
                    total_shape_loss = total_shape_loss + 2.0 * repulsive_loss + positive_pot_loss
                    loss_components_count += 1
        
        mid_mask = (r_norms > 0.4) & (r_norms < 0.8)
        if torch.any(mid_mask):
            mid_vectors = r_vectors[mid_mask].clone().requires_grad_(True)
            mid_potentials = model.compute_potential(mid_vectors)
            mid_potentials = torch.clamp(mid_potentials, min=-10.0, max=10.0)
            
            attraction_loss = torch.mean(F.relu(mid_potentials))
            
            grad_sum = torch.sum(mid_potentials)
            mid_grads = torch.autograd.grad(grad_sum, mid_vectors, create_graph=True)[0]
            
            if mid_grads is not None:
                mid_grads = torch.clamp(mid_grads, min=-100.0, max=100.0)
                
                safe_norms = torch.clamp(r_norms[mid_mask].unsqueeze(1), min=1e-6)
                dir_vectors = mid_vectors / safe_norms
                
                grad_radial = torch.sum(mid_grads * dir_vectors, dim=1)
                
                has_positive = torch.any(grad_radial > 0)
                has_negative = torch.any(grad_radial < 0)
                
                min_indicator_loss = torch.tensor(0.0 if has_positive and has_negative else 1.0, device=device)
                
                if torch.isfinite(attraction_loss) and torch.isfinite(min_indicator_loss):
                    total_shape_loss = total_shape_loss + 0.5 * attraction_loss + 0.5 * min_indicator_loss
                    loss_components_count += 1
        
        decay_mask = r_norms > 1.0
        if torch.any(decay_mask):
            sorted_indices = torch.argsort(r_norms[decay_mask])
            decay_vectors = r_vectors[decay_mask][sorted_indices]
            decay_potentials = model.compute_potential(decay_vectors)
            decay_potentials = torch.clamp(decay_potentials, min=-10.0, max=10.0)
            
            if len(decay_potentials) > 1:
                diff = decay_potentials[1:] - decay_potentials[:-1]
                monotonic_decay_loss = torch.mean(F.relu(-diff))
                
                smoothness_loss = torch.mean(torch.abs(diff))
                
                if torch.isfinite(monotonic_decay_loss) and torch.isfinite(smoothness_loss):
                    total_shape_loss = total_shape_loss + 0.5 * monotonic_decay_loss + 0.2 * smoothness_loss
                    loss_components_count += 1
        
        if loss_components_count == 0:
            return torch.tensor(0.1, device=device, requires_grad=True)
            
        total_shape_loss = total_shape_loss / max(1, loss_components_count)
        
        if not torch.isfinite(total_shape_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)
            
        return total_shape_loss
    
    def conservation_loss(self, model, positions, velocities, masses=None, dt=0.001):
        """Enforce energy conservation in dynamics"""
        if masses is None:
            masses = torch.ones(positions.shape[0], device=positions.device)
            
        batch_size = positions.shape[0]
        n_particles = positions.shape[1]
        
        if batch_size == 0 or n_particles == 0:
            return torch.tensor(0.0, device=positions.device)
        
        rel_positions = positions.reshape(batch_size, n_particles, 1, 3) - positions.reshape(batch_size, 1, n_particles, 3)
        
        eye_mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        batch_mask = ~eye_mask.unsqueeze(0).expand(batch_size, n_particles, n_particles)
        
        kinetic_energy = 0.5 * torch.sum(masses.reshape(-1, 1) * torch.sum(velocities**2, dim=2), dim=1)
        
        new_positions = positions + velocities * dt
        
        rel_pos_flat_initial = rel_positions[batch_mask].reshape(-1, 3)
        
        new_rel_positions = new_positions.reshape(batch_size, n_particles, 1, 3) - new_positions.reshape(batch_size, 1, n_particles, 3)
        rel_pos_flat_final = new_rel_positions[batch_mask].reshape(-1, 3)
        
        if rel_pos_flat_initial.shape[0] == 0:
            return torch.tensor(0.0, device=positions.device)
            
        potential_initial = model.compute_potential(rel_pos_flat_initial)
        potential_final = model.compute_potential(rel_pos_flat_final)
        
        if not (torch.isfinite(potential_initial).all() and torch.isfinite(potential_final).all()):
            return torch.tensor(0.0, device=positions.device)
        
        n_pairs = n_particles * (n_particles - 1)
        potential_initial = potential_initial.reshape(batch_size, n_pairs)
        potential_final = potential_final.reshape(batch_size, n_pairs)
        
        total_potential_initial = torch.sum(potential_initial, dim=1) * 0.5  
        total_potential_final = torch.sum(potential_final, dim=1) * 0.5
        
        total_energy_initial = kinetic_energy + total_potential_initial
        total_energy_final = kinetic_energy + total_potential_final 
        
        energy_diff = torch.abs(total_energy_final - total_energy_initial)
        conservation_loss = torch.mean(energy_diff)
        
        # Проверка на конечность результата
        if torch.isfinite(conservation_loss):
            return conservation_loss
        else:
            return torch.tensor(0.0, device=positions.device)
    
    def nuclear_yukawa_form_loss(self, model, r_vectors):
        """Enforce a Yukawa-like form for the potential to match nuclear physics"""
        r_norms = torch.norm(r_vectors, dim=1)
        device = r_vectors.device
        
        valid_mask = (r_norms > 0.3) & (r_norms < 4.0)
        
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=device)
        
        valid_r = r_vectors[valid_mask]
        valid_norms = r_norms[valid_mask]
        
        potentials = model.compute_potential(valid_r)
        
        if not torch.isfinite(potentials).all():
            return torch.tensor(0.0, device=device)
        
        r_times_v = valid_norms * torch.abs(potentials) + 1e-8
        log_r_times_v = torch.log(r_times_v)
        
        total_loss = torch.tensor(0.0, device=device)
        components_count = 0
        
        small_r_mask = valid_norms < 1.0
        if torch.any(small_r_mask):
            small_r = valid_norms[small_r_mask]
            small_log_rv = log_r_times_v[small_r_mask]
            
            if len(small_r) > 1:
                expected_slope = -3.9  
                expected_values = small_log_rv[0] + expected_slope * (small_r - small_r[0])
                small_r_loss = F.mse_loss(small_log_rv, expected_values)
                
                if torch.isfinite(small_r_loss):
                    total_loss = total_loss + 0.5 * small_r_loss
                    components_count += 1
            
        mid_r_mask = (valid_norms >= 1.0) & (valid_norms <= 3.0)
        if torch.any(mid_r_mask):
            mid_r = valid_norms[mid_r_mask]
            mid_log_rv = log_r_times_v[mid_r_mask]
            
            if len(mid_r) > 1:
                expected_slope = -0.7 
                expected_values = mid_log_rv[0] + expected_slope * (mid_r - mid_r[0])
                mid_r_loss = F.mse_loss(mid_log_rv, expected_values)
                
                if torch.isfinite(mid_r_loss):
                    total_loss = total_loss + 0.5 * mid_r_loss
                    components_count += 1
        
        if components_count == 0:
            return torch.tensor(0.0, device=device)
            
        return total_loss / components_count
    
    def total_loss(self, model, r_vectors, true_force, positions=None, velocities=None, masses=None, epoch=0, potential_params=None):
        """Combined loss function with physical priors"""
        r_vectors_size = r_vectors.shape[0] if r_vectors.shape[0] > 0 else 0
        
        loss_components = {
            'force': 0.0,
            'potential': 0.0,
            'symmetry': 0.0,
            'shape': 0.0,
            'conservation': 0.0,
            'yukawa': 0.0,
            'total': 0.0
        }
        
        if r_vectors_size == 0:
            return torch.tensor(0.0, device=true_force.device), loss_components
        
        pred_force = model.compute_force(r_vectors)
        
        force_l = self.force_loss(pred_force, true_force)
        loss_components['force'] = force_l.item() if isinstance(force_l, torch.Tensor) else 0.0
        
        potential_l = torch.tensor(0.0, device=r_vectors.device)
        if not self.use_true_potential_only_for_init and self.true_potential_weight > 0 and (self.true_potential_model is not None or potential_params is not None):
            potential_l = self.true_potential_loss(model, r_vectors, potential_params, initialization_only=False)
            loss_components['potential'] = potential_l.item() if isinstance(potential_l, torch.Tensor) else 0.0
        
        symmetry_l = torch.tensor(0.0, device=r_vectors.device)
        if self.symmetric_weight > 0:
            symmetry_l = self.symmetry_loss(model, r_vectors)
            loss_components['symmetry'] = symmetry_l.item() if isinstance(symmetry_l, torch.Tensor) else 0.0
        
        shape_l = torch.tensor(0.0, device=r_vectors.device)
        if self.shape_weight > 0:
            shape_l = self.shape_constraints_loss(model, r_vectors)
            loss_components['shape'] = shape_l.item() if isinstance(shape_l, torch.Tensor) else 0.0
        
        conservation_l = torch.tensor(0.0, device=r_vectors.device)
        valid_conservation = (positions is not None and velocities is not None and 
                             positions.shape[0] > 0 and velocities.shape[0] > 0 and
                             positions.shape[0] == velocities.shape[0])
        
        if self.conservation_weight > 0 and valid_conservation:
            conservation_l = self.conservation_loss(model, positions, velocities, masses)
            loss_components['conservation'] = conservation_l.item() if isinstance(conservation_l, torch.Tensor) else 0.0
        
        yukawa_l = torch.tensor(0.0, device=r_vectors.device)
        yukawa_l = self.nuclear_yukawa_form_loss(model, r_vectors)
        loss_components['yukawa'] = yukawa_l.item() if isinstance(yukawa_l, torch.Tensor) else 0.0
        
        shape_weight_dynamic = self.shape_weight * min(1.0, epoch / 10.0)
        conservation_weight_dynamic = self.conservation_weight * min(1.0, epoch / 15.0)
        yukawa_weight = 0.2 * min(1.0, epoch / 5.0)
        
        true_potential_weight_dynamic = 0.0
        if not self.use_true_potential_only_for_init:
            true_potential_weight_dynamic = self.true_potential_weight * min(1.0, epoch / 3.0)
        
        total = (
            self.force_weight * force_l + 
            true_potential_weight_dynamic * potential_l +
            self.symmetric_weight * symmetry_l +
            shape_weight_dynamic * shape_l +
            conservation_weight_dynamic * conservation_l +
            yukawa_weight * yukawa_l
        )
        
        if not torch.isfinite(total):
            total = force_l
            if not torch.isfinite(total):
                total = torch.tensor(1.0, device=r_vectors.device, requires_grad=True)
        
        loss_components['total'] = total.item()
        
        return total, loss_components