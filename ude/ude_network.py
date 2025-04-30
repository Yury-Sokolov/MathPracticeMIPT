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
    def __init__(self, hidden_dim=256, num_blocks=4, input_dim=3, max_potential=100.0, dropout_rate=0.1):
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
        return self.compute_force(r_vectors)


class KANPotentialModel(nn.Module):
    """Potential model based on Kolmogorov-Arnold Network (KAN)"""
    def __init__(self, hidden_dim=64, num_layers=3, max_potential=100.0):
        super().__init__()
        
        if kan is None:
            raise ImportError("pykan is not installed. Please install with: pip install pykan")
            
        self.max_potential = max_potential
        self.hidden_dim = hidden_dim
        self.grid_size = 16
        
        try:
            width_list = [3] + [hidden_dim] * num_layers + [1]
            
            self.kan_network = kan.MultKAN(
                width_list=width_list,
                grid=self.grid_size,
                name="potential_kan",
                activation="softsign",
                init_sparsity=0.5
            )
        except TypeError as e:
            print(f"MultKAN инициализация не удалась, пробуем альтернативную сигнатуру: {e}")
            self.kan_network = kan.MultKAN(
                width_list, 
                grid=self.grid_size,
                name="potential_kan",
                activation="softsign"
            )
        
        self.scaling_factor = nn.Parameter(torch.ones(1) * 0.1)
    
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
        
        if kan_input.shape[0] == 1:
            potential_raw = self.kan_network(kan_input)
        else:

            potential_parts = []
            batch_size = 4 
            
            for i in range(0, kan_input.shape[0], batch_size):
                batch_input = kan_input[i:i+batch_size]
                batch_output = self.kan_network(batch_input)
                potential_parts.append(batch_output)
            
            if potential_parts:
                potential_raw = torch.cat(potential_parts, dim=0)
            else:
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
        return self.compute_force(r_vectors)


class LossManager:
    """Class to handle different loss functions and combinations"""
    def __init__(self, potential_weight=1.0, force_weight=1.0, 
                 smoothness_weight=0.1, symmetric_weight=0.5):
        self.potential_weight = potential_weight
        self.force_weight = force_weight
        self.smoothness_weight = smoothness_weight
        self.symmetric_weight = symmetric_weight
        
    def potential_loss(self, pred_potential, true_potential):
        """MSE loss for potential values"""
        return F.mse_loss(pred_potential, true_potential)
    
    def force_loss(self, pred_force, true_force):
        """Combined L1 and L2 loss for forces"""
        mse_loss = F.mse_loss(pred_force, true_force)
        mae_loss = F.l1_loss(pred_force, true_force)
        pred_norm = torch.norm(pred_force, dim=-1, keepdim=True) + 1e-8
        true_norm = torch.norm(true_force, dim=-1, keepdim=True) + 1e-8
        pred_dir = pred_force / pred_norm
        true_dir = true_force / true_norm
        direction_loss = 1.0 - F.cosine_similarity(pred_dir, true_dir, dim=-1).mean()
        
        return mse_loss + 0.2 * mae_loss + 0.3 * direction_loss
    
    def smoothness_loss(self, model, r_vectors):
        """Regularization to ensure smooth potentials"""
        r_vectors_grad = r_vectors.clone().requires_grad_(True)
        forces = model.compute_force(r_vectors_grad)
        
        divergence = 0
        for i in range(3):
            dfi_dri = torch.autograd.grad(
                forces[:, i].sum(), r_vectors_grad, create_graph=True
            )[0][:, i]
            divergence += dfi_dri
        
        return torch.mean(divergence**2)
    
    def symmetry_loss(self, model, r_vectors):
        """Enforce rotational symmetry"""
        batch_size = r_vectors.shape[0]
        
        theta = torch.rand(batch_size, 1, device=r_vectors.device) * 2 * np.pi
        phi = torch.rand(batch_size, 1, device=r_vectors.device) * np.pi
        
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        zeros = torch.zeros_like(cos_t)
        ones = torch.ones_like(cos_t)
        
        R = torch.stack([
            torch.cat([cos_t, -sin_t, zeros], dim=1),
            torch.cat([sin_t, cos_t, zeros], dim=1),
            torch.cat([zeros, zeros, ones], dim=1)
        ], dim=1)
        
        r_rotated = torch.bmm(r_vectors.unsqueeze(1), R).squeeze(1)
        
        f_original = model.compute_force(r_vectors)
        f_rotated = model.compute_force(r_rotated)
        f_rotated_inv = torch.bmm(f_rotated.unsqueeze(1), R.transpose(1, 2)).squeeze(1)
        
        return F.mse_loss(f_original, f_rotated_inv)
    
    def total_loss(self, model, r_vectors, true_force, true_potential=None):
        """Combined loss function"""
        pred_force = model.compute_force(r_vectors)
        force_l = self.force_loss(pred_force, true_force)
        
        potential_l = 0
        if true_potential is not None and self.potential_weight > 0:
            pred_potential = model.compute_potential(r_vectors)
            potential_l = self.potential_loss(pred_potential, true_potential)
        
        smoothness_l = 0
        if self.smoothness_weight > 0:
            smoothness_l = self.smoothness_loss(model, r_vectors)
            
        symmetry_l = 0
        if self.symmetric_weight > 0:
            symmetry_l = self.symmetry_loss(model, r_vectors)
        
        total = (
            self.force_weight * force_l + 
            self.potential_weight * potential_l +
            self.smoothness_weight * smoothness_l +
            self.symmetric_weight * symmetry_l
        )
        
        loss_components = {
            'force': force_l.item(),
            'potential': potential_l.item() if isinstance(potential_l, torch.Tensor) else 0,
            'smoothness': smoothness_l.item() if isinstance(smoothness_l, torch.Tensor) else 0,
            'symmetry': symmetry_l.item() if isinstance(symmetry_l, torch.Tensor) else 0,
            'total': total.item()
        }
        
        return total, loss_components