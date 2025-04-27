import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        identity = x
        out = self.lin1(x)
        out = self.norm1(out)
        out = F.silu(out)
        out = self.lin2(out)
        out = self.norm2(out)
        out = F.silu(out + identity)
        return out

class PotentialNN(nn.Module):

    def __init__(self, hidden_dim=128, num_blocks=3, input_dim=3, max_potential=100.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.max_potential = max_potential
        
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU()
        )
        
        self.inverse_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm1 = nn.LayerNorm(hidden_dim)
        self.output_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.output_layer3 = nn.Linear(hidden_dim // 2, 1)  
        
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
        
        d_embedding = self.distance_embedding(distances)
        inv_embedding = self.inverse_embedding(inverse_distances)
        
        x = torch.cat([d_embedding, inv_embedding], dim=-1)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.output_layer1(x)
        x = self.output_norm1(x)
        x = F.silu(x)
        
        x = self.output_layer2(x)
        x = self.output_norm2(x)
        x = F.silu(x)
        
        potential = self.output_layer3(x)
        
        scaled_potential = potential * inverse_distances
        
        r_cutoff = 5.0
        cutoff_factor = torch.exp(-distances / r_cutoff)
        
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