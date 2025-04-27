import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    def __init__(self, hidden_dim=128, num_blocks=3, input_dim=3, output_dim=3, max_force=1000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_force = max_force
        
        self.input_layer = nn.Linear(input_dim+4, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm1 = nn.LayerNorm(hidden_dim)
        self.output_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.output_layer3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self._init_weights()
        
    def _init_weights(self):

        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_layer.bias)
        
        nn.init.kaiming_normal_(self.output_layer1.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer1.bias)
        
        nn.init.kaiming_normal_(self.output_layer2.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer2.bias)
        
        nn.init.normal_(self.output_layer3.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer3.bias, -0.1)
        
    def forward(self, x):
        """
        Прямой проход сети, возвращающей вектор силы для заданной разности положений
        
        Args:
            x: Тензор разности положений [batch_size, 3]
            
        Returns:
            torch.Tensor: Сила [batch_size, 3]
        """
        batch_size = x.shape[0]
        
        dist = torch.norm(x, dim=-1, keepdim=True)
        safe_dist = torch.clamp(dist, min=1e-6)
        
        direction = x / safe_dist
        
        inverse_dist = 1.0 / (safe_dist + 0.1)
        exp_short = torch.exp(-5.0 * safe_dist)
        exp_long = torch.exp(-0.5 * safe_dist)
        
        x_features = torch.cat([
            direction,
            safe_dist,
            inverse_dist,
            exp_short,
            exp_long
        ], dim=-1)
        
        h = self.input_layer(x_features)
        h = self.input_norm(h)
        h = F.silu(h)
        
        for block in self.res_blocks:
            h = block(h)
        
        h = self.output_layer1(h)
        h = self.output_norm1(h)
        h = F.silu(h)
        
        h = self.output_layer2(h)
        h = self.output_norm2(h)
        h = F.silu(h)
        
        force_vector = self.output_layer3(h)
        
        force_magnitude = torch.sum(force_vector * direction, dim=-1, keepdim=True)
        
        scaling_factor = 2.0 * self.max_force / math.pi
        force_magnitude = scaling_factor * torch.atan(force_magnitude / scaling_factor)
        
        radial_force = force_magnitude * direction
        
        return radial_force