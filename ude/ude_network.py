import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def __init__(self, hidden_dim=64, num_blocks=2, input_dim=3, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_layer = nn.Linear(input_dim+1, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer2 = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for better stability"""
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_layer.bias)
        
        nn.init.kaiming_normal_(self.output_layer1.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer1.bias)
        
        nn.init.normal_(self.output_layer2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer2.bias)
        
    def forward(self, x):

        batch_size = x.shape[0]
        
        dist = torch.norm(x, dim=-1, keepdim=True)
        
        safe_dist = torch.clamp(dist, min=1e-6)
        
        x_scaled = x / (safe_dist + 1.0)
        
        x_with_dist = torch.cat([x_scaled, torch.log1p(safe_dist)], dim=-1)
        
        h = self.input_layer(x_with_dist)
        h = self.input_norm(h)
        h = F.silu(h)
        
        for block in self.res_blocks:
            h = block(h)
        
        h = self.output_layer1(h)
        h = self.output_norm(h)
        h = F.silu(h)
        force = self.output_layer2(h)
        
        direction = x / safe_dist
        force_magnitude = torch.sum(force * direction, dim=-1, keepdim=True)
        
        force_magnitude = torch.tanh(force_magnitude) * 5.0
        force = force_magnitude * direction
        
        return force