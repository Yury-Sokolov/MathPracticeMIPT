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
        
        self.input_layer = nn.Linear(1, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
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
        """Инициализация весов для стабильного обучения"""
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_layer.bias)
        
        nn.init.kaiming_normal_(self.output_layer1.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer1.bias)
        
        nn.init.kaiming_normal_(self.output_layer2.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer2.bias)
        
        nn.init.normal_(self.output_layer3.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_layer3.bias, 0.1)  

    def compute_potential(self, r_vectors):
        """
        Вычисляет скалярный потенциал для данных векторов расстояний
        
        Args:
            r_vectors: Тензор векторов расстояний [batch_size, 3]
            
        Returns:
            Тензор скалярных потенциалов [batch_size]
        """
        distances = torch.norm(r_vectors, dim=-1, keepdim=True)
        
        distances = torch.clamp(distances, min=1e-6)
        
        x = self.input_layer(distances)
        x = self.input_norm(x)
        x = F.silu(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.output_layer1(x)
        x = self.output_norm1(x)
        x = F.silu(x)
        
        x = self.output_layer2(x)
        x = self.output_norm2(x)
        x = F.silu(x)
        

        potential = self.output_layer3(x)
        
        potential_scaled = torch.tanh(potential / self.max_potential) * self.max_potential
        
        attenuation = torch.exp(-distances)
        final_potential = potential_scaled * attenuation
        
        return final_potential.squeeze(-1)
    
    def compute_force(self, r_vectors):
        """
        Вычисляет векторную силу как отрицательный градиент потенциала
        
        Args:
            r_vectors: Тензор векторов расстояний [batch_size, 3]
            
        Returns:
            Тензор векторных сил [batch_size, 3]
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
        Прямой проход, вычисляющий силы по векторам расстояний.
        Силы вычисляются как отрицательный градиент потенциала.
        
        Args:
            r_vectors: Тензор векторов расстояний [batch_size, 3]
            
        Returns:
            Тензор векторных сил [batch_size, 3]
        """
        return self.compute_force(r_vectors)