import torch
import torch.nn as nn

class PotentialNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, relative_pos):
        relative_pos = relative_pos.float()
        force = self.network(relative_pos)
        return force