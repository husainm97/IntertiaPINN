import torch
import torch.nn as nn

class GridInertiaPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        out = self.net(x)
        inertia = out[:,0]
        damping = out[:,1]
        freq_dev = out[:,2]
        return inertia, damping, freq_dev