import torch
import torch.nn as nn

class SimpleRegNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(SimpleRegNet, self).__init__()
        
        # A simple feedforward network with one hidden layer
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
def load_simple_reg_net(version):
    if version == "base":
        hidden_dim = 64
    else:
        hidden_dim = int(version)
    return SimpleRegNet(hidden_dim=hidden_dim)