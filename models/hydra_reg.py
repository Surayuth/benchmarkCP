import torch.nn as nn

class HydraReg(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3):
        """
        There are 3 outputs corresponding to: qlow, q0.5, and qhigh.
        """
        super(HydraReg, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
def load_hydra_reg_net(version):
    if version == "base":
        hidden_dim = 64
    else:
        hidden_dim = int(version)
    return HydraReg(hidden_dim=hidden_dim)