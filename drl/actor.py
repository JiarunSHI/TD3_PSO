import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)

        w = 0.5*(x[:,0]+1)
        c1 = x[:,1]+1
        c2 = x[:,2]+1

        return torch.stack([w,c1,c2],dim=1)