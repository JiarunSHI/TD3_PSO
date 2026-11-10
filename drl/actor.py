import torch.nn as nn

class Actor(nn.Module):
    """输出 [-1, 1] 的原始动作，换算成 PSO 参数在外部完成（trainer.train.scale_action）。"""

    def __init__(self, state_dim, action_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

    def pre_tanh(self, x):
        return self.net[:-1](x)
