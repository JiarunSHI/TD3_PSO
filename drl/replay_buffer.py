import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=200000, rng=None):
        self.max_size = max_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.s = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_ = np.zeros((max_size, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_batch(self, s, a, r, s_):
        idx = (self.ptr + np.arange(len(s))) % self.max_size
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx, 0] = r
        self.s_[idx] = s_
        self.ptr = (self.ptr + len(s)) % self.max_size
        self.size = min(self.size + len(s), self.max_size)

    def sample(self, batch_size, device):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.s[idx], device=device),
            torch.as_tensor(self.a[idx], device=device),
            torch.as_tensor(self.r[idx], device=device),
            torch.as_tensor(self.s_[idx], device=device),
        )
