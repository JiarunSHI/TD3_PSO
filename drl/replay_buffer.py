import random
import torch

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer=[]
        self.max_size=max_size

    def add(self,s,a,r,s_):
        if len(self.buffer)>=self.max_size:
            self.buffer.pop(0)
        self.buffer.append((s,a,r,s_))

    def sample(self,batch_size,device):
        batch=random.sample(self.buffer,batch_size)
        s,a,r,s_=zip(*batch)

        return (
            torch.FloatTensor(s).to(device),
            torch.FloatTensor(a).to(device),
            torch.FloatTensor(r).unsqueeze(1).to(device),
            torch.FloatTensor(s_).to(device),
        )