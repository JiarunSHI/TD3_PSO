import os
import random
import torch
from env.fixed_env import MapEnv
#from env.fixed_env import MapEnv
from pso.particle import Particle
from drl.td3 import TD3
from drl.replay_buffer import ReplayBuffer
from trainer.train import train
from utils.visualization import plot_map
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 固定经验回放的采样顺序；NumPy 会在 MapEnv.generate_mixed()
    # 中恢复为 seed=42，从而固定地图、初始种群和 PSO 随机序列。
    random.seed(42)

    env = MapEnv()

    obstacles = env.generate_mixed()

    swarm = [Particle(NUM_POINTS, obstacles) for _ in range(NUM_PARTICLES)]

    td3 = TD3(device)

    # TD3 创建时保留随机权重；创建完成后固定 PyTorch 随机序列，
    # 避免训练阶段的 torch 随机噪声成为额外实验变量。
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)





    buffer = ReplayBuffer()




    best_path, history, collector = train(td3, buffer, swarm, visualize=True)

    plot_map(obstacles, best_path, START, GOAL)


if __name__ == "__main__":
    main()
