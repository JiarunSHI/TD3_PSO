import numpy as np
from config import START, GOAL
from env.fitness import fitness


class Particle:
    def __init__(self, m, obstacles):
        # ===== 1. 基准直线 =====
        base = np.linspace(START, GOAL, m)

        # ===== 2. 高斯增量扰动（仅中间点）=====
        noise = np.zeros((m, 2))

        if m > 2:
            increments = np.random.randn(m - 2, 2)
            noise[1:-1] = np.cumsum(increments, axis=0)

            # 标准化，控制幅度
            noise_std = np.std(noise[1:-1]) + 1e-8
            noise[1:-1] = noise[1:-1] / noise_std

        # ===== 3. 最终路径 =====
        self.pos = base + 2.5 * noise

        # 固定起点终点
        self.pos[0] = START
        self.pos[-1] = GOAL

        # ===== 4. PSO 状态 =====
        self.vel = np.zeros_like(self.pos)
        self.obstacles = obstacles

        # ===== 5. 适应度 =====
        self.f = fitness(self.pos, obstacles)
        self.prev_f = self.f

        # ===== 6. 个体最优 =====
        self.pbest = self.pos.copy()
        self.pbest_f = self.f

        # ===== 7. 停滞计数 =====
        self.stagnation = 0