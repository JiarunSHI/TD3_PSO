import numpy as np
from copy import deepcopy

class TraditionalPSO:

    def __init__(self, fitness_func, dim, swarm_size, bounds,
                 w=0.7, c1=1.5, c2=1.5, max_vel=1.0):

        self.fitness_func = fitness_func
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_vel = max_vel

        # 处理边界
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
            # 统一边界
            self.low = np.full(dim, bounds[0])
            self.high = np.full(dim, bounds[1])
        else:
            # 每个维度不同边界
            self.low = np.array([b[0] for b in bounds])
            self.high = np.array([b[1] for b in bounds])

        # 初始化种群
        self.positions = np.random.uniform(self.low, self.high, (swarm_size, dim))
        self.velocities = np.random.uniform(-self.max_vel, self.max_vel, (swarm_size, dim))

        # 个体最优
        self.pbest_pos = deepcopy(self.positions)
        self.pbest_fit = np.full(swarm_size, np.inf)
        # 全局最优
        self.gbest_pos = None
        self.gbest_fit = np.inf

        # 记录历史最优
        self.history = []

    def evaluate(self):
        """评估所有粒子适应度并更新最优"""
        for i in range(self.swarm_size):
            fit = self.fitness_func(self.positions[i])
            if fit < self.pbest_fit[i]:
                self.pbest_fit[i] = fit
                self.pbest_pos[i] = self.positions[i].copy()
            if fit < self.gbest_fit:
                self.gbest_fit = fit
                self.gbest_pos = self.positions[i].copy()
        self.history.append(self.gbest_fit)

    def update(self):
        """更新粒子速度和位置"""
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            # 速度更新
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.pbest_pos[i] - self.positions[i]) +
                                  self.c2 * r2 * (self.gbest_pos - self.positions[i]))
            # 速度限制
            self.velocities[i] = np.clip(self.velocities[i], -self.max_vel, self.max_vel)
            # 位置更新
            self.positions[i] += self.velocities[i]
            # 边界处理（超出边界则反弹并设置速度反向）
            for d in range(self.dim):
                if self.positions[i, d] < self.low[d]:
                    self.positions[i, d] = self.low[d]
                    self.velocities[i, d] = -self.velocities[i, d]
                elif self.positions[i, d] > self.high[d]:
                    self.positions[i, d] = self.high[d]
                    self.velocities[i, d] = -self.velocities[i, d]

    def run(self, max_iter, verbose=True):
        """运行PSO主循环"""
        self.evaluate()  # 初始评估
        for it in range(max_iter):
            self.update()
            self.evaluate()
            if verbose and (it+1) % 10 == 0:
                print(f"Iter {it+1:4d} | Best Fitness: {self.gbest_fit:.6f}")
        return self.gbest_pos, self.gbest_fit, self.history