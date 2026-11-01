import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import START, GOAL
from env.fitness import fitness
from env.fixed_env import MapEnv


# =========================
# PSO参数
# =========================
NUM_PARTICLES = 150
NUM_POINTS = 10
MAX_ITERS = 250

W = 0.7
C1 = 1.5
C2 = 1.5

# 在 seed=42 生成的固定初始种群上叠加极小扰动。
# 独立 RNG 不会改变后续 PSO 使用的 NumPy 全局随机序列。
INIT_JITTER_STD = 0.01


# =========================
# 粒子定义（已修改初始化）
# =========================
class Particle:
    def __init__(self, obstacles, jitter_rng=None):

        # =====================================================
        # 1. base：起点到终点直线
        # =====================================================
        base = np.linspace(START, GOAL, NUM_POINTS)

        # =====================================================
        # 2. 高斯累积扰动（核心替换部分）
        # =====================================================
        noise = np.zeros((NUM_POINTS, 2))

        if NUM_POINTS > 2:
            # 中间点随机增量
            increments = np.random.randn(NUM_POINTS - 2, 2)

            # 累积生成“轨迹型扰动”
            noise[1:-1] = np.cumsum(increments, axis=0)




            std = np.std(noise[1:-1]) + 1e-8
            noise[1:-1] = noise[1:-1] / std

        # =====================================================
        # 3. 最终路径
        # =====================================================
        self.pos = base + 2.5 * noise

        # 只扰动中间路径点，保留固定种群的整体形状。
        if jitter_rng is not None and INIT_JITTER_STD > 0:
            self.pos[1:-1] += jitter_rng.normal(
                loc=0.0,
                scale=INIT_JITTER_STD,
                size=(NUM_POINTS - 2, 2),
            )

        self.pos[0] = START
        self.pos[-1] = GOAL

        # =====================================================

        # =====================================================
        self.vel = np.zeros_like(self.pos)
        self.obstacles = obstacles

        self.f = fitness(self.pos, obstacles)

        self.pbest = self.pos.copy()
        self.pbest_f = self.f

def visualize_path(path, obstacles, title="Best Path"):
    plt.figure(figsize=(8, 6))
    for ox, oy, radius in obstacles:
        circle = plt.Circle((ox, oy), radius, color='red', alpha=0.5)
        plt.gca().add_patch(circle)

    if path is not None and len(path) > 0:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, marker='o', markersize=4, label='Path')
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
        plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='Goal')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
# =========================
# 环境初始化
# =========================
env = MapEnv()
obstacles = env.generate_mixed()


# =========================
# 初始化粒子群
# =========================
jitter_rng = np.random.default_rng()
swarm = [Particle(obstacles, jitter_rng) for _ in range(NUM_PARTICLES)]

gbest = min(swarm, key=lambda p: p.f)
gbest_pos = gbest.pos.copy()
gbest_f = gbest.f


# =========================
# PSO 主循环
# =========================
for it in range(MAX_ITERS):

    for p in swarm:

        r1 = np.random.rand(NUM_POINTS, 2)
        r2 = np.random.rand(NUM_POINTS, 2)

        # velocity update
        p.vel = (
            W * p.vel
            + C1 * r1 * (p.pbest - p.pos)
            + C2 * r2 * (gbest_pos - p.pos)
        )

        # position update
        p.pos = p.pos + p.vel

        # 固定起点终点
        p.pos[0] = START
        p.pos[-1] = GOAL

        # fitness
        p.f = fitness(p.pos, obstacles)

        # pbest
        if p.f < p.pbest_f:
            p.pbest = p.pos.copy()
            p.pbest_f = p.f

    # gbest
    best_particle = min(swarm, key=lambda p: p.f)

    if best_particle.f < gbest_f:
        gbest_f = best_particle.f
        gbest_pos = best_particle.pos.copy()

    print(f"Iteration {it:03d} | Best Fitness = {gbest_f:.6f}")


# =========================
# 最终结果
# =========================
print("\n========== FINAL RESULT ==========")
print("Best Fitness:", gbest_f)
print("Best Path:\n", gbest_pos)

visualize_path(gbest_pos,obstacles)
# =========================
# 可视化
# =========================
