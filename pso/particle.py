import numpy as np
from config import START, GOAL, MAP_MIN, MAP_MAX
from env.fitness import fitness


class Particle:
    def __init__(self, m, obstacles):
        # 与批量初始化共用拒绝采样规则；保留旧入口的全局 RNG。
        self.pos = init_population(1, m, np.random)[0]

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


def init_population(n, m, rng=None, start=START, goal=GOAL,
                    map_min=MAP_MIN, map_max=MAP_MAX, max_attempts=None):
    """
    直线 + 2.5 倍尺度归一化的累计高斯扰动；整条候选路径越界则丢弃。
    收集到 n 条后才返回，不裁剪、不复制补齐，也不排除碰撞路径。
    :param rng: 默认使用系统熵创建新 Generator，不传固定种子；可注入 RNG 用于测试。
    :param max_attempts: 防止异常配置无限循环；耗尽时明确报错，不返回不足 n 条的种群。
    """
    rng = np.random.default_rng() if rng is None else rng
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if n <= 0 or m < 2:
        raise ValueError("n must be positive and m must be at least 2")
    if not np.isfinite([map_min, map_max]).all() or map_min >= map_max:
        raise ValueError("map_min must be finite and smaller than map_max")
    for endpoint in (start, goal):
        if endpoint.shape != (2,) or not np.isfinite(endpoint).all():
            raise ValueError("start and goal must be finite 2D points")
        if np.any(endpoint < map_min) or np.any(endpoint > map_max):
            raise ValueError("start and goal must be inside the map bounds")
    max_attempts = max(10000, 1000 * n) if max_attempts is None else max_attempts
    base = np.linspace(start, goal, m)
    pos = np.empty((n, m, 2), dtype=np.float64)
    accepted = 0
    for _ in range(max_attempts):
        candidate = base.copy()
        if m > 2:
            noise = np.cumsum(rng.standard_normal((m - 2, 2)), axis=0)
            noise /= np.std(noise) + 1e-8
            candidate[1:-1] += 2.5 * noise
        if (np.isfinite(candidate).all()
                and np.all(candidate >= map_min) and np.all(candidate <= map_max)):
            pos[accepted] = candidate
            accepted += 1
            if accepted == n:
                return pos
    raise RuntimeError(f"Accepted only {accepted}/{n} paths after {max_attempts} attempts; "
                       "check map bounds, endpoints and initialization scale")
