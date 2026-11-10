import numpy as np
from config import START, GOAL

class MapEnv:
    def __init__(self, seed=42):
        self.seed = seed

    def generate_mixed(self, rng=None, endpoints=None):
        # 不传 rng 时用 seed 重置 NumPy 全局随机序列。
        # 传入 np.random.RandomState(seed) 时生成同一张地图，但不影响全局随机序列。
        if rng is None:
            np.random.seed(self.seed)
            rng = np.random

        # =========================
        # 1. 边界障碍
        # =========================
        boundary_obs = [
            (0, 10, 0.85), (20, 10, 0.85),
            (10, 0, 0.85), (10, 20, 0.85)
        ]

        obstacles = []

        # =========================
        # 2. 参数（轻微降低密度）
        # =========================
        num_center = 32   # ↓ 原来40 → 32（中心略松）
        num_global = 18   # ↓ 原来20 → 18（整体稍减）

        protected = (START, GOAL) if endpoints is None else tuple(endpoints)

        def is_overlap(x, y, r, existing):
            # Chosen endpoints must not be trapped inside an obstacle.
            if any(np.linalg.norm(np.array([x, y]) - np.asarray(endpoint, dtype=float)) <= r + 0.005
                   for endpoint in protected):
                return True
            for ox, oy, orr in existing:
                if np.hypot(x - ox, y - oy) < (r + orr + 0.06):
                    return True
            return False

        # =========================
        # 3. 中心密集（高斯分布）
        # =========================
        center = np.array([10, 10])

        count = 0
        attempts = 0
        while count < num_center and attempts < 3000:
            attempts += 1

            # ↓ 稍微扩散一点（3.0 → 3.3）
            x, y = center + rng.randn(2) * 3.3

            if not (1 < x < 19 and 1 < y < 19):
                continue

            r = rng.uniform(0.3, 0.75)

            if not is_overlap(x, y, r, obstacles + boundary_obs):
                obstacles.append((x, y, r))
                count += 1

        # =========================
        # 4. 全局随机障碍
        # =========================
        count = 0
        attempts = 0
        while count < num_global and attempts < 3000:
            attempts += 1

            x = rng.uniform(1, 19)
            y = rng.uniform(1, 19)
            r = rng.uniform(0.25, 0.7)

            if not is_overlap(x, y, r, obstacles + boundary_obs):
                obstacles.append((x, y, r))
                count += 1

        return boundary_obs + obstacles
