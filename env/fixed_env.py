import numpy as np
from config import START, GOAL

class MapEnv:
    def generate_mixed(self):
        # 固定地图、粒子群初始化以及后续 NumPy 随机序列，
        # 用于隔离研究 TD3 网络初始权重带来的影响。
        np.random.seed(42)

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

        def is_overlap(x, y, r, existing):
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
            x, y = center + np.random.randn(2) * 3.3

            if not (1 < x < 19 and 1 < y < 19):
                continue

            r = np.random.uniform(0.3, 0.75)

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

            x = np.random.uniform(1, 19)
            y = np.random.uniform(1, 19)
            r = np.random.uniform(0.25, 0.7)

            if not is_overlap(x, y, r, obstacles + boundary_obs):
                obstacles.append((x, y, r))
                count += 1

        return boundary_obs + obstacles
