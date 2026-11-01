import numpy as np

class MapEnv:

    def generate_mixed(self):
        obs = []

        # 地图边界扩大为 0~20
        x_min, x_max = 0, 20
        y_min, y_max = 0, 20

        # ========== 迷宫墙：多个水平墙，均匀分布 ==========
        # 水平墙1: y = 5，从 x=3 到 17，中间留缺口（x=10附近）
        for i in range(3, 18):
            if i < 9 or i > 11:   # 缺口在10左右
                obs.append((i, 5, 0.35))
        # 水平墙2: y = 15，从 x=3 到 17，同样留缺口
        for i in range(3, 18):
            if i < 9 or i > 11:
                obs.append((i, 15, 0.35))
        # 垂直墙: x = 10，从 y=3 到 17，中间留缺口（y=10附近）
        for j in range(3, 18):
            if j < 8 or j > 12:
                obs.append((10, j, 0.35))

        # ========== 陷阱：多个小型障碍物区域，均匀分布 ==========
        # 左下区域陷阱 (3,3) 附近
        for i in range(2, 5):
            for j in range(2, 5):
                obs.append((i, j, 0.3))
        # 右上区域陷阱 (16,16) 附近
        for i in range(16, 19):
            for j in range(16, 19):
                obs.append((i, j, 0.3))
        # 中央区域陷阱 (10,10) 附近
        for i in range(9, 12):
            for j in range(9, 12):
                obs.append((i, j, 0.3))

        # ========== 杂乱小障碍：均匀随机分布，数量增加 ==========
        num_clutter = 30   # 增加数量使分布更均匀
        for _ in range(num_clutter):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            r = np.random.uniform(0.2, 0.5)
            obs.append((x, y, r))

        return obs