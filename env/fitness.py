from env.path_util import path_length, min_obstacle_dist, collision
import numpy as np
from config import GOAL
def fitness(path, obstacles, safe_radius=0.005, angle_penalty_coef=0.2, path_penalty=0.5, boundary_penalty=100.0, collision_penalty=40.0):
    """
    改进的适应度函数
    :param path: 路径点列表，每个点为 (x, y)
    :param obstacles: 障碍物列表，每个障碍物为 (x, y, radius)
    :param safe_radius: 安全距离阈值，小于此距离产生平方惩罚
    :param angle_penalty_coef: 角度变化惩罚系数
    :param boundary_penalty: 边界惩罚系数（超出边界的距离平方乘以该系数）
    :param collision_penalty: 碰撞惩罚（一旦发生线段碰撞或点碰撞）
    :return: 适应度值（越小越好）
    :path_penalty: 111
    """
    # 1. 长度项
    length = path_length(path)

    # 2. 安全项（平滑惩罚）
    safe = 0.0
    max_angle=0.0
    for p in path:
        # 计算到最近障碍物的距离
        d = min_obstacle_dist(p, obstacles)
        # 如果距离小于安全半径，累加平方惩罚
        if d < safe_radius:
            safe += (safe_radius - d) ** 2
    # 安全项系数可调，此处设为 15.0 使安全项与长度项量级相当
    safe *= 15.0

    # 3. 线段碰撞检测（点碰撞已包含在线段检测中，但保留原有点碰撞作为备用）
    collide = False
    # 检查每个线段是否与任何障碍物相交
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        # 假设 path_util 提供了 segment_collision 函数
        # 如果没有，可调用自定义函数，此处写为注释示意
        if segment_collision(p1, p2, obstacles):
            collide = True
            break
    # 也可保留原有点碰撞检测作为补充（但线段碰撞已涵盖点碰撞）
    if not collide:
        collide = any(collision(p, obstacles) for p in path)

    # 4. 角度变化惩罚（平滑性）
    angle_penalty = 0.0
    if len(path) >= 3:
        for i in range(1, len(path)-1):
            # 计算向量
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])
            # 计算夹角（弧度），取绝对值
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                # 限制余弦值范围
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)
                if angle>max_angle:
                    max_angle=angle
                angle_penalty += angle
        angle_penalty=angle_penalty-max_angle+max_angle*2
        for i in range(len(path)-1):
            p1=path[i]
            p2=path[i+1]
            dist_before = np.hypot(p1[0] - GOAL[0], p1[1] - GOAL[1])
            dist_after = np.hypot(p2[0] - GOAL[0], p2[1] - GOAL[1])
            if dist_after>dist_before:
                angle_penalty+=0.2*(dist_after-dist_before)    #折线的另外一个维度的体现也就是是否保持向前的趋势。




    angle_penalty *= angle_penalty_coef

    # 5. 边界惩罚
    boundary = 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = 0.0, 20.0
    for p in path:
        if p[0] < x_min:
            boundary += (x_min - p[0]) ** 2
        elif p[0] > x_max:
            boundary += (p[0] - x_max) ** 2
        if p[1] < y_min:
            boundary += (y_min - p[1]) ** 2
        elif p[1] > y_max:
            boundary += (p[1] - y_max) ** 2
    boundary *= boundary_penalty

    # 6. 总适应度
    total = path_penalty*length + 0.05*safe + angle_penalty + boundary

    if collide:
        total += collision_penalty

    return total


# 辅助函数：线段与圆形障碍物碰撞检测（如果 path_util 未提供）
def segment_collision(p1, p2, obstacles):
    """
    检查线段 p1-p2 是否与任何圆形障碍物相交
    p1, p2: (x, y)
    obstacles: [(x, y, radius), ...]
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    for (ox, oy, r) in obstacles:
        o = np.array([ox, oy])
        # 线段到圆心的最近距离
        v = p2 - p1
        w = o - p1
        t = np.dot(w, v) / np.dot(v, v) if np.dot(v, v) != 0 else 0
        t = max(0.0, min(1.0, t))
        closest = p1 + t * v
        dist = np.linalg.norm(o - closest)
        if dist <= r:
            return True
    return False