import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import START, GOAL, EPISODES, STEPS_PER_EP, UPDATE_STEPS
from env.path_util import path_length, min_obstacle_dist, collision
from env.fitness import fitness, segment_collision
from pso.swarm import update_particle
from pso.particle import Particle

# =========================================================
# 统一常量
# =========================================================
ANGLE_PENALTY_COEF = 0.2
SAFE_RADIUS = 0.005
PATH_PENALTY = 0.5
BOUNDARY_PENALTY = 100.0
COLLISION_PENALTY = 40.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 1) 粒子状态修正
# =========================================================
def ensure_particle_fields(p, obstacles):
    p.pos = np.asarray(p.pos, dtype=np.float32)
    p.vel = np.asarray(getattr(p, "vel", np.zeros_like(p.pos)), dtype=np.float32)

    if not hasattr(p, "pbest") or p.pbest is None:
        p.pbest = p.pos.copy()
    else:
        p.pbest = np.asarray(p.pbest, dtype=np.float32)

    if not hasattr(p, "f") or p.f is None:
        p.f = fitness(p.pos, obstacles)

    if not hasattr(p, "pbest_f") or p.pbest_f is None:
        p.pbest_f = p.f

    if not hasattr(p, "prev_f") or p.prev_f is None:
        p.prev_f = p.f

    if not hasattr(p, "stagnation") or p.stagnation is None:
        p.stagnation = 0

    if not hasattr(p, "start_point") or p.start_point is None:
        p.start_point = p.pos[0].copy()
    else:
        p.start_point = np.asarray(p.start_point, dtype=np.float32)

    if not hasattr(p, "end_point") or p.end_point is None:
        p.end_point = p.pos[-1].copy()
    else:
        p.end_point = np.asarray(p.end_point, dtype=np.float32)

    p.obstacles = obstacles
    return p


def make_particle_from_path(path, obstacles):
    path = np.asarray(path, dtype=np.float32)
    p = Particle(len(path), obstacles)
    p.pos = path.copy()
    p.vel = np.zeros_like(path, dtype=np.float32)
    p.pbest = path.copy()
    p.f = fitness(path, obstacles)
    p.pbest_f = p.f
    p.prev_f = p.f
    p.stagnation = 0

    p.start_point = p.pos[0].copy()
    p.end_point = p.pos[-1].copy()

    p.obstacles = obstacles
    return p


# =========================================================
# 2) 轨迹收集器
# =========================================================
class TrajectoryCollector:
    def __init__(self):
        self.trajectories = []

    def add_trajectory(self, traj_dict):
        self.trajectories.append(traj_dict)

    def save(self, path="trajectory_dataset.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.trajectories, f)
        print(f"[Dataset Saved] {len(self.trajectories)} trajectories written to {path}")
        self.trajectories = []


# =========================================================
# 3) 状态构造
# =========================================================
def get_state(p, gbest_f, delta_f, delta_g):
    return np.array([
        p.f,
        p.pbest_f,
        gbest_f,
        delta_f,
        delta_g,
    ], dtype=np.float32)


# =========================================================
# 4) 路径距离与碰撞检测
# =========================================================
def min_path_distance(path, obstacles):
    min_d = float("inf")
    path = np.asarray(path, dtype=float)

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        len2 = dx * dx + dy * dy

        if len2 == 0:
            for (ox, oy, r) in obstacles:
                d = np.hypot(x1 - ox, y1 - oy) - r
                min_d = min(min_d, d)
            continue

        for (ox, oy, r) in obstacles:
            vx = x1 - ox
            vy = y1 - oy
            t = - (vx * dx + vy * dy) / len2
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            d = np.hypot(px - ox, py - oy) - r
            min_d = min(min_d, d)

    return min_d


def is_path_collision(path, obstacles):
    for i in range(len(path) - 1):
        if segment_collision(path[i], path[i + 1], obstacles):
            return True
    return False


# =========================================================
# 5) 奖励函数：保持不变
# =========================================================
def compute_reward(old_f, new_f, old_g, new_g,
                   collision_flag, b, c,
                   path, obstacles,
                   prev_d, gbest_d):
    delta_local = old_f - new_f
    reward = 0.5 * delta_local

    d = min_path_distance(path, obstacles)
    delta_d = d - prev_d
    reward += 15 * delta_d

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        seg_d = min_path_distance([p1, p2], obstacles)
        if seg_d < 0:
            cc = -seg_d
            reward -= 1 * (np.exp(cc) - 1)

    reward -= 1 * np.log(1 + b)

    return float(reward), d


# =========================================================
# 6) 适应度分解：与 fitness 同口径
# =========================================================
def fitness_breakdown(path, obstacles,
                      safe_radius=SAFE_RADIUS,
                      angle_penalty_coef=ANGLE_PENALTY_COEF,
                      path_penalty=PATH_PENALTY,
                      boundary_penalty=BOUNDARY_PENALTY,
                      collision_penalty=COLLISION_PENALTY):
    """
    与 env.fitness.fitness 同公式，额外返回分解项
    """
    length = path_length(path)

    safe = 0.0
    max_angle = 0.0
    for p in path:
        d = min_obstacle_dist(p, obstacles)
        if d < safe_radius:
            safe += (safe_radius - d) ** 2
    safe *= 15.0

    collide = False
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        if segment_collision(p1, p2, obstacles):
            collide = True
            break
    if not collide:
        collide = any(collision(p, obstacles) for p in path)

    angle_penalty_raw = 0.0
    if len(path) >= 3:
        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i - 1])
            v2 = np.array(path[i + 1]) - np.array(path[i])
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)
                if angle > max_angle:
                    max_angle = angle
                angle_penalty_raw += angle

        angle_penalty_raw = angle_penalty_raw - max_angle + max_angle * 2

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            dist_before = np.hypot(p1[0] - GOAL[0], p1[1] - GOAL[1])
            dist_after = np.hypot(p2[0] - GOAL[0], p2[1] - GOAL[1])
            if dist_after > dist_before:
                angle_penalty_raw += 0.2 * (dist_after - dist_before)

    angle_penalty = angle_penalty_raw * angle_penalty_coef

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

    total = path_penalty * length + 0.05 * safe + angle_penalty + boundary
    if collide:
        total += collision_penalty

    return total, {
        "length": length,
        "safe": safe,
        "angle_penalty_raw": angle_penalty_raw,
        "angle_penalty": angle_penalty,
        "boundary": boundary,
        "collide": collide,
        "total": total,
    }


# =========================================================
# 7) 可视化
# =========================================================
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


import numpy as np
import matplotlib.pyplot as plt

# ========== 全局字体设置（在代码开头运行一次即可）==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

def visualize_path_pair(before_path, after_path, obstacles, ep):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _draw(ax, path):
        # 绘制障碍物
        for ox, oy, radius in obstacles:
            circle = plt.Circle((ox, oy), radius, color='red', alpha=0.5, ec='black', lw=0.5)
            ax.add_patch(circle)

        # 绘制路径和起终点
        if path is not None and len(path) > 0:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, marker='o', markersize=4, label='Path')
            ax.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
            ax.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='Goal')

        # 轴标签：使用词语描述
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # 分别绘制左右子图（不再传入 title 参数）
    _draw(axes[0], before_path)
    _draw(axes[1], after_path)

    # 移除所有图片内标题（符合 IEEE 要求）
    # 原先的 ax.set_title() 和 fig.suptitle() 已全部删除

    plt.tight_layout()
    plt.show()


# =========================================================
# 8) 圆弧化细化
# =========================================================
def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
    n = _norm(v)
    if n < eps:
        return None
    return v / n


def _angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 < eps or n2 < eps:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.arccos(cosv))


def _closest_obstacle_by_distance(origin: np.ndarray, obstacles):
    origin = np.asarray(origin, dtype=np.float64)
    best_obs = None
    best_dist = float("inf")

    for obs in obstacles:
        ox, oy, r = obs
        c = np.array([ox, oy], dtype=np.float64)
        d = np.linalg.norm(origin - c) - r
        if d < best_dist:
            best_dist = d
            best_obs = obs

    return best_obs


def _ray_first_intersection_with_obstacles(
    origin: np.ndarray,
    direction: np.ndarray,
    obstacles,
    eps: float = 1e-8
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    d = _normalize(direction, eps=eps)
    if d is None:
        return None, None

    origin = np.asarray(origin, dtype=np.float64)

    best_t = None
    best_pt = None

    for (ox, oy, r) in obstacles:
        c = np.array([ox, oy], dtype=np.float64)

        f = origin - c
        a = np.dot(d, d)
        b = 2.0 * np.dot(f, d)
        cc = np.dot(f, f) - r * r
        disc = b * b - 4.0 * a * cc

        if disc < 0:
            continue

        sqrt_disc = np.sqrt(max(disc, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        for t in (t1, t2):
            if t > eps and (best_t is None or t < best_t):
                best_t = t
                best_pt = origin + t * d

    if best_pt is None:
        return None, None

    return best_pt.astype(np.float32), float(best_t)


def _segment_parameter(pt: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < eps:
        return 0.0
    return float(np.dot(pt - a, ab) / denom)


def _append_unique(path_list: List[np.ndarray], pt: np.ndarray, tol: float = 1e-6):
    pt = np.asarray(pt, dtype=np.float32)
    if len(path_list) == 0:
        path_list.append(pt.copy())
        return

    if np.linalg.norm(np.asarray(path_list[-1], dtype=np.float32) - pt) > tol:
        path_list.append(pt.copy())


def _build_arc_at_corner(
    prev_pt: np.ndarray,
    curr_pt: np.ndarray,
    next_pt: np.ndarray,
    obstacles,
    inner_angle_thresh: float = np.deg2rad(130.0),
    fallback_center_len: float = 0.8,
    insert_points: int = 3
) -> Optional[List[np.ndarray]]:
    """
    圆弧方向修复版：
    用真实行进方向 prev->curr->next 来判断转向，
    生成和原折线趋势一致的劣弧。
    """
    prev_pt = np.asarray(prev_pt, dtype=np.float32)
    curr_pt = np.asarray(curr_pt, dtype=np.float32)
    next_pt = np.asarray(next_pt, dtype=np.float32)

    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt

    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None

    u1 = v1 / n1
    u2 = v2 / n2

    inner_angle = _angle_between(u1, u2)
    if inner_angle >= inner_angle_thresh:
        return None

    bisector = u1 + u2
    bnorm = _norm(bisector)
    if bnorm < 1e-8:
        return None
    bisector = bisector / bnorm

    hit_pt, hit_dist = _ray_first_intersection_with_obstacles(curr_pt, bisector, obstacles)

    if hit_pt is None or hit_dist is None or hit_dist > 5.5:
        adaptive_len = float(np.clip(0.35 * min(n1, n2), 0.6, 4.0))
        center = curr_pt + adaptive_len * bisector
    else:
        center = hit_pt
    center = np.asarray(center, dtype=np.float32)

    d_center = _norm(center - curr_pt)
    if d_center < 1e-8:
        d_center = fallback_center_len

    tan_len = d_center * np.cos(inner_angle / 2.0)
    tan_len = min(tan_len, 0.45 * min(n1, n2))
    if tan_len < 1e-6:
        return None

    raw_t_in = curr_pt + u1 * tan_len
    raw_t_out = curr_pt + u2 * tan_len

    left_t = _segment_parameter(raw_t_in, prev_pt, curr_pt)
    right_t = _segment_parameter(raw_t_out, curr_pt, next_pt)

    left_outside = (left_t < 0.0) or (left_t > 1.0)
    right_outside = (right_t < 0.0) or (right_t > 1.0)

    left_anchor = prev_pt.copy() if left_outside else np.asarray(raw_t_in, dtype=np.float32)
    right_anchor = next_pt.copy() if right_outside else np.asarray(raw_t_out, dtype=np.float32)

    if left_outside and right_outside:
        if segment_collision(prev_pt, next_pt, obstacles):
            return None
        return [prev_pt.copy(), next_pt.copy()]

    radius = _norm(left_anchor - center)
    if radius < 1e-8:
        radius = d_center * np.sin(inner_angle / 2.0)
    if radius < 1e-8:
        return None

    incoming = curr_pt - prev_pt
    outgoing = next_pt - curr_pt
    turn_cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])

    if abs(turn_cross) < 1e-10:
        return None

    ccw = turn_cross > 0

    ang1 = float(np.arctan2(left_anchor[1] - center[1], left_anchor[0] - center[0]))
    ang2 = float(np.arctan2(right_anchor[1] - center[1], right_anchor[0] - center[0]))

    if ccw:
        if ang2 <= ang1:
            ang2 += 2.0 * np.pi
    else:
        if ang2 >= ang1:
            ang2 -= 2.0 * np.pi

    delta = ang2 - ang1
    if abs(delta) > np.pi:
        return None

    arc_angles = np.linspace(ang1, ang2, insert_points + 2)
    arc_points = []
    for a in arc_angles[1:-1]:
        x = center[0] + radius * np.cos(a)
        y = center[1] + radius * np.sin(a)
        arc_points.append(np.array([x, y], dtype=np.float32))

    proposed = [left_anchor] + arc_points + [right_anchor]

    if np.linalg.norm(prev_pt - left_anchor) > 1e-8:
        if segment_collision(prev_pt, left_anchor, obstacles):
            return None

    if np.linalg.norm(right_anchor - next_pt) > 1e-8:
        if segment_collision(right_anchor, next_pt, obstacles):
            return None

    for j in range(len(proposed) - 1):
        if segment_collision(proposed[j], proposed[j + 1], obstacles):
            return None

    return proposed


def refine_path_arc(
    path,
    obstacles,
    max_points: int = 80,
    inner_angle_thresh: float = np.deg2rad(150.0)
):
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 3:
        return path.copy()

    new_path = [path[0].copy()]

    for i in range(1, len(path) - 1):
        prev_pt = path[i - 1]
        curr_pt = path[i]
        next_pt = path[i + 1]

        arc_seg = _build_arc_at_corner(
            prev_pt=prev_pt,
            curr_pt=curr_pt,
            next_pt=next_pt,
            obstacles=obstacles,
            inner_angle_thresh=inner_angle_thresh,
            fallback_center_len=0.8,
            insert_points=3
        )

        if arc_seg is not None and len(new_path) + len(arc_seg) <= max_points:
            for pt in arc_seg:
                _append_unique(new_path, pt)
        else:
            _append_unique(new_path, curr_pt.copy())

        if len(new_path) >= max_points - 1:
            break

    if len(new_path) < max_points:
        _append_unique(new_path, path[-1].copy())

    new_path = np.array(new_path[:max_points], dtype=np.float32)
    new_path[0] = path[0]
    new_path[-1] = path[-1]
    return new_path


# =========================================================
# 9) 局部更新：只对最佳路径，不考虑 gbest
# =========================================================
def action_to_local_params(action):
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if len(a) == 0:
        return 0.45, 1.2, 0.05

    a0 = float(np.tanh(a[0]))
    a1 = float(np.tanh(a[1])) if len(a) > 1 else 0.0
    a2 = float(np.tanh(a[2])) if len(a) > 2 else 0.0

    w = 0.35 + 1.20 * (a0 + 1.0) * 0.5
    c1 = 1.0 + 2.0 * (a1 + 1.0) * 0.5
    noise_scale = 0.2 + 0.1 * (a2 + 1.0) * 0.5
    return w, c1, noise_scale


def update_particle_local_no_gbest(p, action):
    w, c1, noise_scale = action_to_local_params(action)

    pos = p.pos.copy()
    vel = p.vel.copy()

    for k in range(1, len(pos) - 1):
        r1 = np.random.rand(2).astype(np.float32)
        cognitive = c1 * r1 * (p.pbest[k] - pos[k])
        noise = np.random.randn(2).astype(np.float32) * noise_scale

        vel[k] = w * vel[k] + cognitive + noise * 2.5
        pos[k] = pos[k] + vel[k]

    pos[:, 0] = np.clip(pos[:, 0], 0.0, 20.0)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, 20.0)
    pos[0] = p.start_point
    pos[-1] = p.end_point

    p.pos = pos
    p.vel = vel


# =========================================================
# 10) 训练主函数
# =========================================================
def train(td3, buffer, swarm, visualize=True):
    if swarm is None or len(swarm) == 0:
        raise ValueError("swarm 不能为空。")

    obstacles = swarm[0].obstacles
    swarm = [ensure_particle_fields(p, obstacles) for p in swarm]

    device = td3.device
    collector = TrajectoryCollector()

    LOCAL_ANGLE_COEF =0.2

    gbest = min(swarm, key=lambda p: p.f)
    gbest_f = gbest.f
    gbest_pos = gbest.pos.copy()
    gbest_d = min_path_distance(gbest_pos, obstacles)

    history = []
    c_global = 0
    prev_dists = [min_path_distance(p.pos, p.obstacles) for p in swarm]

    local_only_mode = False
    last_step_delta_g = 0.0

    start_pt = np.asarray(START, dtype=np.float32)
    goal_pt = np.asarray(GOAL, dtype=np.float32)

    for ep in range(EPISODES):
        for step in range(STEPS_PER_EP):
            old_g = gbest_f

            states = []
            actions = []

            for p in swarm:
                delta_f = p.prev_f - p.f
                s = get_state(p, gbest_f, delta_f, last_step_delta_g)

                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                with torch.no_grad():
                    a = td3.actor(s_tensor).detach().cpu().numpy()[0]

                states.append(s)
                actions.append(a)

            if not local_only_mode:
                for i, p in enumerate(swarm):
                    update_particle(p, gbest_pos, actions[i])
                    p.pos[0] = start_pt
                    p.pos[-1] = goal_pt
            else:
                for i, p in enumerate(swarm):
                    update_particle_local_no_gbest(p, actions[i])

            for p in swarm:
                p.prev_f = p.f

                if local_only_mode:
                    p.f = fitness(p.pos, p.obstacles, angle_penalty_coef=LOCAL_ANGLE_COEF)
                else:
                    p.f = fitness(p.pos, p.obstacles, angle_penalty_coef=ANGLE_PENALTY_COEF)

                if p.f < p.pbest_f:
                    p.pbest = p.pos.copy()
                    p.pbest_f = p.f
                    p.stagnation = 0
                else:
                    p.stagnation += 1

            new_best = min(swarm, key=lambda p: p.f)
            if new_best.f < gbest_f:
                gbest_f = new_best.f
                gbest_pos = new_best.pos.copy()
                gbest_d = min_path_distance(gbest_pos, obstacles)
                c_global = 0
            else:
                c_global += 1

            current_step_delta_g = old_g - gbest_f
            last_step_delta_g = current_step_delta_g

            new_g = gbest_f
            for i, p in enumerate(swarm):
                s = states[i]
                a = actions[i]

                delta_f = p.prev_f - p.f
                delta_g = current_step_delta_g
                s_ = get_state(p, gbest_f, delta_f, delta_g)

                collision_flag = is_path_collision(p.pos, p.obstacles)

                r, current_d = compute_reward(
                    p.prev_f, p.f,
                    old_g, new_g,
                    collision_flag,
                    p.stagnation,
                    c_global,
                    p.pos,
                    p.obstacles,
                    prev_d=prev_dists[i],
                    gbest_d=gbest_d
                )
                prev_dists[i] = current_d

                buffer.add(s, a, r, s_)

            print(f"[Episode {ep:3d} | Step {step:3d}] Global Best = {gbest_f:.6f}")

        # =====================================================
        # episode 结束后：圆弧化更新
        # =====================================================
        if gbest_f < 11:
            before_refine_path = gbest_pos.copy()

            refined_path = refine_path_arc(
                gbest_pos,
                obstacles=obstacles,
                max_points=80,
                inner_angle_thresh=np.deg2rad(150.0)
            )

            refined_f = fitness(refined_path, obstacles, angle_penalty_coef=ANGLE_PENALTY_COEF)
            refined_d = min_path_distance(refined_path, obstacles)

            changed = (
                len(refined_path) != len(gbest_pos)
                or not np.allclose(refined_path, gbest_pos, atol=1e-5)
            )

            print(
                f"\n[Episode {ep:3d}] Arc-refinement applied "
                f"(changed={changed}), "
                f"old_fitness={gbest_f:.6f}, new_fitness={refined_f:.6f}, "
                f"old_points={len(gbest_pos)}, new_points={len(refined_path)}"
            )

            if visualize:
                visualize_path_pair(before_refine_path, refined_path, obstacles, ep)

            gbest_pos = refined_path.copy()
            gbest_f = refined_f
            gbest_d = refined_d

            best_particle = make_particle_from_path(gbest_pos, obstacles)
            best_particle.f = gbest_f
            best_particle.pbest_f = gbest_f
            best_particle.prev_f = gbest_f
            best_particle.pbest = gbest_pos.copy()

            swarm = [best_particle]
            prev_dists = [gbest_d]
            local_only_mode = True
            c_global = 0
            last_step_delta_g = 0.0

        if gbest_f < 50:
            dummy_particle = make_particle_from_path(gbest_pos, obstacles)
            dummy_particle.f = gbest_f
            dummy_particle.pbest_f = gbest_f

        print(f"\n--- Episode {ep} finished. Training TD3 with {UPDATE_STEPS} updates ---")
        for _ in range(UPDATE_STEPS):
            if hasattr(td3, "train_td3"):
                td3.train_td3(buffer)
            else:
                td3.train(buffer)

        # 只在最后一个 episode 结束后输出一次
        if ep == EPISODES - 1:
            gbest_total, gbest_info = fitness_breakdown(
                gbest_pos,
                obstacles,
                safe_radius=SAFE_RADIUS,
                angle_penalty_coef=ANGLE_PENALTY_COEF,
                path_penalty=PATH_PENALTY,
                boundary_penalty=BOUNDARY_PENALTY,
                collision_penalty=COLLISION_PENALTY
            )

            gbest_angle_percent = 0.0
            if abs(gbest_total) > 1e-12:
                gbest_angle_percent = 100.0 * gbest_info["angle_penalty"] / gbest_total

            print("\n======== gbest_pos 适应度分解（final episode）========")
            print(f"路径点数                  = {len(gbest_pos)}")
            print(f"路径长度 length           = {gbest_info['length']:.6f}")
            print(f"安全惩罚 safe             = {gbest_info['safe']:.6f}")
            print(f"角度惩罚原始 angle_raw    = {gbest_info['angle_penalty_raw']:.6f}")
            print(f"角度惩罚 angle_penalty    = {gbest_info['angle_penalty']:.6f}")
            print(f"边界惩罚 boundary         = {gbest_info['boundary']:.6f}")
            print(f"是否碰撞 collide          = {gbest_info['collide']}")
            print(f"总适应度 total            = {gbest_info['total']:.6f}")
            print(f"角度惩罚占总适应度百分比   = {gbest_angle_percent:.2f}%")

            if abs(gbest_total - gbest_f) > 1e-5:
                print(f"[Warn] gbest_f={gbest_f:.6f} 与分解计算 total={gbest_total:.6f} 有轻微差异")

        history.append(gbest_f)
        print(f"Episode {ep} completed | Best Fitness: {gbest_f:.6f}\n")

    if visualize:
        visualize_path(gbest_pos, obstacles, title=f"Best Path (Fitness: {gbest_f:.4f})")

    if hasattr(td3, "save_model"):
        td3.save_model("result_model/td3_finals")

    collector.save("trajectory_dataset.pkl")

    return gbest_pos, history, collector