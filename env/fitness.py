"""Path cost: length, total/max turn, boundary violation, worst penetration."""
import numpy as np
from config import GOAL, MAP_MIN, MAP_MAX

PENETRATION_WEIGHT = 40.0


def segment_clearances(paths, obstacles):
    paths = np.asarray(paths, dtype=np.float64)
    obs = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    if not len(obs):
        return np.full((len(paths), paths.shape[1] - 1), np.inf)
    seg = np.diff(paths, axis=1)
    vv = (seg ** 2).sum(-1)[..., None]
    delta = obs[:, :2] - paths[:, :-1, None, :]
    t = (delta * seg[:, :, None, :]).sum(-1) / np.where(vv > 0, vv, 1.0)
    t = np.where(vv > 0, np.clip(t, 0, 1), 0)
    closest = paths[:, :-1, None, :] + t[..., None] * seg[:, :, None, :]
    return (np.linalg.norm(obs[:, :2] - closest, axis=-1) - obs[:, 2]).min(axis=2)


def collision_metrics(paths, obstacles):
    """H = max over all segment/obstacle penetration depths; tangency is collision."""
    clearance = segment_clearances(paths, obstacles).min(axis=1)
    return np.maximum(-clearance, 0), clearance <= 0


def evaluate_paths(paths, obstacles, angle_penalty_coef=0.2, path_penalty=0.5,
                   boundary_penalty=100.0, collision_penalty=PENETRATION_WEIGHT,
                   map_min=MAP_MIN, map_max=MAP_MAX):
    paths = np.asarray(paths, dtype=np.float64)
    seg = np.diff(paths, axis=1)
    lengths = np.linalg.norm(seg, axis=-1)
    angle_cost = np.zeros(len(paths))
    if paths.shape[1] >= 3:
        valid = (lengths[:, :-1] > 1e-6) & (lengths[:, 1:] > 1e-6)
        denom = np.where(valid, lengths[:, :-1] * lengths[:, 1:], 1.0)
        cos = (seg[:, :-1] * seg[:, 1:]).sum(-1) / denom
        angles = np.where(valid, np.arccos(np.clip(cos, -1, 1)), 0)
        angle_cost = angles.sum(axis=1) + angles.max(axis=1)
    outside = np.maximum(map_min - paths, 0) ** 2 + np.maximum(paths - map_max, 0) ** 2
    depth, collide = collision_metrics(paths, obstacles)
    cost = (path_penalty * lengths.sum(axis=1) + angle_penalty_coef * angle_cost
            + boundary_penalty * outside.sum(axis=(1, 2)) + collision_penalty * depth)
    return cost, collide, depth


def fitness_batch(paths, obstacles, safe_radius=0.005, angle_penalty_coef=0.2,
                  path_penalty=0.5, boundary_penalty=100.0,
                  collision_penalty=PENETRATION_WEIGHT, goal=GOAL,
                  map_min=MAP_MIN, map_max=MAP_MAX):
    # safe_radius/goal remain accepted for legacy callers; neither contributes to this cost.
    f, collide, _ = evaluate_paths(paths, obstacles, angle_penalty_coef, path_penalty,
                                   boundary_penalty, collision_penalty, map_min, map_max)
    return f, collide


def fitness(path, obstacles, safe_radius=0.005, angle_penalty_coef=0.2,
            path_penalty=0.5, boundary_penalty=100.0, collision_penalty=PENETRATION_WEIGHT):
    return float(fitness_batch(np.asarray(path)[None], obstacles, safe_radius,
                              angle_penalty_coef, path_penalty, boundary_penalty,
                              collision_penalty)[0][0])


def segment_collision(p1, p2, obstacles):
    return bool(collision_metrics(np.asarray([[p1, p2]]), obstacles)[1][0])


def clearance_batch(paths, obstacles):
    """Legacy diagnostic only; the new fitness/reward use collision_metrics' max depth."""
    clear = segment_clearances(paths, obstacles)
    return clear.min(axis=1), np.expm1(np.maximum(-clear, 0)).sum(axis=1)
