import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import START, GOAL
from env.fitness import fitness, fitness_batch, clearance_batch
from env.fixed_env import MapEnv
from env.path_util import collision
from env.fitness import segment_collision
from pso.particle import Particle, init_population


def test_map_seed_matches_legacy():
    legacy = MapEnv().generate_mixed()
    for seed in (42, 0, 7):
        a = MapEnv(seed).generate_mixed()
        b = MapEnv(seed).generate_mixed(rng=np.random.RandomState(seed))
        assert np.allclose(a, b), f"map mismatch for seed {seed}"
    assert np.allclose(legacy, MapEnv(42).generate_mixed(rng=np.random.RandomState(42)))


def test_fitness_batch_matches_scalar():
    obstacles = MapEnv(42).generate_mixed(rng=np.random.RandomState(42))
    rng = np.random.RandomState(0)
    paths = rng.uniform(-3, 23, size=(300, 10, 2))
    paths[:, 0], paths[:, -1] = START, GOAL
    paths[0, 3] = paths[0, 4]
    paths[1] = np.linspace(START, GOAL, 10)
    paths = np.concatenate([paths, init_population(100, 10, np.random.RandomState(1))])

    f_batch, collide_batch = fitness_batch(paths, obstacles)
    for i, p in enumerate(paths):
        assert abs(fitness(p, obstacles) - f_batch[i]) < 1e-8, f"fitness mismatch at {i}"
        seg_hit = any(segment_collision(p[k], p[k + 1], obstacles) for k in range(len(p) - 1))
        pt_hit = any(collision(q, obstacles) for q in p)
        assert collide_batch[i] == (seg_hit or pt_hit), f"collision mismatch at {i}"


def _legacy_seg_clearance(p1, p2, obstacles):
    x1, y1 = p1
    dx, dy = p2[0] - x1, p2[1] - y1
    len2 = dx * dx + dy * dy
    best = float("inf")
    for ox, oy, r in obstacles:
        t = 0.0 if len2 == 0 else max(0.0, min(1.0, -((x1 - ox) * dx + (y1 - oy) * dy) / len2))
        best = min(best, np.hypot(x1 + t * dx - ox, y1 + t * dy - oy) - r)
    return best


def test_clearance_batch_matches_legacy_reward_terms():
    obstacles = MapEnv(42).generate_mixed(rng=np.random.RandomState(42))
    paths = init_population(100, 10, np.random.RandomState(2))
    clear, pen = clearance_batch(paths, obstacles)
    for i, p in enumerate(paths):
        segs = [_legacy_seg_clearance(p[k], p[k + 1], obstacles) for k in range(len(p) - 1)]
        assert abs(min(segs) - clear[i]) < 1e-8
        assert abs(sum(np.exp(-s) - 1 for s in segs if s < 0) - pen[i]) < 1e-8


def test_init_population_matches_particle():
    obstacles = MapEnv(42).generate_mixed(rng=np.random.RandomState(42))
    np.random.seed(123)
    legacy = np.array([Particle(10, obstacles).pos for _ in range(20)])
    new = init_population(20, 10, np.random.RandomState(123))
    assert np.allclose(legacy, new)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[ok] {name}")
