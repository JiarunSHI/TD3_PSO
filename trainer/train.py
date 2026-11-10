import numpy as np

import copy

import torch

from config import (START, GOAL, MAP_MIN, MAP_MAX, V_MAX, PARAM_LOW, PARAM_HIGH,
                    EXPL_DECAY_ITERS, REWARD_PBEST, REWARD_GBEST,
                    REWARD_COLLISION_PROGRESS, REWARD_FIRST_FEASIBLE)
from drl.replay_buffer import ReplayBuffer
from env.fitness import evaluate_paths, PENETRATION_WEIGHT
from pso.particle import init_population
from pso.swarm import pso_step

ACTION_DIM = 3
STATE_SCHEMA = "compact11_distances_worst_depth_v2"
STATE_NAMES = ("gap_pbest", "gap_gbest", "fitness", "velocity_pbest_cos",
               "velocity_gbest_cos", "velocity_turn_cos", "penetration", "collide", "speed_rms",
               "distance_pbest", "distance_gbest")
STATE_DIM = len(STATE_NAMES)
PARAM_NAMES = ("w", "c1", "c2")
REWARD_NAMES = ("pbest", "adjacent_fitness", "gbest", "adjacent_depth", "first_feasible")
_LOW, _HIGH = np.asarray(PARAM_LOW), np.asarray(PARAM_HIGH)


def scale_action(a):
    return _LOW + (np.clip(a, -1, 1) + 1) * 0.5 * (_HIGH - _LOW)


def _cosine(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.divide((a * b).sum(axis=1), denom, out=np.zeros(len(a)), where=denom > 1e-12)
    return np.clip(out, -1, 1)  # floating-point roundoff only


def depth_scale(obstacles):
    obs = np.asarray(obstacles).reshape(-1, 3)
    return float(obs[:, 2].max()) if len(obs) else 1.0


def build_states(f, pbest_f, gbest_f, pos, pbest_pos, gbest_pos, vel, prev_vel,
                 depth, collide, h_ref, v_max=V_MAX, start=None, goal=None):
    """Fixed geometric normalization shared by training and frozen evaluation.

    No time feature, running statistics or clipping of fitness/reward outliers.
    This compressed observation is not a complete Markov state.
    """
    start = START if start is None else np.asarray(start, dtype=np.float64)
    goal = GOAL if goal is None else np.asarray(goal, dtype=np.float64)
    f0 = max(0.5 * np.linalg.norm(goal - start), 1e-8)
    upper = (0.5 * (pos.shape[1] - 1) * np.sqrt(2) * (MAP_MAX - MAP_MIN)
             + 0.2 * (pos.shape[1] - 1) * np.pi + PENETRATION_WEIGHT * h_ref)
    def normalize(value):
        if np.any(value < -1e-8):
            raise ValueError("Negative fitness/gap violates state definition")
        return 2 * np.log1p(np.maximum(value, 0) / f0) / np.log1p(upper / f0) - 1
    v, old_v = vel[:, 1:-1], prev_vel[:, 1:-1]
    to_pbest = (pbest_pos - pos)[:, 1:-1]
    to_gbest = (gbest_pos[None] - pos)[:, 1:-1]
    def normalized_distance(delta):
        rms = np.sqrt(np.mean(delta ** 2, axis=(1, 2))) if delta.shape[1] else np.zeros(len(f))
        return 2 * rms / (MAP_MAX - MAP_MIN) - 1
    speed = np.sqrt(np.mean(v ** 2, axis=(1, 2))) / v_max if v.shape[1] else np.zeros(len(f))
    result = np.stack([
        normalize(f - pbest_f), normalize(f - gbest_f), normalize(f),
        _cosine(v, to_pbest), _cosine(v, to_gbest), _cosine(v, old_v),
        2 * depth / h_ref - 1, 2 * collide.astype(float) - 1, 2 * speed - 1,
        normalized_distance(to_pbest), normalized_distance(to_gbest),
    ], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite state")
    return result


def compute_rewards(old_f, new_f, old_pbest_f, new_pbest_f, old_gbest_f,
                    old_depth, new_depth, ever_feasible, new_collide, h_ref,
                    return_components=False, pbest_weight=REWARD_PBEST):
    components = np.stack([
        pbest_weight * np.maximum(np.log(old_pbest_f / new_pbest_f), 0),
        np.log(old_f / new_f),
        REWARD_GBEST * np.maximum(np.log(old_gbest_f / new_f), 0),
        REWARD_COLLISION_PROGRESS * (old_depth - new_depth) / h_ref,
        REWARD_FIRST_FEASIBLE * (~ever_feasible & ~new_collide),
    ], axis=1).astype(np.float32)
    rewards = components.sum(axis=1)
    return (rewards, components) if return_components else rewards


class FixedController:
    name = "fixed"

    def __init__(self, w, c1, c2):
        self.params = np.array([w, c1, c2], dtype=np.float64)

    def act(self, states, t, iters):
        return np.repeat(self.params[None], len(states), axis=0)

    def observe(self, states, rewards, next_states, t, iters):
        pass


class LinearWController:
    name = "ldiw"

    def __init__(self, w_start, w_end, c1, c2):
        self.w_start, self.w_end, self.c1, self.c2 = w_start, w_end, c1, c2

    def act(self, states, t, iters):
        w = self.w_start + (self.w_end - self.w_start) * t / max(iters - 1, 1)
        return np.repeat([[w, self.c1, self.c2]], len(states), axis=0)

    def observe(self, states, rewards, next_states, t, iters):
        pass


class RandomController:
    name = "random"

    def __init__(self, rng):
        self.rng = rng

    def act(self, states, t, iters):
        # One draw per PSO step, shared by every particle.
        shared = scale_action(self.rng.uniform(-1, 1, ACTION_DIM))
        return np.repeat(shared[None], len(states), axis=0)

    def observe(self, states, rewards, next_states, t, iters):
        pass


def policy_metrics(policy, unbounded_action, logits):
    boundary = np.abs(policy) >= 0.98
    clipped = np.abs(unbounded_action) > 1.0
    result = {"actor_boundary_ratio": float(boundary.mean()),
              "exploration_clip_ratio": float(clipped.mean()),
              "logit_gt3_ratio": float((np.abs(logits) > 3).mean())}
    for j, name in enumerate(PARAM_NAMES):
        result["actor_boundary_" + name] = float(boundary[:, j].mean())
        result["exploration_clip_" + name] = float(clipped[:, j].mean())
    return result


class FrozenTD3Controller:
    """Pure inference: no exploration, replay insertion or optimizer updates."""
    name = "td3"

    def __init__(self, td3):
        self.td3 = td3
        self.begin_episode()

    def begin_episode(self):
        self.policy_log = []
        self.last_metrics = {}

    def act(self, states, t, iters):
        policy = self.td3.act(states)
        self.policy_log.append(scale_action(policy).astype(np.float32))
        self.last_metrics = policy_metrics(policy, policy, self.td3.pre_tanh(states))
        self.last_metrics.update(sigma=0.0, warmup=0)
        return scale_action(policy)

    def observe(self, states, rewards, next_states, t, iters):
        pass

    def saturation(self, states, threshold=3.0):
        return float((np.abs(self.td3.pre_tanh(states)) > threshold).mean())


class TD3Controller(FrozenTD3Controller):
    """Keep this controller alive across episodes; warmup/noise use global PSO steps."""
    def __init__(self, td3, buffer, rng, warmup_iters, updates_per_iter, batch_size, expl_noise,
                 expl_decay_iters=EXPL_DECAY_ITERS):
        super().__init__(td3)
        self.buffer, self.rng = buffer, rng
        self.warmup_iters, self.updates_per_iter = warmup_iters, updates_per_iter
        self.batch_size, self.expl_noise = batch_size, expl_noise
        if not np.isfinite(expl_decay_iters) or expl_decay_iters <= 0:
            raise ValueError("expl_decay_iters must be finite and positive")
        self.expl_decay_iters, self.env_steps = expl_decay_iters, 0

    def act(self, states, t, iters):
        policy = self.td3.act(states)
        self.policy_log.append(scale_action(policy).astype(np.float32))
        warmup = self.env_steps < self.warmup_iters
        sigma = 0.0
        if warmup:
            unbounded = self.rng.uniform(-1, 1, policy.shape)
        else:
            elapsed = self.env_steps - self.warmup_iters
            sigma = self.expl_noise[1] + (self.expl_noise[0] - self.expl_noise[1]) * np.exp(
                -elapsed / self.expl_decay_iters)
            unbounded = policy + self.rng.normal(0, sigma, policy.shape)
        self._raw = np.clip(unbounded, -1, 1)
        self.last_metrics = policy_metrics(policy, unbounded, self.td3.pre_tanh(states))
        self.last_metrics.update(sigma=float(sigma), warmup=int(warmup))
        return scale_action(self._raw)

    def observe(self, states, rewards, next_states, t, iters):
        # Last transition uses the actual next observation, never a reset observation.
        self.buffer.add_batch(states, self._raw, rewards, next_states)
        self.env_steps += 1
        losses = []
        diagnostics = []
        # The warmup_iters-th stored step still only fills the replay.
        # Actor actions and TD3 updates begin on the following step.
        if self.env_steps > self.warmup_iters:
            for _ in range(self.updates_per_iter):
                loss = self.td3.train_td3(self.buffer, self.batch_size)
                if loss is not None:
                    losses.append(loss)
                    diagnostics.append(self.td3.last_train_metrics.copy())
        self.last_metrics["critic_loss"] = float(np.mean(losses)) if losses else float("nan")
        for key in self.td3.last_train_metrics:
            values = [row[key] for row in diagnostics if np.isfinite(row[key])]
            self.last_metrics[key] = float(np.mean(values)) if values else float("nan")


class OnlineTD3Controller(FrozenTD3Controller):
    """One scenario of TD3 training. The first five PSO steps only fill the replay.

    Weights restart from the supplied initial network. The source network is never updated.
    Target-policy noise uses a private torch seed and restores the global generator.
    """
    name = "online_td3"
    collect_steps = 5

    def __init__(self, source, rng, updates_per_iter, batch_size, expl_noise,
                 expl_decay_iters, torch_seed, replay_size):
        self.template = copy.deepcopy(source)
        self.rng = rng
        self.updates_per_iter = updates_per_iter
        self.batch_size = batch_size
        self.expl_noise = tuple(expl_noise)
        self.expl_decay_iters = expl_decay_iters
        self.torch_seed = int(torch_seed)
        self.replay_size = replay_size
        self.td3 = self.template
        self.begin_episode()

    def begin_episode(self):
        self.td3 = copy.deepcopy(self.template)
        self.buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, self.replay_size,
                                    np.random.default_rng(self.torch_seed))
        self.env_steps = 0
        self.update_index = 0
        self.policy_log = []
        self.last_metrics = {}
        self._raw = None

    def act(self, states, t, iters):
        policy = self.td3.act(states)
        self.policy_log.append(scale_action(policy).astype(np.float32))
        sigma = self.expl_noise[1] + (self.expl_noise[0] - self.expl_noise[1]) * np.exp(
            -self.env_steps / self.expl_decay_iters)
        unbounded = policy + self.rng.normal(0, sigma, policy.shape)
        self._raw = np.clip(unbounded, -1, 1)
        self.last_metrics = policy_metrics(policy, unbounded, self.td3.pre_tanh(states))
        self.last_metrics.update(sigma=float(sigma), warmup=int(self.env_steps < self.collect_steps))
        return scale_action(self._raw)

    def observe(self, states, rewards, next_states, t, iters):
        self.buffer.add_batch(states, self._raw, rewards, next_states)
        self.env_steps += 1
        losses = []
        diagnostics = []
        if self.env_steps <= self.collect_steps:
            self.last_metrics["critic_loss"] = float("nan")
            for key in self.td3.last_train_metrics:
                self.last_metrics[key] = float("nan")
            return
        for _ in range(self.updates_per_iter):
            torch_state = torch.get_rng_state()
            torch.manual_seed(self.torch_seed + self.update_index)
            try:
                loss = self.td3.train_td3(self.buffer, self.batch_size)
            finally:
                torch.set_rng_state(torch_state)
            self.update_index += 1
            if loss is not None:
                losses.append(loss)
                diagnostics.append(self.td3.last_train_metrics.copy())
        self.last_metrics["critic_loss"] = float(np.mean(losses)) if losses else float("nan")
        for key in self.td3.last_train_metrics:
            values = [row[key] for row in diagnostics if np.isfinite(row[key])]
            self.last_metrics[key] = float(np.mean(values)) if values else float("nan")


def run_pso(obstacles, controller, n_particles, n_points, iters, pop_seed=None, pso_seed=None,
            v_max=V_MAX, vel_noise=0.0, log_states=False, verbose=False,
            initial_pos=None, step_callback=None, reward_pbest=REWARD_PBEST,
            start=None, goal=None):
    """Continuing rollout truncated at iters; no terminal transition.

    Population RNG defaults to system entropy. pop_seed is legacy/debug only;
    multi-episode training never sets it. initial_pos enables paired evaluation.
    """
    start = np.asarray(START if start is None else start, dtype=np.float64)
    goal = np.asarray(GOAL if goal is None else goal, dtype=np.float64)
    if not np.isfinite(v_max) or v_max <= 0 or iters <= 0:
        raise ValueError("v_max and iters must be positive")
    if hasattr(controller, "begin_episode"):
        controller.begin_episode()
    if initial_pos is None:
        rng = None if pop_seed is None else np.random.RandomState(pop_seed)
        pos = init_population(n_particles, n_points, rng, start=start, goal=goal)
    else:
        pos = np.asarray(initial_pos, dtype=np.float64).copy()
        if (pos.shape != (n_particles, n_points, 2) or not np.isfinite(pos).all()
                or np.any(pos < MAP_MIN) or np.any(pos > MAP_MAX)
                or not np.allclose(pos[:, 0], start) or not np.allclose(pos[:, -1], goal)):
            raise ValueError("Invalid initial population")
    initial = pos.copy()
    rng = np.random.default_rng(pso_seed)
    vel = np.zeros_like(pos)
    f, collide, depth = evaluate_paths(pos, obstacles)
    h_ref = depth_scale(obstacles)
    ever_feasible = ~collide
    first_feasible_iteration = 0 if ever_feasible.any() else -1
    pbest, pbest_f, pbest_collide = pos.copy(), f.copy(), collide.copy()
    g = int(np.argmin(f))
    gbest, gbest_f, gbest_collide = pos[g].copy(), float(f[g]), bool(collide[g])
    best_iteration = 0
    feasible_f, feasible_pos, feasible_iteration = None, None, -1
    if ever_feasible.any():
        feasible_index = int(np.argmin(np.where(ever_feasible, f, np.inf)))
        feasible_f, feasible_pos, feasible_iteration = float(f[feasible_index]), pos[feasible_index].copy(), 0
    states = build_states(f, pbest_f, gbest_f, pos, pbest, gbest, vel, vel, depth, collide, h_ref, v_max,
                          start, goal)
    curve, rows = [gbest_f], []
    feasible_curve = [feasible_f if feasible_f is not None else np.inf]
    params_log = np.empty((iters, n_particles, ACTION_DIM), dtype=np.float32)
    states_log = np.empty((iters, n_particles, STATE_DIM), dtype=np.float32) if log_states else None
    for t in range(iters):
        if hasattr(controller, "bind"):
            controller.bind(pos, vel, pbest, gbest, obstacles, start, goal, v_max)
        params = controller.act(states, t, iters)
        params_log[t] = params
        if log_states:
            states_log[t] = states
        old_vel = vel.copy()
        pos, vel = pso_step(pos, vel, pbest, gbest, params, rng, v_max, start, goal,
                            MAP_MIN, MAP_MAX, vel_noise)
        new_f, new_collide, new_depth = evaluate_paths(pos, obstacles)
        # Independent feasible archive: preserve valid paths without changing PSO attractors.
        if (~new_collide).any():
            feasible_index = int(np.argmin(np.where(~new_collide, new_f, np.inf)))
            if feasible_f is None or new_f[feasible_index] < feasible_f:
                feasible_f = float(new_f[feasible_index])
                feasible_pos, feasible_iteration = pos[feasible_index].copy(), t + 1
        old_pbest_f, old_gbest_f = pbest_f.copy(), gbest_f
        improved = new_f < pbest_f
        pbest[improved], pbest_f[improved] = pos[improved], new_f[improved]
        pbest_collide[improved] = new_collide[improved]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
            gbest_collide, best_iteration = bool(pbest_collide[g]), t + 1
        rewards, components = compute_rewards(f, new_f, old_pbest_f, pbest_f, old_gbest_f,
                                               depth, new_depth, ever_feasible, new_collide,
                                               h_ref, return_components=True, pbest_weight=reward_pbest)
        ever_feasible |= ~new_collide
        if first_feasible_iteration == -1 and ever_feasible.any():
            first_feasible_iteration = t + 1
        f, collide, depth = new_f, new_collide, new_depth
        next_states = build_states(f, pbest_f, gbest_f, pos, pbest, gbest, vel, old_vel,
                                   depth, collide, h_ref, v_max, start, goal)
        controller.observe(states, rewards, next_states, t, iters)
        states = next_states
        curve.append(gbest_f)
        feasible_curve.append(feasible_f if feasible_f is not None else np.inf)
        row = dict(iteration=t + 1, best_f=gbest_f, best_iteration=best_iteration,
                   best_collide=int(gbest_collide), reward_mean=float(rewards.mean()),
                   collision_ratio=float(collide.mean()), depth_mean=float(depth.mean()))
        row.update(best_feasible_f=feasible_f, best_feasible_iteration=feasible_iteration)
        row.update({"reward_" + name: float(components[:, j].mean()) for j, name in enumerate(REWARD_NAMES)})
        row.update({"reward_" + name + "_abs_mean": float(np.abs(components[:, j]).mean())
                    for j, name in enumerate(REWARD_NAMES)})
        row.update({"reward_" + name + "_nonzero_ratio": float((np.abs(components[:, j]) > 1e-12).mean())
                    for j, name in enumerate(REWARD_NAMES)})
        row.update({"action_" + name: float(params[:, j].mean()) for j, name in enumerate(PARAM_NAMES)})
        row.update(getattr(controller, "last_metrics", {}))
        rows.append(row)
        if step_callback is not None:
            step_callback(row)
        if verbose and ((t + 1) % 25 == 0 or t + 1 == iters):
            print("[{}] iter {} | best {:.4f} @ {}".format(controller.name, t + 1, gbest_f, best_iteration))
    return dict(best_pos=gbest, best_f=gbest_f, collide=gbest_collide,
                best_iteration=best_iteration, first_feasible_iteration=first_feasible_iteration,
                curve=np.asarray(curve), params=params_log, states=states_log,
                final_states=states, initial_pos=initial, step_metrics=rows,
                best_feasible_f=feasible_f, best_feasible_pos=feasible_pos,
                best_feasible_iteration=feasible_iteration, feasible_curve=np.asarray(feasible_curve))
