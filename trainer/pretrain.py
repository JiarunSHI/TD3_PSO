"""Multi-episode simulator training and independent frozen-policy evaluation."""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from config import (NUM_PARTICLES, NUM_POINTS, V_MAX, FIXED_PARAMS, LDIW_RANGE,
                    START, GOAL, MAP_MIN, MAP_MAX, PARAM_LOW, PARAM_HIGH, GAMMA, REWARD_PBEST,
                    ENDPOINT_CHOICES)
from drl.td3 import TD3
from drl.replay_buffer import ReplayBuffer
from env.fixed_env import MapEnv
from env.fitness import PENETRATION_WEIGHT
from pso.particle import init_population
from trainer.train import (STATE_DIM, STATE_NAMES, STATE_SCHEMA, TD3Controller,
                           FrozenTD3Controller, FixedController, LinearWController,
                           RandomController, OnlineTD3Controller, run_pso)

TD3_SATURATION_METHODS = ("td3", "initial_actor", "online_td3")
FITNESS_MILESTONES = (10, 20, 30, 40, 50, 100)


def environment_spec():
    return dict(state_schema=STATE_SCHEMA, state_names=list(STATE_NAMES),
                fitness="0.5*length+0.2*(sum_turn+max_turn)+100*boundary_squared+40*H",
                penetration_weight=PENETRATION_WEIGHT, start=START.tolist(), goal=GOAL.tolist(),
                bounds=[MAP_MIN, MAP_MAX], action_low=list(PARAM_LOW), action_high=list(PARAM_HIGH),
                endpoints=[{"name": name, "start": [float(s[0]), float(s[1])],
                            "goal": [float(g[0]), float(g[1])]} for name, s, g in ENDPOINT_CHOICES])


class CsvLog:
    def __init__(self, path):
        self.path, self.file, self.writer = Path(path), None, None

    def write(self, row):
        if self.file is None:
            self.file = self.path.open("w", newline="", encoding="utf-8-sig")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def endpoint_points():
    points = []
    for _, start, goal in ENDPOINT_CHOICES:
        points.extend((start, goal))
    return points


def endpoint_order(map_seed, trials, seed):
    """Each block of four trials is one permutation of the four endpoint choices."""
    rng = np.random.default_rng(int(np.random.SeedSequence([205, seed, map_seed]).generate_state(1)[0]))
    order = []
    while len(order) < trials:
        order.extend(int(index) for index in rng.permutation(len(ENDPOINT_CHOICES)))
    return order[:trials]


def map_for_seed(seed, endpoints=None):
    return np.asarray(MapEnv(seed).generate_mixed(rng=np.random.RandomState(seed), endpoints=endpoints))


def create_cases(directory, map_seeds, trials, particles, points, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    protected = endpoint_points()
    for map_seed in map_seeds:
        obstacles = map_for_seed(map_seed, protected)
        for trial, index in enumerate(endpoint_order(map_seed, trials, seed)):
            name, start, goal = ENDPOINT_CHOICES[index]
            # Do not seed population draws. Save actual arrays for paired comparisons.
            initial = init_population(particles, points, start=start, goal=goal)
            pso_seed = int(np.random.SeedSequence([201, seed, map_seed, trial]).generate_state(1)[0])
            path = directory / ("map{}_trial{}.npz".format(map_seed, trial))
            np.savez_compressed(str(path), obstacles=obstacles, initial_pos=initial,
                                map_seed=map_seed, trial=trial, pso_seed=pso_seed,
                                start=np.asarray(start, dtype=np.float64),
                                goal=np.asarray(goal, dtype=np.float64), endpoint=name)
            paths.append(path)
    return paths


def validation_cases(args, directory):
    if not args.validation_cases:
        return create_cases(directory, args.eval_map_seeds, args.eval_trials,
                            args.particles, args.points, args.seed)
    paths = sorted(Path(args.validation_cases).glob("*.npz"))
    if not paths:
        raise ValueError("No saved validation cases found")
    seen, maps = set(), set()
    for path in paths:
        with np.load(str(path), allow_pickle=False) as data:
            map_seed, trial = int(data["map_seed"]), int(data["trial"])
            if map_seed in args.train_map_seeds or map_seed not in args.eval_map_seeds:
                raise ValueError("Saved validation map not in the configured held-out set")
            if (map_seed, trial) in seen:
                raise ValueError("Duplicate validation case")
            seen.add((map_seed, trial))
            maps.add(map_seed)
            if "start" not in data.files or "goal" not in data.files or "endpoint" not in data.files:
                raise ValueError("Saved validation case has no start/goal; regenerate it for the four endpoint choices")
            if data["initial_pos"].shape != (args.particles, args.points, 2):
                raise ValueError("Saved validation population shape differs from configuration")
            if not np.allclose(data["initial_pos"][:, 0], data["start"]) or not np.allclose(data["initial_pos"][:, -1], data["goal"]):
                raise ValueError("Saved population endpoints do not match its start/goal")
    if maps != set(args.eval_map_seeds) or any(sum(m == seed for m, _ in seen) != args.eval_trials for seed in maps):
        raise ValueError("Saved validation case counts differ from configured maps/trials")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        target = directory / path.name
        shutil.copy2(str(path), str(target))
        copied.append(target)
    return copied


def online_from(source):
    read = source.get if isinstance(source, dict) else lambda key: getattr(source, key)
    return dict(updates_per_iter=read("updates_per_iter"), batch_size=read("batch_size"),
                expl_noise=read("expl_noise"), expl_decay_iters=read("expl_decay_iters"),
                replay_size=read("replay_size"), actor_lr=read("actor_lr"), critic_lr=read("critic_lr"))


def fresh_online_model(reference, init_seed, config):
    """Initialize all six networks and both optimizers without copying checkpoint weights.

    TD3 initializes targets from their corresponding new online networks. Preserve
    global RNGs so constructing this baseline cannot affect other methods.
    """
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        model = TD3(reference.device, reference.actor.net[0].in_features,
                    action_dim=reference.actor.net[-2].out_features,
                    hidden=reference.actor.net[0].out_features, gamma=reference.gamma,
                    tau=reference.tau, policy_noise=reference.policy_noise,
                    noise_clip=reference.noise_clip, policy_delay=reference.policy_delay,
                    actor_lr=float(config.get("actor_lr", reference.opt_a.param_groups[0]["lr"])),
                    critic_lr=float(config.get("critic_lr", reference.opt_c.param_groups[0]["lr"])),
                    seed=init_seed)
    model.reward_pbest = getattr(reference, "reward_pbest", REWARD_PBEST)
    return model


def evaluate(td3, cases, iters, v_max, seed=0, log=None, episode=0, artifact_dir=None,
             initial_td3=None, online=None, online_random_init=False):
    """Paired per-scenario evaluation. No cross-scenario fitness average is computed.

    Intermediate budgets reuse rollout prefixes.
    """
    rows, scenes = [], []
    milestones = sorted({t for t in FITNESS_MILESTONES + (iters,) if 0 < t <= iters})
    for path in cases:
        with np.load(str(path), allow_pickle=False) as data:
            if "start" not in data.files or "goal" not in data.files:
                raise ValueError("Evaluation case {} has no start/goal".format(path))
            scenes.append((data["obstacles"].copy(), data["initial_pos"].copy(),
                           int(data["map_seed"]), int(data["trial"]), int(data["pso_seed"]),
                           data["start"].astype(np.float64).copy(), data["goal"].astype(np.float64).copy(),
                           str(data["endpoint"])))

    def saturation_at(result, method, step):
        if method in TD3_SATURATION_METHODS:
            return float(result["step_metrics"][step - 1]["actor_boundary_ratio"])
        return None

    def run_case(case_index, scene, method, controller):
        obstacles, initial, map_seed, trial, pso_seed, start, goal, endpoint = scene
        result = run_pso(obstacles, controller, len(initial), initial.shape[1], iters,
                         pso_seed=pso_seed, v_max=v_max, initial_pos=initial,
                         reward_pbest=getattr(td3, "reward_pbest", REWARD_PBEST),
                         start=start, goal=goal)
        row = dict(episode=episode, map_seed=map_seed, trial=trial, endpoint=endpoint,
                   method=method, iterations=iters, best_f=float(result["best_f"]),
                   best_iteration=int(result["best_iteration"]), best_collide=int(result["collide"]),
                   first_feasible_iteration=int(result["first_feasible_iteration"]),
                   best_feasible_f=None if result["best_feasible_f"] is None else float(result["best_feasible_f"]),
                   best_feasible_iteration=int(result["best_feasible_iteration"]),
                   online_init_mode=getattr(controller, "init_mode", None),
                   online_init_seed=getattr(controller, "init_seed", None))
        for step in milestones:
            row["best_f_at_{}".format(step)] = float(result["curve"][step])
            value = result["feasible_curve"][step]
            row["best_feasible_f_at_{}".format(step)] = float(value) if np.isfinite(value) else None
            rate = saturation_at(result, method, step)
            row["actor_boundary_at_{}".format(step)] = None if rate is None else float(rate)
        # A finite observed return for this scenario, not an average across scenarios.
        row["discounted_reward_prefix"] = float(np.dot(td3.gamma ** np.arange(iters),
                                                       [step["reward_mean"] for step in result["step_metrics"]]))
        rows.append(row)
        if log is not None:
            log.write(row)
        if artifact_dir is not None:
            folder = Path(artifact_dir)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(folder / ("case{}_{}.npz".format(case_index, method))),
                                curve=result["curve"], best_pos=result["best_pos"],
                                feasible_curve=result["feasible_curve"],
                                best_feasible_pos=result["best_feasible_pos"]
                                if result["best_feasible_pos"] is not None else np.empty((0, 2)))
        return result

    for case_index, scene in enumerate(scenes):
        controllers = {"td3": FrozenTD3Controller(td3)}
        if initial_td3 is not None:
            controllers["initial_actor"] = FrozenTD3Controller(initial_td3)
            cfg = online or {}
            torch_seed = int(np.random.SeedSequence([203, seed, scene[2], scene[3]]).generate_state(1)[0] % (2 ** 31))
            init_seed = None
            online_source = initial_td3
            if online_random_init:
                init_seed = int(np.random.SeedSequence([204, seed, scene[2], scene[3]]).generate_state(1)[0])
                online_source = fresh_online_model(td3, init_seed, cfg)
            controllers["online_td3"] = OnlineTD3Controller(
                online_source, np.random.default_rng([203, seed, scene[2], scene[3]]),
                int(cfg.get("updates_per_iter", 4)), int(cfg.get("batch_size", 256)),
                tuple(cfg.get("expl_noise", (0.2, 0.05))), float(cfg.get("expl_decay_iters", 2000.0)),
                torch_seed, int(cfg.get("replay_size", 200000)))
            controllers["online_td3"].init_mode = "fresh_random" if online_random_init else "training_initial_snapshot"
            controllers["online_td3"].init_seed = init_seed
        controllers.update(
            fixed=FixedController(*FIXED_PARAMS),
            fixed_midpoint=FixedController(*((np.asarray(PARAM_LOW) + np.asarray(PARAM_HIGH)) / 2)),
            ldiw=LinearWController(*LDIW_RANGE, *FIXED_PARAMS[1:]),
            random=RandomController(np.random.default_rng([202, seed, scene[2], scene[3]])))
        for method, controller in controllers.items():
            run_case(case_index, scene, method, controller)
    td3_values = [row["best_f"] for row in rows if row["method"] == "td3"]
    if not td3_values:
        raise RuntimeError("Evaluation produced no TD3 scenarios")
    return dict(scenarios=rows, methods=list(dict.fromkeys(row["method"] for row in rows)),
                selection_median_best_f=float(np.median(td3_values)))


def save_checkpoint(path, td3, args, episode, env_steps, score, initial_actor=None,
                    train_map_seeds_used=None):
    # Inference checkpoints. A fresh training run deliberately starts new replay/optimizers.
    # validation_score is the median of per-scenario TD3 fitness, used only to pick a checkpoint.
    checkpoint = dict(spec=environment_spec(), args=vars(args), episode=episode,
                      env_steps=env_steps, gradient_steps=td3.total_it, validation_score=score,
                      actor=td3.actor.state_dict(), critic1=td3.c1.state_dict(), critic2=td3.c2.state_dict(),
                      initial_actor=initial_actor, train_map_seeds_used=list(train_map_seeds_used or []))
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, str(temporary))
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    if checkpoint.get("spec") != environment_spec():
        raise ValueError("Checkpoint state/fitness/action schema is incompatible; retrain this model")
    saved_args = checkpoint["args"]
    td3 = TD3(torch.device(device), STATE_DIM, hidden=128, gamma=saved_args["gamma"],
              tau=saved_args.get("tau", 0.005), actor_lr=saved_args.get("actor_lr", 1e-4),
              critic_lr=saved_args.get("critic_lr", 3e-4))
    td3.reward_pbest = checkpoint["args"].get("reward_pbest", REWARD_PBEST)
    td3.actor.load_state_dict(checkpoint["actor"])
    td3.c1.load_state_dict(checkpoint["critic1"])
    td3.c2.load_state_dict(checkpoint["critic2"])
    for net, target in ((td3.actor, td3.actor_t), (td3.c1, td3.c1_t), (td3.c2, td3.c2_t)):
        target.load_state_dict(net.state_dict())
        net.eval()
        target.eval()
    return td3, checkpoint


def initial_reference(td3, checkpoint):
    if checkpoint.get("initial_actor") is None:
        raise ValueError("Checkpoint lacks the initial actor reference")
    initial = copy.deepcopy(td3)
    initial.actor.load_state_dict(checkpoint["initial_actor"])
    initial.actor_t.load_state_dict(checkpoint["initial_actor"])
    return initial


def train_parser():
    p = argparse.ArgumentParser(description="Train TD3 across truncated PSO episodes; no deployment learning")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--train-iters", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-after-episodes", type=int, default=10,
                   help="Do not validate during the first N episodes, including the final-episode check")
    p.add_argument("--eval-trials", type=int, default=4)
    p.add_argument("--train-map-seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--eval-map-seeds", type=int, nargs="+", default=list(range(1000, 1005)))
    p.add_argument("--test-map-seeds", type=int, nargs="+", default=list(range(3000, 3020)),
                   help="Held out of training; default test uses this range")
    p.add_argument("--validation-cases", help="Reuse saved validation arrays, e.g. the previous run's validation_cases")
    p.add_argument("--particles", type=int, default=NUM_PARTICLES)
    p.add_argument("--points", type=int, default=NUM_POINTS)
    p.add_argument("--v-max", type=float, default=V_MAX)
    p.add_argument("--seed", type=int, default=0, help="Networks, PSO, replay, exploration; NOT populations")
    p.add_argument("--warmup-iters", type=int, default=20, help="Global PSO steps, applied only once")
    p.add_argument("--updates-per-iter", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-size", type=int, default=200000)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--reward-pbest", type=float, default=REWARD_PBEST)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--expl-noise", type=float, nargs=2, default=[0.2, 0.05])
    p.add_argument("--expl-decay-iters", type=float, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="results/pretrain_100_long_return")
    p.add_argument("--dry-run", action="store_true", help="Validate configuration only; no training or files")
    return p


def validation_due(episode, eval_every, episodes, warmup_iters, train_iters, eval_after_episodes=10):
    """Skip the initial episode window and fully random collection episodes."""
    warmup_episodes = warmup_iters // train_iters
    return episode > max(warmup_episodes, eval_after_episodes) and (episode % eval_every == 0 or episode == episodes)


def validate_args(args):
    for key in ("episodes", "train_iters", "eval_iters", "eval_every", "eval_trials", "particles",
                "batch_size", "replay_size", "threads", "updates_per_iter"):
        if getattr(args, key) <= 0:
            raise ValueError(key + " must be positive")
    if args.points < 3 or args.warmup_iters < 0:
        raise ValueError("points >= 3 and warmup_iters >= 0 required")
    if args.eval_after_episodes < 0:
        raise ValueError("eval_after_episodes must be nonnegative")
    if args.replay_size < max(args.batch_size, args.particles):
        raise ValueError("replay_size must fit at least one batch and one population")
    if set(args.train_map_seeds) & set(args.eval_map_seeds):
        raise ValueError("Training and validation map sets must be disjoint")
    if set(args.eval_map_seeds) & set(args.test_map_seeds):
        raise ValueError("Validation and reserved test map sets must be disjoint")
    if min([args.seed] + args.train_map_seeds + args.eval_map_seeds + args.test_map_seeds) < 0:
        raise ValueError("Seeds must be nonnegative")
    if not (0 < args.gamma < 1 and 0 < args.tau <= 1):
        raise ValueError("Continuing task requires 0 < gamma < 1; 0 < tau <= 1")
    for value in (args.actor_lr, args.critic_lr, args.v_max, args.expl_decay_iters):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Learning rates, v_max and noise decay must be finite and positive")
    if not np.isfinite(args.expl_noise).all() or min(args.expl_noise) < 0:
        raise ValueError("Exploration standard deviations must be finite and nonnegative")
    if not np.isfinite(args.reward_pbest) or args.reward_pbest < 0:
        raise ValueError("reward_pbest must be finite and nonnegative")


def train(args):
    validate_args(args)
    if args.dry_run:
        print(json.dumps(dict(configuration=vars(args), spec=environment_spec(),
                              pso_steps=args.episodes * args.train_iters,
                              transitions=args.episodes * args.train_iters * args.particles), indent=2))
        return
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; use a new --out to preserve previous results")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    td3 = TD3(torch.device(args.device), STATE_DIM, seed=args.seed,
              actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma, tau=args.tau)
    td3.reward_pbest = args.reward_pbest
    initial_td3 = copy.deepcopy(td3)
    buffer = ReplayBuffer(STATE_DIM, 3, args.replay_size, np.random.default_rng([102, args.seed]))
    controller = TD3Controller(td3, buffer, np.random.default_rng([101, args.seed]),
                               args.warmup_iters, args.updates_per_iter, args.batch_size,
                               args.expl_noise, args.expl_decay_iters)
    schedule_rng = np.random.default_rng([103, args.seed])
    write_json(out / "config.json", dict(args=vars(args), spec=environment_spec(),
                                        population_randomness="new entropy-backed RNG per episode",
                                        truncation_bootstrap=True))
    cases = validation_cases(args, out / "validation_cases")
    episodes_log, steps_log, eval_log = (CsvLog(out / name) for name in
                                        ("episodes.csv", "steps.csv", "validation.csv"))
    initial_dir = out / "train_initials"
    initial_dir.mkdir()
    best_score, best_trained_score, latest_score = float("inf"), float("inf"), None
    initial_evaluated = False
    try:
        used_maps = []
        # Save the initial model without running any startup validation or online baseline.
        save_checkpoint(out / "initial.pt", initial_td3, args, 0, 0, None,
                        initial_td3.actor.state_dict(), used_maps)
        reserved = set(args.eval_map_seeds) | set(args.test_map_seeds)

        def draw_map_seed():
            for _ in range(100000):
                seed = int(schedule_rng.integers(0, 1_000_000))
                if seed not in reserved:
                    return seed
            raise RuntimeError("Could not draw a training map outside validation and test seeds")

        for episode in range(1, args.episodes + 1):
            map_seed = draw_map_seed()
            used_maps.append(map_seed)
            endpoint, start, goal = ENDPOINT_CHOICES[int(schedule_rng.integers(0, len(ENDPOINT_CHOICES)))]
            obstacles = map_for_seed(map_seed, (start, goal))
            pso_seed = int(np.random.SeedSequence([104, args.seed, episode]).generate_state(1)[0])
            initial = init_population(args.particles, args.points, start=start, goal=goal)
            np.savez_compressed(str(initial_dir / ("episode_{:04d}.npz".format(episode))),
                                initial_pos=initial, obstacles=obstacles, map_seed=map_seed, pso_seed=pso_seed,
                                start=np.asarray(start, dtype=np.float64),
                                goal=np.asarray(goal, dtype=np.float64), endpoint=endpoint)
            def log_step(row):
                steps_log.write(dict(episode=episode, global_step=controller.env_steps, **row))
            result = run_pso(obstacles, controller, args.particles, args.points, args.train_iters,
                             pso_seed=pso_seed, v_max=args.v_max, initial_pos=initial,
                             step_callback=log_step, reward_pbest=args.reward_pbest, start=start, goal=goal)
            started = controller.env_steps - args.train_iters
            if controller.env_steps <= args.warmup_iters:
                phase = "collect"
            elif started >= args.warmup_iters:
                phase = "train"
            else:
                phase = "mixed"
            row = dict(episode=episode, phase=phase, map_seed=map_seed, endpoint=endpoint, pso_seed=pso_seed,
                       global_steps=controller.env_steps, gradient_steps=td3.total_it,
                       replay_size=len(buffer), best_f=result["best_f"],
                       best_iteration=result["best_iteration"], best_collide=int(result["collide"]),
                       first_feasible_iteration=result["first_feasible_iteration"],
                       best_feasible_f=result["best_feasible_f"],
                       best_feasible_iteration=result["best_feasible_iteration"])
            # Population size is constant, so mean step ratios equal pooled action ratios.
            for key in result["step_metrics"][0]:
                if key.startswith(("actor_boundary", "exploration_clip", "logit_", "reward_", "action_", "td_")):
                    values = [step[key] for step in result["step_metrics"] if np.isfinite(step[key])]
                    row[key] = float(np.mean(values)) if values else float("nan")
            row["sigma_last"] = controller.last_metrics["sigma"]
            row["critic_loss_last"] = controller.last_metrics["critic_loss"]
            episodes_log.write(row)
            np.savez_compressed(str(initial_dir / ("result_{:04d}.npz".format(episode))),
                                best_pos=result["best_pos"], curve=result["curve"],
                                feasible_curve=result["feasible_curve"],
                                best_feasible_pos=result["best_feasible_pos"]
                                if result["best_feasible_pos"] is not None else np.empty((0, 2)))
            print("Episode {}/{} {} | {} | best {:.5f} @ {} | collision {}".format(
                episode, args.episodes, phase, endpoint, result["best_f"], result["best_iteration"],
                int(result["collide"])), flush=True)
            latest_score = None  # A score must describe this exact checkpoint, not an older actor.
            if validation_due(episode, args.eval_every, args.episodes, args.warmup_iters, args.train_iters,
                              args.eval_after_episodes):
                summary = evaluate(td3, cases, args.eval_iters, args.v_max, args.seed, eval_log, episode,
                                   initial_td3=initial_td3, online=online_from(args))
                if not initial_evaluated:
                    initial_rows = [row for row in summary["scenarios"] if row["method"] == "initial_actor"]
                    best_score = float(np.median([row["best_f"] for row in initial_rows]))
                    write_json(out / "validation_initial.json",
                               dict(evaluated_after_episode=episode, scenarios=initial_rows,
                                    selection_median_best_f=best_score))
                    for name in ("initial.pt", "best.pt"):
                        save_checkpoint(out / name, initial_td3, args, 0, 0, best_score,
                                        initial_td3.actor.state_dict(), used_maps)
                    initial_evaluated = True
                latest_score = summary["selection_median_best_f"]
                write_json(out / ("validation_{:04d}.json".format(episode)), summary)
                print("Validation episode {} | {} scenarios written | TD3 selection median {:.5f}".format(
                    episode, sum(row["method"] == "td3" for row in summary["scenarios"]), latest_score), flush=True)
                if latest_score < best_score:
                    best_score = latest_score
                    save_checkpoint(out / "best.pt", td3, args, episode, controller.env_steps, latest_score,
                                    initial_td3.actor.state_dict(), used_maps)
                if latest_score < best_trained_score:
                    best_trained_score = latest_score
                    save_checkpoint(out / "best_trained.pt", td3, args, episode, controller.env_steps, latest_score,
                                    initial_td3.actor.state_dict(), used_maps)
            save_checkpoint(out / "last.pt", td3, args, episode, controller.env_steps, latest_score,
                            initial_td3.actor.state_dict(), used_maps)
        print("Saved inference checkpoints and logs to {}".format(out.resolve()))
        if not initial_evaluated:
            print("Validation skipped: initial.pt and last.pt saved; best checkpoints require a validation result.")
    finally:
        for log in (episodes_log, steps_log, eval_log):
            log.close()


def eval_main(argv=None):
    p = argparse.ArgumentParser(description="Frozen TD3 vs fixed/LDIW/random on paired short rollouts")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--map-seeds", nargs="+", type=int, default=list(range(3000, 3020)))
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--cases", help="Reuse a directory of previously saved evaluation cases")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="results/test_50")
    args = p.parse_args(argv)
    if min(args.iters, args.trials, args.threads) <= 0 or min([args.seed] + args.map_seeds) < 0:
        p.error("Invalid iteration/trial/thread counts or seed")
    torch.set_num_threads(args.threads)
    td3, checkpoint = load_checkpoint(args.checkpoint, args.device)
    initial_td3 = initial_reference(td3, checkpoint)
    if checkpoint.get("initial_actor") is None:
        p.error("Use a validated checkpoint (best.pt, best_trained.pt, or the final last.pt)")
    model_args = checkpoint["args"]
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be new or empty")
    if args.cases:
        cases = sorted(Path(args.cases).glob("*.npz"))
        if not cases:
            p.error("No evaluation cases found")
    else:
        used = checkpoint.get("train_map_seeds_used") or []
        overlap = set(args.map_seeds) & (set(model_args["eval_map_seeds"]) | set(used))
        if overlap:
            p.error("Test maps overlap validation or maps already used in training: {}".format(sorted(overlap)))
        outside = set(args.map_seeds) - set(model_args.get("test_map_seeds", args.map_seeds))
        if outside:
            p.error("Test maps are outside the reserved test range saved with the checkpoint: {}".format(sorted(outside)))
        cases = None
    out.mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = create_cases(out / "cases", args.map_seeds, args.trials,
                             model_args["particles"], model_args["points"], args.seed)
    log = CsvLog(out / "comparison.csv")
    try:
        summary = evaluate(td3, cases, args.iters, model_args["v_max"], args.seed, log,
                           artifact_dir=out / "curves", initial_td3=initial_td3,
                           online=online_from(model_args), online_random_init=True)
        write_json(out / "summary.json", summary)
        write_json(out / "config.json", dict(args=vars(args), model_episode=checkpoint["episode"],
                                            online_initialization="fresh_random_per_scenario",
                                            cases=[str(Path(c).resolve()) for c in cases]))
        print(json.dumps(summary, indent=2))
    finally:
        log.close()
