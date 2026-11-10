"""Regression tests; no multi-episode optimization run is launched by this suite."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from config import START, GOAL
from drl.td3 import TD3
from drl.replay_buffer import ReplayBuffer
from env.fitness import evaluate_paths, collision_metrics
from pso.particle import init_population
from trainer.train import (STATE_DIM, STATE_NAMES, build_states, compute_rewards,
                           FixedController, FrozenTD3Controller, TD3Controller,
                           RandomController, OnlineTD3Controller, run_pso, policy_metrics, scale_action)
from trainer.pretrain import (train_parser, train, map_for_seed, create_cases, evaluate,
                              save_checkpoint, load_checkpoint, validate_args, eval_main, validation_cases,
                              validation_due, fresh_online_model)

torch.set_num_threads(1)


class InitializationTests(unittest.TestCase):
    def test_rejection_preserves_accepted_candidates_without_clipping(self):
        rng = np.random.RandomState(0)
        expected, draws = [], 0
        while len(expected) < 150:
            candidate = np.linspace(START, GOAL, 10)
            noise = np.cumsum(rng.randn(8, 2), axis=0)
            candidate[1:-1] += 2.5 * noise / (np.std(noise) + 1e-8)
            draws += 1
            if np.all((candidate >= 0) & (candidate <= 20)):
                expected.append(candidate)
        actual = init_population(150, 10, np.random.RandomState(0))
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        self.assertGreater(draws, 150)
        self.assertEqual(actual.shape, (150, 10, 2))
        self.assertTrue(np.all((actual >= 0) & (actual <= 20)))
        self.assertEqual(len(np.unique(actual.reshape(150, -1), axis=0)), 150)

    def test_default_population_uses_entropy_not_global_seed(self):
        rng = np.random.default_rng(7)
        with patch("pso.particle.np.random.default_rng", return_value=rng) as factory:
            result = init_population(150, 10)
        factory.assert_called_once_with()
        self.assertEqual(result.shape, (150, 10, 2))
        np.random.seed(123)
        first = init_population(10, 10)
        np.random.seed(123)
        second = init_population(10, 10)
        self.assertFalse(np.array_equal(first, second))

    def test_invalid_bounds_and_exhaustion_are_explicit(self):
        with self.assertRaises(ValueError):
            init_population(2, 10, start=(-1, 0))
        with self.assertRaises(RuntimeError):
            init_population(150, 10, np.random.RandomState(0), max_attempts=1)

    def test_generated_maps_leave_fixed_endpoints_clear(self):
        for seed in (0, 42, 52, 58, 1000, 2000):
            obs = map_for_seed(seed)
            for endpoint in (START, GOAL):
                self.assertTrue(np.all(np.linalg.norm(obs[:, :2] - endpoint, axis=1) > obs[:, 2]))


class StateRewardTests(unittest.TestCase):
    def test_flattened_cosines_and_rms_distinguish_speed(self):
        pos = np.tile(np.linspace(START, GOAL, 4), (4, 1, 1))
        pbest, gbest = pos.copy(), pos[0].copy()
        pbest[:, 1:-1, 0] += 1
        gbest[1:-1, 1] += 1
        v, old = np.zeros_like(pos), np.zeros_like(pos)
        v[1:3, 1:-1, 0], v[3, 1:-1, 0] = 1, 2
        old[1, 1:-1, 0], old[2, 1:-1, 0], old[3, 1:-1, 0] = 1, -1, 1
        state = build_states(np.full(4, 20.), np.full(4, 18.), 15., pos, pbest, gbest,
                             v, old, np.zeros(4), np.zeros(4, dtype=bool), 1.)
        col = lambda name: state[:, STATE_NAMES.index(name)]
        self.assertEqual(state.shape, (4, 11))
        np.testing.assert_allclose(col("velocity_turn_cos"), [0, 1, -1, 1])
        np.testing.assert_allclose(col("velocity_pbest_cos"), [0, 1, 1, 1])
        np.testing.assert_allclose(col("velocity_gbest_cos"), [0, 0, 0, 0])
        np.testing.assert_allclose(col("speed_rms"), [-1, 1 / np.sqrt(2) - 1,
                                                     1 / np.sqrt(2) - 1, np.sqrt(2) - 1], atol=1e-7)
        self.assertTrue(np.all(np.abs(state) <= 1))
        self.assertNotIn("progress", STATE_NAMES)
        np.testing.assert_allclose(col("distance_pbest"), 2 / (20 * np.sqrt(2)) - 1, atol=1e-7)
        np.testing.assert_allclose(col("distance_gbest"), 2 / (20 * np.sqrt(2)) - 1, atol=1e-7)
        self.assertFalse(any("stagnation" in name or "diversity" in name for name in STATE_NAMES))

    def test_attraction_distance_separates_equal_direction_states(self):
        pos = np.tile(np.linspace(START, GOAL, 4), (2, 1, 1))
        pbest = pos.copy()
        pbest[0, 1:-1, 0] += 1
        pbest[1, 1:-1, 0] += 3
        vel = np.zeros_like(pos)
        vel[:, 1:-1, 0] = 1
        state = build_states(np.full(2, 20.), np.full(2, 18.), 15., pos, pbest, pos[0],
                             vel, vel, np.zeros(2), np.zeros(2, bool), 1.)
        self.assertEqual(state[0, STATE_NAMES.index("velocity_pbest_cos")],
                         state[1, STATE_NAMES.index("velocity_pbest_cos")])
        self.assertLess(state[0, STATE_NAMES.index("distance_pbest")],
                        state[1, STATE_NAMES.index("distance_pbest")])

    def test_fitness_has_exact_requested_terms(self):
        path = np.array([[[1., 1.], [4., 1.], [4., 5.]]])
        f, hit, depth = evaluate_paths(path, [(2., 1., .5)])
        self.assertAlmostEqual(float(f[0]), .5 * 7 + .2 * np.pi + 40 * .5)
        self.assertTrue(hit[0])
        self.assertEqual(depth[0], .5)
        # Deepest segment is counted once, even when multiple segments cross the obstacle.
        loop = np.array([[[1., 1.], [3., 1.], [1., 1.]]])
        self.assertEqual(collision_metrics(loop, [(2., 1., .5)])[0][0], .5)
        outside = np.array([[[-1., 1.], [1., 1.]]])
        self.assertEqual(evaluate_paths(outside, [])[0][0], 101.)

    def test_adjacent_rewards_retain_negative_values(self):
        a = lambda value: np.array([value])
        reward, parts = compute_rewards(a(20.), a(40.), a(10.), a(10.), 5., a(.1), a(.4),
                                          a(True), a(True), 1., True)
        self.assertLess(parts[0, 1], 0)
        self.assertLess(parts[0, 3], 0)
        self.assertLess(reward[0], 0)

    def test_first_feasible_is_once_and_tangency_is_not_escape(self):
        path = np.array([[[1., 2.], [3., 2.]]])
        depth, hit = collision_metrics(path, [(2., 1., 1.)])
        self.assertEqual(depth[0], 0.)
        self.assertTrue(hit[0])
        f, zero = np.array([10.]), np.zeros(1)
        _, tangent = compute_rewards(f, f, f, f, 5., zero, zero, np.array([False]), hit, 1., True)
        self.assertEqual(tangent[0, 4], 0)
        _, first = compute_rewards(f, f, f, f, 5., zero, zero, np.array([False]), ~hit, 1., True)
        _, again = compute_rewards(f, f, f, f, 5., zero, zero, np.array([True]), ~hit, 1., True)
        self.assertEqual(first[0, 4], 1)
        self.assertEqual(again[0, 4], 0)

    def test_rewards_are_not_capped(self):
        old, new = np.full(3, 100.), np.array([99., 90., 50.])
        r = compute_rewards(old, new, old, new, 40., np.zeros(3), np.zeros(3),
                             np.ones(3, bool), np.zeros(3, bool), 1.)
        self.assertTrue(np.all(np.diff(r) > 0))
        self.assertGreater(r[-1], 1)

    def test_saturation_metrics_separate_policy_and_exploration(self):
        policy = np.array([[.99, 0, -.99], [0., 0., 0.]])
        executed = np.array([[1.1, 0, -.99], [0., -1.1, 0.]])
        metrics = policy_metrics(policy, executed, np.arctanh(policy))
        self.assertAlmostEqual(metrics["actor_boundary_ratio"], 2 / 6)
        self.assertAlmostEqual(metrics["exploration_clip_ratio"], 2 / 6)
        self.assertEqual(metrics["actor_boundary_c1"], 0)
        np.testing.assert_allclose(scale_action([[-2, 0, 2]]), [[.4, 1.5, 2.]])

    def test_online_td3_updates_from_a_private_copy(self):
        source = TD3(torch.device("cpu"), STATE_DIM, seed=1)
        frozen = {key: value.clone() for key, value in source.actor.state_dict().items()}
        torch_state = torch.get_rng_state().clone()
        controller = OnlineTD3Controller(source, np.random.default_rng(0), 1, 4, (0.2, 0.05), 20, 3, 100)
        initial = init_population(4, 4, np.random.RandomState(0))
        run_pso([], controller, 4, 4, 6, pso_seed=1, initial_pos=initial)
        self.assertEqual(len(controller.buffer), 24)
        self.assertEqual(controller.td3.total_it, 1)
        self.assertEqual(source.total_it, 0)
        for key, value in frozen.items():
            self.assertTrue(torch.equal(value, source.actor.state_dict()[key]))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_fresh_online_model_resets_weights_targets_optimizers_and_rng(self):
        source = TD3(torch.device("cpu"), STATE_DIM, seed=1, gamma=.97, tau=.01)
        for network in (source.actor, source.c1, source.c2):
            for parameter in network.parameters():
                parameter.data.fill_(7.)
        source.total_it = 123
        state = torch.get_rng_state().clone()
        cfg = dict(actor_lr=2e-4, critic_lr=4e-4)
        first = fresh_online_model(source, 42, cfg)
        again = fresh_online_model(source, 42, cfg)
        different = fresh_online_model(source, 43, cfg)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(first.total_it, 0)
        self.assertEqual(len(first.opt_a.state), 0)
        self.assertEqual(len(first.opt_c.state), 0)
        self.assertEqual(first.opt_a.param_groups[0]["lr"], 2e-4)
        self.assertEqual(first.opt_c.param_groups[0]["lr"], 4e-4)
        self.assertEqual(first.gamma, .97)
        self.assertEqual(first.tau, .01)
        for name, target_name in (("actor", "actor_t"), ("c1", "c1_t"), ("c2", "c2_t")):
            first_values = getattr(first, name).state_dict()
            source_values = getattr(source, name).state_dict()
            same_values = getattr(again, name).state_dict()
            different_values = getattr(different, name).state_dict()
            target_values = getattr(first, target_name).state_dict()
            self.assertTrue(any(not torch.equal(value, source_values[key]) for key, value in first_values.items()))
            self.assertTrue(any(not torch.equal(value, different_values[key]) for key, value in first_values.items()))
            for key, value in first_values.items():
                self.assertTrue(torch.equal(value, same_values[key]))
                self.assertTrue(torch.equal(value, target_values[key]))
                self.assertTrue(torch.all(source_values[key] == 7.))

    def test_random_parameters_are_shared_within_a_step(self):
        controller = RandomController(np.random.default_rng(0))
        first = controller.act(np.zeros((5, 9)), 0, 10)
        second = controller.act(np.zeros((5, 9)), 1, 10)
        self.assertEqual(first.shape, (5, 3))
        np.testing.assert_array_equal(first, np.repeat(first[:1], 5, axis=0))
        self.assertTrue(np.all((first >= [.4, 1., 1.]) & (first <= [.9, 2., 2.])))
        self.assertFalse(np.array_equal(first[0], second[0]))


class IntegrationTests(unittest.TestCase):
    def test_cutoff_preserves_prefix_and_final_transition_bootstraps(self):
        # Fixed controller isolates environment cutoff semantics without launching training.
        initial = init_population(8, 6, np.random.RandomState(7))
        class Recorder(FixedController):
            def __init__(self):
                super().__init__(.7, 1.5, 1.5)
                self.transitions = []
            def observe(self, states, rewards, next_states, t, iters):
                self.transitions.append((states.copy(), next_states.copy()))
        short_ctrl, long_ctrl = Recorder(), Recorder()
        short = run_pso(map_for_seed(42), short_ctrl, 8, 6, 3, pso_seed=8, initial_pos=initial, log_states=True)
        long = run_pso(map_for_seed(42), long_ctrl, 8, 6, 5, pso_seed=8, initial_pos=initial, log_states=True)
        np.testing.assert_array_equal(short["curve"], long["curve"][:4])
        np.testing.assert_array_equal(short["final_states"], long["states"][3])
        np.testing.assert_array_equal(short_ctrl.transitions[-1][1], short["final_states"])
        self.assertEqual(short["best_iteration"], int(np.argmin(short["curve"])))

    def test_continuing_target_keeps_bootstrap(self):
        # One synthetic gradient step checks Bellman arithmetic, not a training experiment.
        td3 = TD3(torch.device("cpu"), STATE_DIM, seed=0)
        for net in (td3.c1, td3.c2, td3.c1_t, td3.c2_t):
            for param in net.parameters():
                param.data.zero_()
        td3.c1_t.net[-1].bias.data.fill_(2.)
        td3.c2_t.net[-1].bias.data.fill_(3.)
        buf = ReplayBuffer(STATE_DIM, 3, max_size=4)
        buf.add_batch(np.zeros((4, STATE_DIM)), np.zeros((4, 3)), np.zeros(4), np.zeros((4, STATE_DIM)))
        self.assertAlmostEqual(td3.train_td3(buf, 4), 2 * (td3.gamma * 2.) ** 2, places=5)
        self.assertAlmostEqual(td3.last_train_metrics["td_bootstrap_mean"], td3.gamma * 2., places=5)
        self.assertAlmostEqual(td3.last_train_metrics["td_target_mean"],
                               td3.last_train_metrics["td_reward_mean"] + td3.last_train_metrics["td_bootstrap_mean"], places=5)

    def test_feasible_archive_survives_lower_cost_colliding_path(self):
        initial = init_population(2, 4, np.random.RandomState(0))
        evaluations = [(np.array([10., 11.]), np.array([False, True]), np.array([0., .1])),
                       (np.array([9., 12.]), np.array([True, False]), np.array([.1, 0.]))]
        with patch("trainer.train.evaluate_paths", side_effect=evaluations):
            result = run_pso([], FixedController(.7, 1.5, 1.5), 2, 4, 1, initial_pos=initial, pso_seed=0)
        self.assertEqual(result["best_f"], 9.)
        self.assertTrue(result["collide"])
        self.assertEqual(result["best_feasible_f"], 10.)
        self.assertEqual(result["best_feasible_iteration"], 0)
        np.testing.assert_array_equal(result["best_feasible_pos"], initial[0])
        np.testing.assert_array_equal(result["feasible_curve"], [10., 10.])

    def test_global_warmup_and_replay_survive_episode_reset(self):
        td3 = TD3(torch.device("cpu"), STATE_DIM, seed=0)
        buffer = ReplayBuffer(STATE_DIM, 3, max_size=100)
        ctrl = TD3Controller(td3, buffer, np.random.default_rng(3), 2, 1, 4, [.2, .05], 20)
        with patch.object(td3, "train_td3", return_value=None) as update:
            first = run_pso([], ctrl, 4, 4, 2, pso_seed=1)
            self.assertEqual(update.call_count, 0)
            second = run_pso([], ctrl, 4, 4, 2, pso_seed=1)
        self.assertEqual(ctrl.env_steps, 4)
        self.assertEqual(len(buffer), 16)
        self.assertEqual(update.call_count, 2)
        self.assertEqual(first["step_metrics"][0]["warmup"], 1)
        self.assertEqual(second["step_metrics"][0]["warmup"], 0)
        self.assertAlmostEqual(second["step_metrics"][0]["sigma"], .2)
        np.testing.assert_array_equal(buffer.s_[4:8], first["final_states"])
        self.assertFalse(np.array_equal(buffer.s_[4:8], buffer.s[8:12]))

    def test_frozen_evaluation_does_not_touch_model_or_random_stream(self):
        td3 = TD3(torch.device("cpu"), STATE_DIM, seed=0)
        weights = {key: value.clone() for key, value in td3.actor.state_dict().items()}
        torch_state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            cases = create_cases(tmp, [1000], 1, 4, 4, 0)
            with patch.object(td3, "train_td3", side_effect=AssertionError("evaluation updated model")):
                a = evaluate(td3, cases, 2, 2.)
                b = evaluate(td3, cases, 2, 2.)
            self.assertEqual(a, b)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(td3.total_it, 0)
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, td3.actor.state_dict()[key]))

    def test_training_orchestration_with_optimizer_disabled(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = train_parser().parse_args(["--episodes", "2", "--train-iters", "2", "--eval-iters", "2",
                "--eval-every", "1", "--eval-after-episodes", "0", "--eval-trials", "1", "--particles", "4", "--points", "4",
                "--warmup-iters", "1", "--updates-per-iter", "1", "--batch-size", "4",
                "--train-map-seeds", "0", "--eval-map-seeds", "1000", "--out", tmp])
            with patch.object(TD3, "train_td3", return_value=None) as update:
                train(args)
            self.assertEqual(update.call_count, 3)  # Online eval is only 2 steps, still inside its 5-step replay collection.
            with (Path(tmp) / "episodes.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(r["replay_size"]) for r in rows], [8, 16])
            self.assertEqual([int(r["global_steps"]) for r in rows], [2, 4])
            self.assertIn("actor_boundary_ratio", rows[0])
            self.assertIn("td_bootstrap_mean", rows[0])
            self.assertIn("reward_gbest_nonzero_ratio", rows[0])
            self.assertIn("best_feasible_f", rows[0])
            model, checkpoint = load_checkpoint(Path(tmp) / "best.pt", "cpu")
            self.assertEqual(checkpoint["spec"]["state_names"], list(STATE_NAMES))
            self.assertEqual(checkpoint["episode"], 0)  # Disabled updates cannot improve identical validation rollouts.
            self.assertAlmostEqual(model.gamma, .97)
            self.assertIsNotNone(checkpoint["initial_actor"])
            self.assertTrue((Path(tmp) / "last.pt").exists())
            self.assertTrue((Path(tmp) / "initial.pt").exists())
            self.assertTrue((Path(tmp) / "best_trained.pt").exists())
            # Exercise the standalone test CLI with frozen random weights.
            eval_main(["--checkpoint", str(Path(tmp) / "best.pt"), "--iters", "2", "--trials", "1",
                       "--map-seeds", "3000", "--out", str(Path(tmp) / "test")])
            self.assertTrue((Path(tmp) / "test" / "comparison.csv").exists())
            summary = json.loads((Path(tmp) / "test" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("online_td3", summary["methods"])
            self.assertIn("initial_actor", summary["methods"])
            self.assertIn("fixed_midpoint", summary["methods"])
            self.assertNotIn("fixed_actor_mean", summary["methods"])
            self.assertIn("selection_median_best_f", summary)
            self.assertNotIn("mean_best_f", json.dumps(summary))
            with (Path(tmp) / "test" / "comparison.csv").open(encoding="utf-8-sig") as f:
                compared = list(csv.DictReader(f))
            td3_rows = [row for row in compared if row["method"] == "td3"]
            self.assertTrue(td3_rows)
            self.assertIn("best_f_at_2", td3_rows[0])
            self.assertTrue(td3_rows[0]["actor_boundary_at_2"])
            online_rows = [row for row in compared if row["method"] == "online_td3"]
            self.assertTrue(online_rows)
            self.assertEqual(online_rows[0]["online_init_mode"], "fresh_random")
            self.assertTrue(online_rows[0]["online_init_seed"])
            self.assertEqual(td3_rows[0]["online_init_seed"], "")
            with self.assertRaises(ValueError):
                train(args)  # existing output is protected

    def test_old_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp, contextlib.redirect_stdout(io.StringIO()):
            old = TD3(torch.device("cpu"), 23, seed=0)
            old.save_model(str(Path(tmp) / "old"))
            new = TD3(torch.device("cpu"), STATE_DIM, seed=1)
            with self.assertRaisesRegex(ValueError, "state/action schema"):
                new.load_model(str(Path(tmp) / "old"))

    def test_warmup_episodes_skip_validation(self):
        self.assertFalse(validation_due(10, 10, 50, 1000, 100))
        self.assertTrue(validation_due(20, 10, 50, 1000, 100))
        self.assertTrue(validation_due(50, 10, 50, 1000, 100))
        self.assertFalse(validation_due(1, 1, 2, 1, 2))
        self.assertTrue(validation_due(1, 1, 2, 1, 2, 0))
        self.assertFalse(validation_due(10, 10, 10, 0, 100))
        self.assertFalse(validation_due(20, 10, 50, 2000, 100))

    def test_first_ten_episodes_never_run_validation_including_start_and_end(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = train_parser().parse_args(["--episodes", "10", "--train-iters", "1", "--eval-iters", "1",
                "--eval-every", "1", "--eval-trials", "1", "--particles", "4", "--points", "4",
                "--warmup-iters", "1", "--updates-per-iter", "1", "--batch-size", "4",
                "--eval-map-seeds", "1000", "--out", tmp])
            with patch.object(TD3, "train_td3", return_value=None), \
                    patch("trainer.pretrain.evaluate", side_effect=AssertionError("Unexpected validation")) as evaluation:
                train(args)
            evaluation.assert_not_called()
            self.assertTrue((Path(tmp) / "initial.pt").exists())
            self.assertTrue((Path(tmp) / "last.pt").exists())
            self.assertFalse((Path(tmp) / "best.pt").exists())
            self.assertFalse((Path(tmp) / "validation.csv").exists())

    def test_initial_actor_is_first_measured_after_skip_window(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = train_parser().parse_args(["--episodes", "11", "--train-iters", "1", "--eval-iters", "1",
                "--eval-every", "10", "--eval-trials", "1", "--particles", "4", "--points", "4",
                "--warmup-iters", "1", "--updates-per-iter", "1", "--batch-size", "4",
                "--eval-map-seeds", "1000", "--out", tmp])
            with patch.object(TD3, "train_td3", return_value=None), \
                    patch("trainer.pretrain.evaluate", wraps=evaluate) as evaluation:
                train(args)
            self.assertEqual(evaluation.call_count, 1)
            self.assertEqual(evaluation.call_args.args[6], 11)
            baseline = json.loads((Path(tmp) / "validation_initial.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline["evaluated_after_episode"], 11)
            _, checkpoint = load_checkpoint(Path(tmp) / "best.pt", "cpu")
            self.assertEqual(checkpoint["episode"], 0)

    def test_default_configuration_is_100_episodes_and_short_cutoffs(self):
        args = train_parser().parse_args([])
        validate_args(args)
        self.assertEqual((args.episodes, args.train_iters, args.eval_iters), (100, 100, 50))
        self.assertEqual(args.gamma, .97)
        self.assertEqual(args.reward_pbest, 5.)
        self.assertFalse(hasattr(args, "pop_seed"))
        args.eval_map_seeds = [0]
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_saved_validation_arrays_are_reused_exactly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            source = Path(tmp) / "source"
            original = create_cases(source, [1000], 1, 4, 4, 0)
            args = train_parser().parse_args(["--validation-cases", str(source), "--particles", "4",
                                              "--points", "4", "--eval-map-seeds", "1000", "--eval-trials", "1"])
            copied = validation_cases(args, Path(tmp) / "copied")
            self.assertEqual(original[0].read_bytes(), copied[0].read_bytes())
            args.train_map_seeds = [1000]
            with self.assertRaises(ValueError):
                validation_cases(args, Path(tmp) / "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
