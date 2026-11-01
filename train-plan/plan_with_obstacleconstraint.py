import argparse
import json
import os
from os.path import join
from types import SimpleNamespace
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import torch


from env.map_env import MapEnv
from search.core import beam_plan
from search.utils import build_initial_sequence
from TT_structure import GPT


TYPE_GOAL = 0
TYPE_STATE = 1
TYPE_ACTION = 2
TYPE_REWARD = 3
TYPE_RETURN = 4
TYPE_PAD = 5

REWARD_DIM = 1
VALUE_DIM = 1


def build_parser():
    parser = argparse.ArgumentParser(description="Beam-search inference: best full trajectory")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--model_vocab_size", type=int, default=100)
    parser.add_argument("--model_block_size", type=int, default=512)
    parser.add_argument("--model_n_layer", type=int, default=6)
    parser.add_argument("--model_n_head", type=int, default=8)
    parser.add_argument("--model_n_embd", type=int, default=128)
    parser.add_argument("--model_embd_pdrop", type=float, default=0.1)
    parser.add_argument("--model_resid_pdrop", type=float, default=0.1)
    parser.add_argument("--model_attn_pdrop", type=float, default=0.1)

    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--n_expand", type=int, default=8)

    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=2)

    parser.add_argument("--k_state", type=int, default=8)
    parser.add_argument("--k_act", type=int, default=8)
    parser.add_argument("--k_rew", type=int, default=1)
    parser.add_argument("--cdf_state", type=float, default=None)
    parser.add_argument("--cdf_act", type=float, default=None)
    parser.add_argument("--cdf_rew", type=float, default=None)

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--alpha_task", type=float, default=1.0)
    parser.add_argument("--collision_penalty", type=float, default=-1e9)

    parser.add_argument("--start_x", type=float, default=1.0)
    parser.add_argument("--start_y", type=float, default=1.0)
    parser.add_argument("--goal_x", type=float, default=19.0)
    parser.add_argument("--goal_y", type=float, default=19.0)
    parser.add_argument("--goal_radius", type=float, default=0.5)

    parser.add_argument("--savepath", type=str, default="./outputs")
    return parser


def euclidean(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a[:2] - b[:2]))


def _get_discretizer(pack, key_candidates):
    if isinstance(pack, dict):
        for k in key_candidates:
            if k in pack:
                return pack[k]
    return pack


def _decode_generic(discretizer, token_ids):
    token_ids = list(map(int, token_ids))

    for method_name in ("reconstruct", "decode", "inverse_transform"):
        if hasattr(discretizer, method_name):
            fn = getattr(discretizer, method_name)
            try:
                out = fn(token_ids)
                arr = np.asarray(out, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return arr
            except Exception:
                pass

    raise AttributeError(f"Cannot decode tokens with discretizer type {type(discretizer)}")


def _decode_full_best_trajectory(generated_tokens, discretizers, start_xy, state_dim=2, action_dim=2):
    """
    按当前 beam_plan 的生成顺序解析整条最优轨迹。

    当前序列格式（prefix 后）：
        [a1_x, a1_y, r1, R1, s2_x, s2_y, a2_x, a2_y, r2, R2, s3_x, s3_y, ...]
    """
    if generated_tokens.dim() == 2:
        generated_tokens = generated_tokens[0]
    seq = generated_tokens.view(-1).detach().cpu().tolist()

    obs_disc = _get_discretizer(
        discretizers,
        ["obs", "observation", "state", "goal", "state_discretizer"]
    )
    act_disc = _get_discretizer(
        discretizers,
        ["act", "action", "action_discretizer"]
    )
    rew_disc = _get_discretizer(
        discretizers,
        ["reward", "rew", "r", "reward_discretizer"]
    )
    val_disc = _get_discretizer(
        discretizers,
        ["value", "return", "rtg", "R", "value_discretizer"]
    )

    if obs_disc is None or act_disc is None or rew_disc is None or val_disc is None:
        raise RuntimeError("Checkpoint discretizers missing one of: obs/act/reward/value.")

    idx = 0
    states = [np.asarray(start_xy, dtype=np.float32)[:2].copy()]
    transitions = []

    # 第一段：a1_x, a1_y, r1, R1
    first_block_len = action_dim + REWARD_DIM + VALUE_DIM
    if len(seq) < first_block_len:
        return np.asarray(states, dtype=np.float32), transitions

    action_tokens = seq[idx: idx + action_dim]
    reward_token = seq[idx + action_dim]
    value_token = seq[idx + action_dim + 1]
    action = _decode_generic(act_disc, action_tokens)[:action_dim]
    reward = float(_decode_generic(rew_disc, [reward_token])[0])
    rtg = float(_decode_generic(val_disc, [value_token])[0])
    idx += first_block_len

    # 第一个 state block：s2_x, s2_y
    if idx + state_dim <= len(seq):
        next_state_tokens = seq[idx: idx + state_dim]
        next_xy = _decode_generic(obs_disc, next_state_tokens)[:state_dim]
        states.append(np.asarray(next_xy, dtype=np.float32).copy())
        transitions.append(
            {
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "reward": reward,
                "rtg": rtg,
                "next_state": np.asarray(next_xy, dtype=np.float32).tolist(),
            }
        )
        idx += state_dim
    else:
        transitions.append(
            {
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "reward": reward,
                "rtg": rtg,
            }
        )
        return np.asarray(states, dtype=np.float32), transitions

    # 后续循环：a2,r2,R2 -> s3 ; a3,r3,R3 -> s4 ; ...
    while idx < len(seq):
        if idx + first_block_len > len(seq):
            break

        action_tokens = seq[idx: idx + action_dim]
        reward_token = seq[idx + action_dim]
        value_token = seq[idx + action_dim + 1]
        action = _decode_generic(act_disc, action_tokens)[:action_dim]
        reward = float(_decode_generic(rew_disc, [reward_token])[0])
        rtg = float(_decode_generic(val_disc, [value_token])[0])
        idx += first_block_len

        if idx + state_dim <= len(seq):
            next_state_tokens = seq[idx: idx + state_dim]
            next_xy = _decode_generic(obs_disc, next_state_tokens)[:state_dim]
            states.append(np.asarray(next_xy, dtype=np.float32).copy())
            transitions.append(
                {
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "reward": reward,
                    "rtg": rtg,
                    "next_state": np.asarray(next_xy, dtype=np.float32).tolist(),
                }
            )
            idx += state_dim
        else:
            transitions.append(
                {
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "reward": reward,
                    "rtg": rtg,
                }
            )
            break

    return np.asarray(states, dtype=np.float32), transitions


def plot_plan_path(path_xy, obstacles, start_pos, goal_pos, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")

    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color="red", alpha=0.5)
        ax.add_patch(circle)

    path_xy = np.asarray(path_xy, dtype=np.float32)
    if path_xy.size > 0:
        ax.plot(path_xy[:, 0], path_xy[:, 1], "b-", linewidth=2, marker="o", markersize=4, label="Path")

    ax.plot(start_pos[0], start_pos[1], "go", markersize=10, label="Start")
    ax.plot(goal_pos[0], goal_pos[1], "b*", markersize=15, label="Goal")
    ax.set_title("Best Beam Trajectory")
    ax.legend()
    ax.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_artifacts(model_path, device, fallback_cfg=None):
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        discretizers = getattr(ckpt, "discretizers", None)
        epoch = int(getattr(ckpt, "epoch", 0))
        return model, discretizers, epoch

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint格式不支持: {type(ckpt)}")

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state_dict is None:
        raise RuntimeError("Checkpoint缺少 'model_state_dict' 或 'state_dict'。")

    # 1) 先拿 checkpoint 里的 config
    ckpt_cfg = ckpt.get("model_config", {})
    if isinstance(ckpt_cfg, SimpleNamespace):
        ckpt_cfg = vars(ckpt_cfg)
    elif not isinstance(ckpt_cfg, dict):
        ckpt_cfg = dict(ckpt_cfg)

    # 2) 再拿 fallback
    fb_cfg = fallback_cfg or {}

    # 3) 合并：fallback 提供默认值，ckpt 覆盖默认值
    #    这样即使 checkpoint 缺少 type_size 之类字段，也不会报错
    merged_cfg = dict(fb_cfg)
    merged_cfg.update(ckpt_cfg)

    # 4) 强制补齐关键字段
    merged_cfg.setdefault("vocab_size", 100)
    merged_cfg.setdefault("type_size", 6)
    merged_cfg.setdefault("block_size", 512)
    merged_cfg.setdefault("n_layer", 6)
    merged_cfg.setdefault("n_head", 8)
    merged_cfg.setdefault("n_embd", 128)
    merged_cfg.setdefault("embd_pdrop", 0.1)
    merged_cfg.setdefault("resid_pdrop", 0.1)
    merged_cfg.setdefault("attn_pdrop", 0.1)

    model_cfg = SimpleNamespace(**merged_cfg)

    model = GPT(model_cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    discretizers = ckpt.get("discretizers", ckpt.get("discretizer", None))
    if discretizers is None:
        raise RuntimeError("Checkpoint缺少 'discretizers'。")

    if hasattr(discretizers, "discretize"):
        discretizers = {
            "obs": discretizers,
            "act": discretizers,
            "reward": discretizers,
            "value": discretizers,
        }

    epoch = int(ckpt.get("epoch", 0))
    return model, discretizers, epoch

def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    fallback_cfg = {
        "vocab_size": args.model_vocab_size,
        "type_size": 6,
        "block_size": args.model_block_size,
        "n_layer": args.model_n_layer,
        "n_head": args.model_n_head,
        "n_embd": args.model_n_embd,
        "embd_pdrop": args.model_embd_pdrop,
        "resid_pdrop": args.model_resid_pdrop,
        "attn_pdrop": args.model_attn_pdrop,
    }

    model, discretizers, gpt_epoch = load_artifacts(
        args.model_path,
        device,
        fallback_cfg=fallback_cfg,
    )

    map_env = MapEnv()
    obstacles = map_env.generate_mixed()
    print(f"障碍物数量: {len(obstacles)}")

    start_xy = np.array([args.start_x, args.start_y], dtype=np.float32)
    goal_xy = np.array([args.goal_x, args.goal_y], dtype=np.float32)

    obs_disc = _get_discretizer(
        discretizers,
        ["obs", "observation", "state", "goal", "state_discretizer"]
    )
    if obs_disc is None:
        raise RuntimeError("Cannot find observation discretizer in checkpoint.")

    # prefix = [goal_x, goal_y, start_x, start_y]
    prefix_tokens, prefix_types = build_initial_sequence(
        goal_xy=goal_xy,
        start_xy=start_xy,
        discretizer=obs_disc,
        device=device,
    )

    # 一次 beam search，直接拿最高分整条轨迹
    plan_out = beam_plan(
        model=model,
        prefix_tokens=prefix_tokens,
        prefix_types=prefix_types,
        discretizers=discretizers,
        n_steps=args.horizon,
        beam_width=args.beam_width,
        n_expand=args.n_expand,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        start_xy=start_xy,
        obstacles=obstacles,
        discount=args.discount,
        alpha_task=args.alpha_task,
        max_context_transitions=None,
        k_state=args.k_state,
        k_act=args.k_act,
        k_rew=args.k_rew,
        cdf_state=args.cdf_state,
        cdf_act=args.cdf_act,
        cdf_rew=args.cdf_rew,
        collision_penalty=args.collision_penalty,
        verbose=True,
    )

    if isinstance(plan_out, dict):
        generated_tokens = plan_out["generated_tokens"]
        full_tokens = plan_out.get("full_tokens", None)
        best_score = float(plan_out.get("score", 0.0))
        best_logp = float(plan_out.get("logp", 0.0))
        best_task = float(plan_out.get("task", 0.0))
        collision_cost = float(plan_out.get("collision_cost", 0.0))
    else:
        generated_tokens = plan_out
        full_tokens = None
        best_score = best_logp = best_task = collision_cost = 0.0

    # 解析整条最优轨迹
    path_xy, transitions = _decode_full_best_trajectory(
        generated_tokens=generated_tokens,
        discretizers=discretizers,
        start_xy=start_xy,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )

    # 画图
    plot_plan_path(
        path_xy=path_xy,
        obstacles=obstacles,
        start_pos=start_xy,
        goal_pos=goal_xy,
        save_path=join(args.savepath, "planned_path.png"),
    )

    final_xy = path_xy[-1] if len(path_xy) > 0 else start_xy
    dist = euclidean(final_xy, goal_xy)

    result = {
        "start": start_xy.tolist(),
        "goal": goal_xy.tolist(),
        "final": np.asarray(final_xy, dtype=np.float32).tolist(),
        "steps": int(max(len(path_xy) - 1, 0)),
        "reached_goal": bool(dist <= args.goal_radius),
        "distance_to_goal": float(dist),
        "gpt_epoch": int(gpt_epoch),
        "best_score": float(best_score),
        "best_logp": float(best_logp),
        "best_task": float(best_task),
        "collision_cost": float(collision_cost),
        "path": np.asarray(path_xy, dtype=np.float32).tolist(),
        "transitions": transitions,
    }

    os.makedirs(args.savepath, exist_ok=True)
    with open(join(args.savepath, "rollout.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("Best score:", best_score)
    print("Best logp:", best_logp)
    print("Best task:", best_task)
    print("Collision cost:", collision_cost)
    print("Distance to goal:", dist)
    print("Saved:", join(args.savepath, "planned_path.png"))
    print("Saved:", join(args.savepath, "rollout.json"))


if __name__ == "__main__":
    main()