from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


REWARD_DIM = 1
VALUE_DIM = 1

TYPE_GOAL = 0
TYPE_STATE = 1
TYPE_ACTION = 2
TYPE_REWARD = 3
TYPE_RETURN = 4
TYPE_PAD = 5


@dataclass
class BeamState:
    tokens: torch.Tensor        # [seq_len]
    types: torch.Tensor         # [seq_len]
    logp: float                 # 累积 token log-prob
    task: float                 # 累积任务分数（折扣后的 r + R）
    collision_cost: float       # 累积碰撞惩罚
    score: float                # 最终综合分数
    done_steps: int             # 已完成的 transition 数
    last_xy: np.ndarray         # 上一个 transition 的状态坐标 [2,]


def _recompute_score(beam: BeamState, alpha_task: float) -> None:
    beam.score = beam.logp + alpha_task * beam.task + beam.collision_cost


def _get_discretizer(pack, key_candidates: Sequence[str]):
    if isinstance(pack, dict):
        for k in key_candidates:
            if k in pack:
                return pack[k]
    return pack


def _decode_tokens_generic(disc, token_ids: Sequence[int]) -> np.ndarray:
    """
    尽量兼容各种离散器：
      - reconstruct(...)
      - decode(...)
      - inverse_transform(...)
      - low/high/num_bins 的 bin center 近似
    """
    token_ids = list(map(int, token_ids))

    for method_name in ("reconstruct", "decode", "inverse_transform"):
        if hasattr(disc, method_name):
            fn = getattr(disc, method_name)
            try:
                out = fn(token_ids)
                arr = np.asarray(out, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return arr
            except Exception:
                pass

    # fallback: 由 low/high/num_bins 近似还原 bin center
    if hasattr(disc, "low") and hasattr(disc, "high") and hasattr(disc, "num_bins"):
        low = np.asarray(disc.low, dtype=np.float32).reshape(-1)
        high = np.asarray(disc.high, dtype=np.float32).reshape(-1)
        num_bins = max(int(getattr(disc, "num_bins", 1)), 1)

        if low.size == 1 and len(token_ids) > 1:
            low = np.repeat(low, len(token_ids))
            high = np.repeat(high, len(token_ids))
        elif low.size < len(token_ids):
            low = np.resize(low, len(token_ids))
            high = np.resize(high, len(token_ids))

        token_arr = np.asarray(token_ids, dtype=np.float32)
        span = np.maximum(high - low, 1e-8)
        bin_width = span / num_bins
        return low[: len(token_ids)] + (token_arr + 0.5) * bin_width[: len(token_ids)]

    raise AttributeError(
        f"Cannot decode tokens with discretizer of type {type(disc)}"
    )


def _decode_xy(discretizers, token_pair: Sequence[int]) -> np.ndarray:
    obs_disc = _get_discretizer(
        discretizers,
        ["obs", "observation", "state", "goal", "state_discretizer"]
    )
    arr = _decode_tokens_generic(obs_disc, token_pair).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Failed to decode xy tokens: {token_pair}")
    return arr[:2].copy()


def _decode_scalar(discretizers, token_id: int, key_candidates: Sequence[str]) -> float:
    disc = _get_discretizer(discretizers, key_candidates)
    arr = _decode_tokens_generic(disc, [token_id]).reshape(-1)
    if arr.size < 1:
        raise ValueError(f"Failed to decode scalar token: {token_id}")
    return float(arr[0])


def _model_next_logits(model, tokens: torch.Tensor, types: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """
    model(tokens, types) -> logits
    取最后一个位置 logits 作为 next-token 分布。

    支持：
      - logits: [B, L, V]
      - logits: [B, V]
      - dict(logits=...)
      - tuple(...)
    """
    assert tokens.shape == types.shape, "tokens and types must have same shape"

    max_len = getattr(model, "block_size", None)
    if max_len is not None and tokens.size(1) > max_len:
        if prefix_len >= max_len:
            raise ValueError(
                f"prefix_len({prefix_len}) >= model.block_size({max_len}), cannot truncate safely."
            )
        tail_len = max_len - prefix_len
        tokens = torch.cat([tokens[:, :prefix_len], tokens[:, -tail_len:]], dim=1)
        types = torch.cat([types[:, :prefix_len], types[:, -tail_len:]], dim=1)

    out = model(tokens, types)

    if isinstance(out, dict):
        logits = out["logits"]
    elif isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    if logits.dim() == 3:
        logits = logits[:, -1, :]
    elif logits.dim() != 2:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    return logits


def _select_candidates(
    log_probs: torch.Tensor,
    topk: Optional[int] = None,
    cdf: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
        token_ids: [K]
        token_logps: [K]
    """
    probs = torch.exp(log_probs)

    sorted_logp, sorted_idx = torch.sort(log_probs, descending=True)
    sorted_probs = probs[sorted_idx]

    if cdf is not None:
        cdf = float(cdf)
        cdf = min(max(cdf, 1e-6), 1.0)
        cum = torch.cumsum(sorted_probs, dim=0)
        keep = int(torch.searchsorted(cum, torch.tensor(cdf, device=cum.device)).item()) + 1
        if topk is not None:
            keep = min(keep, int(topk))
        keep = max(keep, 1)
    else:
        keep = int(topk) if topk is not None else 1
        keep = max(keep, 1)

    return sorted_idx[:keep], sorted_logp[:keep]


def _expand_token_beams(
    model,
    beams: List[BeamState],
    prefix_len: int,
    next_type: int,
    token_topk: int,
    token_cdf: Optional[float] = None,
) -> List[BeamState]:
    """
    对每个 beam 只扩展一个 token。
    """
    if len(beams) == 0:
        return []

    seqs = torch.stack([b.tokens for b in beams], dim=0)  # [B, L]
    typs = torch.stack([b.types for b in beams], dim=0)    # [B, L]

    logits = _model_next_logits(model, seqs, typs, prefix_len=prefix_len)
    log_probs = F.log_softmax(logits, dim=-1)              # [B, V]

    # 不允许生成 PAD
    pad_token_id = getattr(model, "vocab_size", None)
    if pad_token_id is not None and pad_token_id < log_probs.size(-1):
        log_probs[:, pad_token_id] = float("-inf")

    new_beams: List[BeamState] = []
    for i, beam in enumerate(beams):
        cand_ids, cand_logps = _select_candidates(
            log_probs[i], topk=token_topk, cdf=token_cdf
        )

        for tok_id, tok_logp in zip(cand_ids.tolist(), cand_logps.tolist()):
            tok = torch.tensor([tok_id], dtype=beam.tokens.dtype, device=beam.tokens.device)
            typ = torch.tensor([next_type], dtype=beam.types.dtype, device=beam.types.device)

            new_beams.append(
                BeamState(
                    tokens=torch.cat([beam.tokens, tok], dim=0),
                    types=torch.cat([beam.types, typ], dim=0),
                    logp=beam.logp + float(tok_logp),
                    task=beam.task,
                    collision_cost=beam.collision_cost,
                    score=beam.score,
                    done_steps=beam.done_steps,
                    last_xy=beam.last_xy.copy(),
                )
            )

    return new_beams


def _prune_beams(
    beams: List[BeamState],
    beam_width: int,
    use_score: bool = True,
) -> List[BeamState]:
    if len(beams) <= beam_width:
        return beams

    key_fn = (lambda b: b.score) if use_score else (lambda b: b.logp)
    return sorted(beams, key=key_fn, reverse=True)[:beam_width]


def _line_collision_check(p1, p2, obstacles, num_checks=10):
    """
    不考虑机器人半径，只检查线段是否穿过障碍物圆。
    obstacles: list of (x, y, radius)
    """
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    for i in range(num_checks + 1):
        alpha = i / num_checks
        x = p1[0] * (1 - alpha) + p2[0] * alpha
        y = p1[1] * (1 - alpha) + p2[1] * alpha

        for ox, oy, orad in obstacles:
            if np.hypot(x - ox, y - oy) <= orad:
                return True
    return False


def _apply_transition_task_update(
    beam: BeamState,
    discretizers,
    discount: float,
    trans_idx: int,
    action_dim: int,
) -> None:
    """
    当前 beam 的末尾已经完成一个 [action..., reward, return] block，
    用最后两个 token 计算 r + R。
    """
    reward_token = int(beam.tokens[-2].item())
    return_token = int(beam.tokens[-1].item())

    reward_val = _decode_scalar(discretizers, reward_token, ["reward", "rew", "r"])
    return_val = _decode_scalar(discretizers, return_token, ["value", "return", "rtg", "R"])

    beam.task += (discount ** trans_idx) * (reward_val + return_val)
    beam.done_steps = trans_idx + 1


def _apply_collision_update(
    beam: BeamState,
    discretizers,
    collision_penalty: float,
) -> None:
    """
    当前 beam 的末尾已经完成一个 state block，
    检查 last_xy -> current_xy 是否碰撞。
    """
    curr_xy = _decode_xy(discretizers, beam.tokens[-2:].tolist())
    if _line_collision_check(beam.last_xy, curr_xy, getattr(discretizers, "obstacles", [])):
        beam.collision_cost += collision_penalty
    beam.last_xy = np.asarray(curr_xy, dtype=np.float32).copy()


@torch.no_grad()
def beam_plan(
    model,
    prefix_tokens: torch.Tensor,
    prefix_types: torch.Tensor,
    discretizers,
    n_steps: int,
    beam_width: int,
    n_expand: int,
    state_dim: int,
    action_dim: int,
    start_xy,
    obstacles,
    discount: float = 1.0,
    alpha_task: float = 1.0,
    max_context_transitions: Optional[int] = None,  # 保留参数，兼容旧接口
    k_state: Optional[int] = None,
    k_act: Optional[int] = None,
    k_rew: int = 1,
    cdf_state: Optional[float] = None,
    cdf_act: Optional[float] = None,
    cdf_rew: Optional[float] = None,
    collision_penalty: float = -1e9,
    verbose: bool = True,
):
    """
    统一后的 token-wise beam search。

    序列规则（prefix 已包含 goal + start）：
        prefix:
            [goal_x, goal_y, start_x, start_y]

        第一段（特殊）：
            [a1_x, a1_y, r1, R1]   # 只生成 4 个 token

        后续段：
            [s2_x, s2_y, a2_x, a2_y, r2, R2]
            [s3_x, s3_y, a3_x, a3_y, r3, R3]
            ...

    搜索策略：
        1) 每一步只扩展一个 token
        2) beam 的对象是整条轨迹的累计 logp
        3) 每完成一个 [action..., reward, return] block，就加入 r + R
        4) 每完成一个 state block，就检查与上一个 state 的碰撞
        5) 最终 score = logp + alpha_task * task + collision_cost

    返回：
        dict，包含 best 轨迹的 tokens / types / score 等信息
    """
    assert prefix_tokens.ndim == 2 and prefix_tokens.shape[0] == 1, "prefix_tokens must be [1, L]"
    assert prefix_types.ndim == 2 and prefix_types.shape[0] == 1, "prefix_types must be [1, L]"
    assert prefix_tokens.shape == prefix_types.shape, "prefix_tokens and prefix_types must have same shape"

    prefix_len = prefix_tokens.shape[1]
    assert prefix_len >= 4, "prefix must contain [goal_x, goal_y, start_x, start_y] at minimum"

    if k_state is None:
        k_state = n_expand
    if k_act is None:
        k_act = n_expand

    action_block_types = [TYPE_ACTION] * action_dim + [TYPE_REWARD] + [TYPE_RETURN]
    state_block_types = [TYPE_STATE] * state_dim

    # 这里保存的是当前候选轨迹
    beams: List[BeamState] = [
        BeamState(
            tokens=prefix_tokens[0].clone(),
            types=prefix_types[0].clone(),
            logp=0.0,
            task=0.0,
            collision_cost=0.0,
            score=0.0,
            done_steps=0,
            last_xy=np.asarray(start_xy, dtype=np.float32)[:2].copy(),
        )
    ]
    _recompute_score(beams[0], alpha_task)

    # ------------------------------------------------------------
    # 1) 第一段 transition：只生成 4 个 token
    #    [a1_x, a1_y, r1, R1]
    # ------------------------------------------------------------
    for next_type in action_block_types:
        token_topk, token_cdf = (
            (k_act, cdf_act) if next_type == TYPE_ACTION else (k_rew, cdf_rew)
        )
        beams = _expand_token_beams(
            model=model,
            beams=beams,
            prefix_len=prefix_len,
            next_type=next_type,
            token_topk=int(token_topk) if token_topk is not None else 1,
            token_cdf=token_cdf,
        )
        for b in beams:
            _recompute_score(b, alpha_task)
        beams = _prune_beams(beams, beam_width, use_score=True)

    # 第一段完成后，更新 task（不做碰撞检查）
    scored_beams: List[BeamState] = []
    for b in beams:
        try:
            _apply_transition_task_update(
                beam=b,
                discretizers=discretizers,
                discount=discount,
                trans_idx=0,
                action_dim=action_dim,
            )
            _recompute_score(b, alpha_task)
            scored_beams.append(b)
        except Exception:
            continue

    if len(scored_beams) == 0:
        raise RuntimeError("All beams failed after first action/reward block.")
    beams = _prune_beams(scored_beams, beam_width, use_score=True)

    # ------------------------------------------------------------
    # 2) 后续 transition
    #    每个 transition 都是：
    #      state block:  [s_{t+1,x}, s_{t+1,y}]
    #      action block: [a_{t+1,x}, a_{t+1,y}, r_{t+1}, R_{t+1}]
    # ------------------------------------------------------------
    for trans_idx in range(1, n_steps):
        # 2.1 先生成 state block
        for next_type in state_block_types:
            beams = _expand_token_beams(
                model=model,
                beams=beams,
                prefix_len=prefix_len,
                next_type=next_type,
                token_topk=int(k_state),
                token_cdf=cdf_state,
            )
            for b in beams:
                _recompute_score(b, alpha_task)
            beams = _prune_beams(beams, beam_width, use_score=True)

        # state block 完成后：检查碰撞
        scored_beams = []
        for b in beams:
            try:
                curr_xy = _decode_xy(discretizers, b.tokens[-2:].tolist())
                if _line_collision_check(b.last_xy, curr_xy, obstacles):
                    b.collision_cost += collision_penalty
                b.last_xy = np.asarray(curr_xy, dtype=np.float32).copy()
                _recompute_score(b, alpha_task)
                scored_beams.append(b)
            except Exception:
                continue

        if len(scored_beams) == 0:
            raise RuntimeError(f"All beams failed at collision stage of transition {trans_idx}.")
        beams = _prune_beams(scored_beams, beam_width, use_score=True)

        # 2.2 再生成 action/reward/return block
        for next_type in action_block_types:
            token_topk, token_cdf = (
                (k_act, cdf_act) if next_type == TYPE_ACTION else (k_rew, cdf_rew)
            )
            beams = _expand_token_beams(
                model=model,
                beams=beams,
                prefix_len=prefix_len,
                next_type=next_type,
                token_topk=int(token_topk) if token_topk is not None else 1,
                token_cdf=token_cdf,
            )
            for b in beams:
                _recompute_score(b, alpha_task)
            beams = _prune_beams(beams, beam_width, use_score=True)

        # action/reward/return block 完成后：更新 task
        scored_beams = []
        for b in beams:
            try:
                _apply_transition_task_update(
                    beam=b,
                    discretizers=discretizers,
                    discount=discount,
                    trans_idx=trans_idx,
                    action_dim=action_dim,
                )
                _recompute_score(b, alpha_task)
                scored_beams.append(b)
            except Exception:
                continue

        if len(scored_beams) == 0:
            raise RuntimeError(f"All beams failed at task stage of transition {trans_idx}.")
        beams = _prune_beams(scored_beams, beam_width, use_score=True)

        if verbose and len(beams) > 0:
            best_now = max(beams, key=lambda b: b.score)
            print(
                f"[beam] trans={trans_idx:03d} | "
                f"best score={best_now.score:.4f} | "
                f"logp={best_now.logp:.4f} | "
                f"task={best_now.task:.4f} | "
                f"collision={best_now.collision_cost:.4f} | "
                f"len={best_now.tokens.numel()}"
            )

    best_beam = max(beams, key=lambda b: b.score)

    return {
        "full_tokens": best_beam.tokens,                    # 包含 prefix
        "full_types": best_beam.types,                      # 包含 prefix
        "generated_tokens": best_beam.tokens[prefix_len:],   # 只看生成部分
        "generated_types": best_beam.types[prefix_len:],
        "score": best_beam.score,
        "logp": best_beam.logp,
        "task": best_beam.task,
        "collision_cost": best_beam.collision_cost,
        "done_steps": best_beam.done_steps,
        "last_xy": best_beam.last_xy,
    }