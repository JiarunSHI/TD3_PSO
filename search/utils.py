from typing import Optional, Tuple
import numpy as np
import torch


TYPE_GOAL = 0
TYPE_STATE = 1
TYPE_ACTION = 2
TYPE_REWARD = 3
TYPE_RETURN = 4
TYPE_PAD = 5


def to_torch(x, dtype=torch.long, device=None):
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def build_initial_sequence(goal_xy, start_xy, discretizer, device=None):
    """
    初始前缀只包含：
        [goal_x, goal_y, start_x, start_y]

    这就是模型的条件前缀。
    第一个生成 token 就是 a1_x，不再重复构造 goal 或 start。
    """
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    start_xy = np.asarray(start_xy, dtype=np.float32)

    goal_tok = to_torch(
        discretizer.discretize(goal_xy),
        dtype=torch.long,
        device=device
    ).view(-1)

    start_tok = to_torch(
        discretizer.discretize(start_xy),
        dtype=torch.long,
        device=device
    ).view(-1)

    tokens = torch.cat([goal_tok, start_tok], dim=0).unsqueeze(0)  # [1, 4]
    types = torch.tensor(
        [TYPE_GOAL, TYPE_GOAL, TYPE_STATE, TYPE_STATE],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)  # [1, 4]

    return tokens, types


def infer_next_token_type(current_length: int, prefix_len: int = 4, obs_dim: int = 2, action_dim: int = 2):
    """
    当前历史长度为 current_length 时，下一个 token 的 type 是什么。

    序列规则：
        prefix:
            [goal_x, goal_y, start_x, start_y]

        后续循环：
            [ACTION, ACTION, REWARD, RETURN, STATE, STATE]
    """
    if current_length < prefix_len:
        raise ValueError("current_length must be >= prefix_len")

    rel = current_length - prefix_len
    cycle_len = action_dim + 2 + obs_dim  # 2 action + reward + return + 2 state = 6
    pos = rel % cycle_len

    if pos < action_dim:
        return TYPE_ACTION
    elif pos == action_dim:
        return TYPE_REWARD
    elif pos == action_dim + 1:
        return TYPE_RETURN
    else:
        return TYPE_STATE


def append_next_token(tokens, types, next_token, next_type, device=None):
    """
    把一个 token 追加到历史序列后面。
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if types.dim() == 1:
        types = types.unsqueeze(0)

    if not torch.is_tensor(next_token):
        next_token = torch.tensor(next_token, dtype=torch.long, device=device)
    else:
        next_token = next_token.to(dtype=torch.long, device=device)

    next_type = torch.tensor(next_type, dtype=torch.long, device=device)

    tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)
    types = torch.cat([types, next_type.view(1, 1)], dim=1)
    return tokens, types













