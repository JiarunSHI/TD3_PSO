import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
import pickle
import functools
import warnings


class AdaptiveDiscretizer:
    """
    对二维状态 (x, y) 做障碍物感知的非均匀离散化。
    靠近障碍物的区域分辨率更细，远离障碍物的区域分辨率更粗。
    """
    def __init__(self, obstacle_specs, low, high, num_bins, alpha=1.0, beta=5.0):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.num_bins = int(num_bins)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.obstacle_specs = list(obstacle_specs)

        if len(self.obstacle_specs) > 0:
            centers = np.array([c for c, _ in self.obstacle_specs], dtype=np.float32)
            self.radii = np.array([r for _, r in self.obstacle_specs], dtype=np.float32)
            self.tree = cKDTree(centers)
        else:
            self.tree = None
            self.radii = None

        corners = np.array([
            [self.low[0], self.low[1]],
            [self.low[0], self.high[1]],
            [self.high[0], self.low[1]],
            [self.high[0], self.high[1]],
        ], dtype=np.float32)

        transformed = self._transform(corners)
        self.trans_low = transformed.min(axis=0)
        self.trans_high = transformed.max(axis=0)
        span = self.trans_high - self.trans_low
        self.bin_width = np.maximum(span / self.num_bins, 1e-8)

    def _distance_to_obstacle(self, points):
        if self.tree is None:
            return np.full(points.shape[0], 1e6, dtype=np.float32)
        dist, idx = self.tree.query(points)
        return dist - self.radii[idx]

    def _transform(self, points):
        if self.tree is None:
            return points
        d = self._distance_to_obstacle(points)
        d = np.maximum(d, 1e-6)
        scale = 1.0 + self.alpha * np.exp(-self.beta * d)
        return points * scale[:, None]

    def discretize(self, points):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        transformed = self._transform(pts)
        idx = np.floor((transformed - self.trans_low) / self.bin_width).astype(np.int64)
        idx = np.clip(idx, 0, self.num_bins - 1)
        return idx

    def discretize_single(self, point):
        return self.discretize(np.asarray(point)[None, :])[0]


class UniformDiscretizer:
    """
    对标量或向量做均匀离散化。
    """
    def __init__(self, low, high, num_bins):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.num_bins = int(num_bins)
        span = self.high - self.low
        self.bin_width = np.maximum(span / self.num_bins, 1e-8)

    def discretize(self, x):
        arr = np.asarray(x, dtype=np.float32)
        idx = np.floor((arr - self.low) / self.bin_width).astype(np.int64)
        return np.clip(idx, 0, self.num_bins - 1)


def compute_discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def prepare_discrete_dataset(traj_path, obstacles_list, cfg):
    """
    输入:
        traj_path: pkl 文件
        obstacles_list: [(x, y, r), ...]
        cfg: 需要包含
            vocab_size
            discount
            alpha
            beta
            global_obs_bounds (可选)
    输出:
        sequences: list[torch.LongTensor], 每条轨迹是一个 1D token 序列
        goals:     torch.LongTensor, shape (N, 2)
        discretizers: dict
    """
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    all_obs, all_acts, all_rewards = [], [], []
    for traj in trajectories:
        all_obs.append(traj['observations'])
        all_acts.append(traj['actions'])
        all_rewards.append(traj['rewards'])

    all_obs = np.concatenate(all_obs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)

    if hasattr(cfg, 'global_obs_bounds') and cfg.global_obs_bounds is not None:
        obs_low = np.array(cfg.global_obs_bounds[0], dtype=np.float32)
        obs_high = np.array(cfg.global_obs_bounds[1], dtype=np.float32)
    else:
        obs_low = all_obs.min(axis=0)
        obs_high = all_obs.max(axis=0)
    obs_high = np.maximum(obs_high, obs_low + 1e-6)

    act_low = all_acts.min(axis=0)
    act_high = all_acts.max(axis=0)

    reward_low = np.min(all_rewards)
    reward_high = np.max(all_rewards)

    all_values = []
    for traj in trajectories:
        returns = compute_discounted_returns(traj['rewards'], cfg.discount)
        all_values.append(returns)
    all_values = np.concatenate(all_values, axis=0)
    value_low = np.min(all_values)
    value_high = np.max(all_values)

    obstacle_specs = [((x, y), r) for (x, y, r) in obstacles_list]

    obs_discretizer = AdaptiveDiscretizer(
        obstacle_specs, obs_low, obs_high, cfg.vocab_size,
        alpha=cfg.alpha, beta=cfg.beta
    )
    act_discretizer = UniformDiscretizer(act_low, act_high, cfg.vocab_size)
    reward_discretizer = UniformDiscretizer(reward_low, reward_high, cfg.vocab_size)
    value_discretizer = UniformDiscretizer(value_low, value_high, cfg.vocab_size)

    sequences = []
    goals = []

    for traj in trajectories:
        obs = traj['observations']
        acts = traj['actions']
        rewards = traj['rewards']
        values = compute_discounted_returns(rewards, cfg.discount)

        # 对齐长度
        if len(obs) == len(acts) + 1:
            goal_obs = obs[-1]
            obs = obs[:len(acts)]
        elif len(obs) == len(acts):
            goal_obs = obs[-1]
        else:
            min_len = min(len(obs), len(acts), len(rewards), len(values))
            warnings.warn(
                f"轨迹长度异常: obs={len(obs)}, acts={len(acts)}, "
                f"rewards={len(rewards)}, values={len(values)}，截断到 {min_len}"
            )
            obs = obs[:min_len]
            acts = acts[:min_len]
            rewards = rewards[:min_len]
            values = values[:min_len]
            goal_obs = traj['observations'][-1]

        # 逐维离散化
        obs_tokens = obs_discretizer.discretize(obs)            # (T, 2)
        act_tokens = act_discretizer.discretize(acts)           # (T, 2)
        rew_tokens = reward_discretizer.discretize(rewards).reshape(-1, 1)  # (T, 1)
        val_tokens = value_discretizer.discretize(values).reshape(-1, 1)    # (T, 1)

        # 每个时间步 6 个 token: [sx, sy, ax, ay, r, R]
        seq = np.concatenate([obs_tokens, act_tokens, rew_tokens, val_tokens], axis=1).reshape(-1)
        sequences.append(torch.from_numpy(seq).long())

        goal_tok = obs_discretizer.discretize_single(goal_obs)
        goals.append(torch.from_numpy(goal_tok).long())

    goals = torch.stack(goals, dim=0) if goals else torch.zeros(0, 2, dtype=torch.long)

    discretizers = {
        'obs': obs_discretizer,
        'act': act_discretizer,
        'reward': reward_discretizer,
        'value': value_discretizer,
        'vocab_size': cfg.vocab_size,
        'pad_token_id': cfg.vocab_size,   # 你的模型里 pad_token 用这个
    }
    return sequences, goals, discretizers


class GoalConditionalDataset(Dataset):
    """
    输入:
        seq: 1D token 序列（不要求已经 padding）
        goal: (2,)
    输出:
        seq, goal, targets, mask
    """
    def __init__(self, sequences, goals):
        self.sequences = sequences
        self.goals = goals

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]          # (L,)
        goal = self.goals[idx]             # (2,)

        L = seq.shape[0]
        targets = torch.full((L,), -100, dtype=torch.long)
        mask = torch.zeros((L,), dtype=torch.bool)

        # next-token 监督：最后一个 token 没有 next token，所以不参与 loss
        if L > 1:
            targets[:-1] = seq[1:]
            mask[:-1] = True

        return seq, goal, targets, mask


def collate_fn(batch, pad_token_id):
    """
    动态 padding，兼容不同长度轨迹。
    """
    seqs, goals, tgts, masks = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    batch_size = len(seqs)

    padded_seqs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_tgts = torch.full((batch_size, max_len), -100, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (s, t, m) in enumerate(zip(seqs, tgts, masks)):
        L = s.shape[0]
        padded_seqs[i, :L] = s
        padded_tgts[i, :L] = t
        padded_masks[i, :L] = m

    goals = torch.stack(goals, dim=0)
    return padded_seqs, goals, padded_tgts, padded_masks


def build_dataloader(sequences, goals, vocab_size, batch_size, shuffle=True):
    dataset = GoalConditionalDataset(sequences, goals)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=functools.partial(collate_fn, pad_token_id=vocab_size),
    )


def save_discretizers(discretizers, path):
    with open(path, 'wb') as f:
        pickle.dump(discretizers, f)


def load_discretizers(path):
    with open(path, 'rb') as f:
        return pickle.load(f)