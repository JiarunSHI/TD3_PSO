import os
import sys
import argparse
from functools import partial

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from TT_structure import GPT
from adaptive_discretizer1.adaptive_discretizer import prepare_discrete_dataset
from env.map_env import MapEnv


TYPE_GOAL = 0
TYPE_STATE = 1
TYPE_ACTION = 2
TYPE_REWARD = 3
TYPE_RETURN = 4
TYPE_PAD = 5


class UnifiedTrajectoryDataset(Dataset):
    """
    统一后的数据格式：

    原始离散序列 seq:
        [s_x, s_y, a_x, a_y, r, R] * T

    goal:
        [goal_x, goal_y]

    最终完整序列：
        [goal_x, goal_y] + seq + [goal_x, goal_y]

    这样 goal 既作为 prefix 条件出现，
    又保留在轨迹终点位置。
    """
    def __init__(self, sequences, goals, obs_dim, act_dim):
        self.samples = []
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.token_per_step = self.obs_dim + self.act_dim + 2

        for seq, goal in zip(sequences, goals):
            seq = torch.as_tensor(seq, dtype=torch.long)
            goal = torch.as_tensor(goal, dtype=torch.long)

            if seq.numel() < self.token_per_step:
                continue

            if seq.numel() % self.token_per_step != 0:
                raise ValueError(
                    f"sequence length must be multiple of {self.token_per_step}, got {seq.numel()}"
                )

            goal_dim = goal.numel()
            if goal_dim != self.obs_dim:
                raise ValueError(
                    f"goal dim ({goal_dim}) must equal obs_dim ({self.obs_dim})"
                )

            # goal 同时出现在前缀和终点
            full_seq = torch.cat([goal, seq, goal], dim=0)

            # type 序列
            step_types = (
                [TYPE_STATE] * self.obs_dim +
                [TYPE_ACTION] * self.act_dim +
                [TYPE_REWARD] +
                [TYPE_RETURN]
            )
            n_steps = seq.numel() // self.token_per_step
            traj_types = torch.tensor(step_types * n_steps, dtype=torch.long)

            prefix_types = torch.tensor([TYPE_GOAL] * self.obs_dim, dtype=torch.long)
            suffix_types = torch.tensor([TYPE_STATE] * self.obs_dim, dtype=torch.long)

            full_types = torch.cat([prefix_types, traj_types, suffix_types], dim=0)

            # next-token training
            X = full_seq[:-1]
            T = full_types[:-1]
            Y = full_seq[1:]

            # 前缀是 [goal_x, goal_y, start_x, start_y]
            # 所以前 3 个 target 不训练，第一条监督从 a_x 开始
            prefix_len = self.obs_dim + self.obs_dim
            loss_mask = torch.ones_like(Y, dtype=torch.float32)
            loss_mask[:prefix_len - 1] = 0.0

            self.samples.append((X, T, Y, loss_mask))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found.")

        self.max_len = max(x.numel() for x, _, _, _ in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch, pad_token_id: int):
    xs, ts, ys, masks = zip(*batch)

    max_len = max(x.numel() for x in xs)
    batch_size = len(xs)

    X = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    T = torch.full((batch_size, max_len), TYPE_PAD, dtype=torch.long)
    Y = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    ATTN = torch.zeros((batch_size, max_len), dtype=torch.bool)
    LOSS = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, (x, t, y, m) in enumerate(zip(xs, ts, ys, masks)):
        n = x.numel()
        X[i, :n] = x
        T[i, :n] = t
        Y[i, :n] = y
        ATTN[i, :n] = True
        LOSS[i, :n] = m

    return X, T, Y, ATTN, LOSS


def build_parser():
    parser = argparse.ArgumentParser(description="Train unified GPT on goal + trajectory")

    parser.add_argument('--trajectory_pkl', type=str, required=True, help='trajectory_dataset.pkl path')
    parser.add_argument('--obs_dim', type=int, default=2)
    parser.add_argument('--act_dim', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=100)

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument(
        '--global_obs_bounds', type=float, nargs=4, default=[0, 0, 20, 20],
        help='观测空间全局范围 [low_x low_y high_x high_y]'
    )

    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) obstacles
    env = MapEnv()
    obstacles = env.generate_mixed()
    print(f"障碍物数量: {len(obstacles)}")

    # 2) discretization config
    class Config:
        pass

    cfg = Config()
    cfg.obs_dim = args.obs_dim
    cfg.act_dim = args.act_dim
    cfg.vocab_size = args.vocab_size
    cfg.discount = args.discount
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.global_obs_bounds = [
        [args.global_obs_bounds[0], args.global_obs_bounds[1]],
        [args.global_obs_bounds[2], args.global_obs_bounds[3]],
    ]

    # 3) load & discretize dataset
    sequences, goals, discretizers = prepare_discrete_dataset(
        args.trajectory_pkl, obstacles, cfg
    )

    print(f"加载轨迹数量: {len(sequences)}")
    print(f"示例序列长度: {len(sequences[0])} tokens")
    print(f"示例 goal tokens: {goals[0].tolist()}")

    pad_token_id = int(discretizers['pad_token_id'])
    assert pad_token_id == args.vocab_size, (
        f"pad_token_id({pad_token_id}) 必须等于 vocab_size({args.vocab_size})"
    )

    # 4) dataset
    dataset = UnifiedTrajectoryDataset(
        sequences=sequences,
        goals=goals,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_batch, pad_token_id=pad_token_id),
    )

    # 5) model config
    model_config = {
        'vocab_size': args.vocab_size,
        'type_size': 6,  # GOAL / STATE / ACTION / REWARD / RETURN / PAD
        'block_size': dataset.max_len,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'embd_pdrop': args.embd_pdrop,
        'resid_pdrop': args.resid_pdrop,
        'attn_pdrop': args.attn_pdrop,
    }

    class ModelConfig:
        pass

    model_cfg = ModelConfig()
    for k, v in model_config.items():
        setattr(model_cfg, k, v)

    model = GPT(model_cfg).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 6) optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    # 7) train
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (X, T, Y, attn_mask, loss_mask) in enumerate(dataloader):
            X = X.to(device)
            T = T.to(device)
            Y = Y.to(device)
            attn_mask = attn_mask.to(device)
            loss_mask = loss_mask.to(device)

            logits, loss = model(
                idx=X,
                type_idx=T,
                targets=Y,
                attn_mask=attn_mask,
                loss_mask=loss_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            global_step += 1

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {batch_idx} | Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")

        # 8) save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': model_config,
                'discretizers': discretizers,
                'pad_token_id': pad_token_id,
                'obs_dim': args.obs_dim,
                'act_dim': args.act_dim,
                'vocab_size': args.vocab_size,
            }, save_path)
            print(f"模型已保存至 {save_path}")

    print("训练完成！")


if __name__ == "__main__":
    main()