import numpy as np
import torch
import torch.nn.functional as F
from drl.actor import Actor
from drl.critic import Critic

class TD3:
    def __init__(self, device, state_dim, action_dim=3, gamma=0.97, tau=0.005, lr=1e-3,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, hidden=128, seed=None,
                 actor_lr=None, critic_lr=None):
        """
        动作空间为 [-1, 1]^action_dim。
        :param seed: 网络初始权重和目标策略平滑噪声的随机种子；None 表示不固定
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0
        self.last_train_metrics = {name: float("nan") for name in (
            "td_reward_mean", "td_reward_abs_mean", "td_bootstrap_mean", "td_bootstrap_abs_mean",
            "td_target_mean", "td_q1_mean", "td_q2_mean", "td_actor_loss")}

        if seed is not None:
            torch.manual_seed(seed)

        # 在线网络
        self.actor = Actor(state_dim, action_dim, hidden).to(device)
        self.c1 = Critic(state_dim, action_dim, hidden).to(device)
        self.c2 = Critic(state_dim, action_dim, hidden).to(device)

        # 目标网络
        self.actor_t = Actor(state_dim, action_dim, hidden).to(device)
        self.c1_t = Critic(state_dim, action_dim, hidden).to(device)
        self.c2_t = Critic(state_dim, action_dim, hidden).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t.load_state_dict(self.c2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr if actor_lr is None else actor_lr)
        self.opt_c = torch.optim.Adam(
            list(self.c1.parameters()) + list(self.c2.parameters()),
            lr=lr if critic_lr is None else critic_lr
        )

    def soft_update(self):
        for net, net_t in ((self.actor, self.actor_t), (self.c1, self.c1_t), (self.c2, self.c2_t)):
            for param, target_param in zip(net.parameters(), net_t.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

    @torch.no_grad()
    def act(self, states):
        """确定性策略输出，states: (N, state_dim) numpy -> (N, action_dim) numpy，范围 [-1, 1]"""
        s = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        return self.actor(s).cpu().numpy()

    @torch.no_grad()
    def pre_tanh(self, states):
        s = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        return self.actor.pre_tanh(s).cpu().numpy()

    def train_td3(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return None

        self.total_it += 1
        s, a, r, s_ = buffer.sample(batch_size, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_ = (self.actor_t(s_) + noise).clamp(-1.0, 1.0)
            q_t = torch.min(self.c1_t(s_, a_), self.c2_t(s_, a_))
            # 持续在线优化：--iters 只是实验截断，不是任务终止，末步仍 bootstrap。
            target = r + self.gamma * q_t

        q1, q2 = self.c1(s, a), self.c2(s, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.last_train_metrics.update(
            td_reward_mean=float(r.mean().item()), td_reward_abs_mean=float(r.abs().mean().item()),
            td_bootstrap_mean=float((self.gamma * q_t).mean().item()),
            td_bootstrap_abs_mean=float((self.gamma * q_t).abs().mean().item()),
            td_target_mean=float(target.mean().item()), td_q1_mean=float(q1.detach().mean().item()),
            td_q2_mean=float(q2.detach().mean().item()), td_actor_loss=float("nan"))
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        if self.total_it % self.policy_delay == 0:
            loss_a = -self.c1(s, self.actor(s)).mean()
            self.last_train_metrics["td_actor_loss"] = float(loss_a.detach().item())
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()
            self.soft_update()

        return float(loss_c.item())

    def save_model(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.c1.state_dict(), f"{path_prefix}_critic1.pth")
        torch.save(self.c2.state_dict(), f"{path_prefix}_critic2.pth")
        print(f"Model saved to {path_prefix}_*.pth")

    def load_model(self, path_prefix):
        weights = [torch.load(f"{path_prefix}_{name}.pth", map_location=self.device, weights_only=True)
                   for name in ("actor", "critic1", "critic2")]
        # 全部验证后再加载，避免旧状态维度的模型部分加载成功。
        for name, net, state in zip(("actor", "critic1", "critic2"),
                                    (self.actor, self.c1, self.c2), weights):
            expected = net.state_dict()
            if state.keys() != expected.keys() or any(state[k].shape != v.shape for k, v in expected.items()):
                raise ValueError(f"Incompatible {name} checkpoint: state/action schema or hidden size changed. "
                                 f"Current actor expects {self.actor.net[0].in_features} state features; "
                                 "retrain a matching model; checkpoints with a different feature schema cannot be reused.")
        self.actor.load_state_dict(weights[0])
        self.c1.load_state_dict(weights[1])
        self.c2.load_state_dict(weights[2])
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t.load_state_dict(self.c2.state_dict())
        print(f"Model loaded from {path_prefix}_*.pth")
