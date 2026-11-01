import torch
import torch.nn.functional as F
from drl.actor import Actor
from drl.critic import Critic

class TD3:
    def __init__(self, device, action_bound=1.0):
        """
        :param device: torch设备 (cpu/cuda)
        :param action_bound: 动作边界，假设动作空间对称，如[-bound, bound]
        """
        self.device = device
        self.action_bound = action_bound

        # 在线网络
        self.actor = Actor().to(device)
        self.c1 = Critic().to(device)
        self.c2 = Critic().to(device)

        # 目标网络
        self.actor_t = Actor().to(device)
        self.c1_t = Critic().to(device)
        self.c2_t = Critic().to(device)

        # 初始化目标网络参数与在线网络一致
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t.load_state_dict(self.c2.state_dict())

        # 优化器
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.opt_c = torch.optim.Adam(
            list(self.c1.parameters()) + list(self.c2.parameters()),
            lr=1e-3
        )

        # TD3超参数
        self.gamma = 0.99
        self.tau = 0.005          # 目标网络软更新系数
        self.policy_noise = 0.1   # 目标策略噪声标准差
        self.noise_clip = 0.2     # 噪声裁剪范围
        self.total_it = 0

    def soft_update(self):
        """软更新目标网络参数"""
        for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.c1.parameters(), self.c1_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.c2.parameters(), self.c2_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_td3(self, buffer, batch_size=256):
        """
        从缓冲区采样并更新网络

        :param batch_size: 批量大小
        """
        # 缓冲区样本不足，跳过训练
        if len(buffer.buffer) < batch_size:
            return

        self.total_it += 1

        # 采样一个batch
        s, a, r, s_ = buffer.sample(batch_size, self.device)

        # ---------- 目标动作与噪声 ----------
        with torch.no_grad():
            # 目标动作加噪声
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_ = self.actor_t(s_) + noise
            # 裁剪动作到合法范围
            a_ = a_.clamp(-self.action_bound, self.action_bound)

            # 目标Q值（取两个Critic的最小值）
            q1_t = self.c1_t(s_, a_)
            q2_t = self.c2_t(s_, a_)
            q_t = torch.min(q1_t, q2_t)
            target = r + self.gamma * q_t

        # ---------- 更新Critic网络 ----------
        q1 = self.c1(s, a)
        q2 = self.c2(s, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        # ---------- 延迟更新Actor网络 ----------
        if self.total_it % 2 == 0:
            # 使用当前Critic的最小值计算Actor损失
            a_pred = self.actor(s)
            q1_a = self.c1(s, a_pred)
            q2_a = self.c2(s, a_pred)
            loss_a =-q1_a.mean()

            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()

            # 每次Actor更新后，软更新目标网络
            self.soft_update()

    def save_model(self, path_prefix):
        """保存模型参数"""
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.c1.state_dict(), f"{path_prefix}_critic1.pth")
        torch.save(self.c2.state_dict(), f"{path_prefix}_critic2.pth")
        # 也可保存优化器状态等（可选）
        print(f"Model saved to {path_prefix}_*.pth")

    def load_model(self, path_prefix):
        """加载模型参数"""
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth", map_location=self.device, weights_only=True))
        self.c1.load_state_dict(torch.load(f"{path_prefix}_critic1.pth", map_location=self.device, weights_only=True))
        self.c2.load_state_dict(torch.load(f"{path_prefix}_critic2.pth", map_location=self.device, weights_only=True))
        # 如果需要，也可以加载目标网络（建议复制在线网络参数）
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t.load_state_dict(self.c2.state_dict())
        print(f"Model loaded from {path_prefix}_*.pth")