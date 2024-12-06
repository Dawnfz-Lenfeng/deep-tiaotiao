from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import Env


@dataclass
class Experience:
    """经验数据"""

    state: np.ndarray  # 当前截图状态
    action: float  # 按压时间
    reward: float  # 奖励 +1/-1
    next_state: np.ndarray  # 下一个截图状态
    done: bool  # 是否结束


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: list[Experience] = [None] * capacity  # 预分配列表空间
        self.position = 0  # 当前插入位置
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加一个新的经验到缓冲区"""
        exp = Experience(state, action, reward, next_state, done)
        self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一个批次的经验"""
        indices = np.random.choice(self.size, batch_size, replace=True)
        batch: list[Experience] = [self.buffer[i] for i in indices]

        # 拆分批次数据
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """返回当前缓冲区中的经验数量"""
        return self.size


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int, size: int = 28):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([channels, size, size])
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([channels, size, size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.ln1(self.conv1(x)))
        x = self.ln2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class Actor(nn.Module):
    """Actor 网络"""

    def __init__(self, state_dim: int, action_dim: int, action_bound: float) -> None:
        super(Actor, self).__init__()
        self.action_bound = action_bound

        # 简化的卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # 简化的全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, action_dim)

        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))

        # 使用sigmoid将输出映射到[0,1]，然后转换到[-1,1]
        x = 2 * torch.sigmoid(self.fc2(x)) - 1
        return x * self.action_bound


class Critic(nn.Module):
    """Critic 网络"""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()

        # 状态编码器
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # 动作处理
        self.action_fc = nn.Linear(action_dim, 64)

        # 合并层
        self.fc1 = nn.Linear(64 * 7 * 7 + 64, 200)
        self.fc2 = nn.Linear(200, 1)

        # 添加动作正则化层
        self.action_regularizer = nn.Linear(action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 处理状态
        x = state.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        # 处理动作
        a = F.relu(self.action_fc(action))

        # 合并
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        # 添加动作惩罚项
        action_penalty = -torch.abs(self.action_regularizer(action))

        return value + action_penalty


class Agent:
    """DDPG智能体"""

    def __init__(
        self, env: Env, memory_capacity: int = 10000, batch_size: int = 64
    ) -> None:
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_bound = env.action_bound
        self.batch_size = batch_size

        # 初始化网络
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic_target = Critic(self.state_dim, self.action_dim)

        # 复制参数到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 经验回放
        self.memory = ReplayBuffer(memory_capacity)

        # 修改训练超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.001  # 软更新参数
        self.epsilon = 0.1  # 进一步增大初始探索率
        self.epsilon_decay = 0.995  # 降低衰减速率
        self.epsilon_min = 0.01  # 提高最小探索值

        # 减小动作噪声
        self.action_noise_std = 0.1  # 降低噪声标准差
        self.action_noise_clip = 0.3  # 降低噪声裁剪范围

    def act(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        self.actor.eval()

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]

        self.actor.train()

        # 在当前策略基础上添加噪声
        noise = np.random.normal(0, self.action_noise_std, size=self.action_dim)
        noise = np.clip(noise, -self.action_noise_clip, self.action_noise_clip)
        action = np.clip(action + noise, -self.action_bound, self.action_bound)

        return action

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self) -> torch.Tensor:
        """从经验回放中学习"""
        if len(self.memory) < 2:  # 至少需要2个样本
            return torch.tensor(0.0)

        # 使用所有可用样本，但不超过batch_size
        actual_batch_size = min(len(self.memory), self.batch_size)

        states, actions, rewards, next_states, dones = self.memory.sample(
            actual_batch_size
        )

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # 更新网络
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        self._update_actor(states)
        self._update_target_networks()

        # 衰减探索噪声
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return critic_loss

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """更新Critic网络"""
        # 计算目标值
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # 给目标动作添加噪声以增加鲁棒性
            noise = torch.normal(0, self.action_noise_std, next_actions.shape).clamp(
                -self.action_noise_clip, self.action_noise_clip
            )
            next_actions = (next_actions + noise).clamp(
                -self.action_bound, self.action_bound
            )

            target_q = self.critic_target(next_states, next_actions)
            target = rewards + (1 - dones) * self.gamma * target_q

        # 计算Critic损失
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)

        # 优化Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss

    def _update_actor(self, states: torch.Tensor) -> None:
        """更新Actor网络"""
        # 计算Actor的梯度
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # 优化Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _update_target_networks(self) -> None:
        """软更新目标网络"""
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def pretrain(self, num_episodes: int = 10) -> None:
        """预训练阶段，收集一些初始经验"""
        print("Starting pretrain phase...")
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                # 随机动作
                action = np.random.uniform(
                    -self.action_bound, self.action_bound, self.action_dim
                )
                next_state, reward, done = self.env.step(action)
                # 存储经验
                self.remember(state, action, reward, next_state, done)
                state = next_state
            print(f"Pretrain episode {episode + 1}/{num_episodes} complete.")
        print(f"Pretrain complete. Collected {len(self.memory)} experiences.")
