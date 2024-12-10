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


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(
        self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha  # 优先级的程度
        self.beta = beta  # 重要性采样的程度
        self.beta_increment = 0.001  # beta的增长率
        self.epsilon = 1e-6  # 防止优先级为0

        self.buffer: list[Experience] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加新经验"""
        exp = Experience(state, action, reward, next_state, done)

        # 新经验获得最高优先级
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0

        self.buffer[self.position] = exp
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """采样一个批次的经验"""
        # 计算采样概率
        priorities = self.priorities[: self.size]
        probs = priorities**self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)

        # 计算重要性权重
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)  # beta逐渐增加到1

        # 获取经验数据
        batch: list[Experience] = [self.buffer[idx] for idx in indices]
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
            indices,  # 返回索引用于更新优先级
            torch.FloatTensor(weights).unsqueeze(1),  # 返回权重用于TD误差加权
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon

    def __len__(self) -> int:
        return self.size


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_bound: float) -> None:
        super(Actor, self).__init__()
        self.action_bound = action_bound

        # 三层卷积结构
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )  # 14x14 -> 7x7

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, action_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))  # 提取基本特征
        x = F.relu(self.conv2(x))  # 第一次降采样
        x = F.relu(self.conv3(x))  # 第二次降采样

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = 2 * torch.sigmoid(self.fc2(x)) - 1

        return x * self.action_bound


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()

        # 状态编码器
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )  # 14x14 -> 7x7

        # 动作处理
        self.action_fc = nn.Linear(action_dim, 64)

        # 合并层
        self.fc1 = nn.Linear(64 * 7 * 7 + 64, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = state.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        # 处理动作
        a = F.relu(self.action_fc(action))

        # 合并
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        return value


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
        self.memory = PrioritizedReplayBuffer(memory_capacity)

        # 修改训练超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.001  # 软更新参数
        self.epsilon = 0.4  # 进一步增大初始探索率
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

        if np.random.rand() < self.epsilon:
            action = np.random.uniform(
                -self.action_bound, self.action_bound, size=self.action_dim
            )
            print("随机", end="")

        print("动作", action)
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
        """从优先经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return torch.tensor(0.0)

        # 采样带有权重的经验
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size)
        )

        # 计算TD误差和critic损失
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.gamma * target_q

        current_q = self.critic(states, actions)
        td_errors = (target - current_q).detach().numpy()

        # 更新优先级
        self.memory.update_priorities(indices, td_errors)

        # 使用重要性权重的critic损失
        critic_loss = (weights * F.mse_loss(current_q, target, reduction="none")).mean()

        # 更新网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新actor
        actor_loss = -(weights * self.critic(states, self.actor(states))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self._update_target_networks()

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return critic_loss

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
