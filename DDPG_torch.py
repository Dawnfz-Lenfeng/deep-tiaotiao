from dataclasses import dataclass
from typing import Tuple, List
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.buffer: List[Experience] = [None] * capacity  # 预分配列表空间
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一个批次的经验"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch: List[Experience] = [self.buffer[i] for i in indices]

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


class Actor(nn.Module):
    """Actor 网络"""
    def __init__(self, state_dim: int, action_dim: int, action_bound: float) -> None:
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.average_pooling_2d = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2d3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(0)
        x = self.average_pooling_2d(state)
        x = F.relu(self.conv2d1(x))
        x = F.relu(self.conv2d2(x))
        x = F.relu(self.conv2d3(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x * self.action_bound


class Critic(nn.Module):
    """Critic 网络"""
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()
        self.average_pooling_2d = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2d3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(0)
        state = self.average_pooling_2d(state)
        state = F.relu(self.conv2d1(state))
        state = F.relu(self.conv2d2(state))
        state = F.relu(self.conv2d3(state))
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class Agent:
    """DDPG智能体"""

    def __init__(self, env: Env, memory_capacity: int = 10000, batch_size: int = 64) -> None:
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

        # 训练超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.001  # 软更新参数
        self.epsilon = 1.0  # 探索噪声
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]
        noise = np.random.normal(0, self.epsilon, size=self.action_dim)
        action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action

    def remember(
        self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self) -> torch.Tensor:
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return torch.tensor(0.0)

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

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
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """更新Critic网络"""
        # 计算目标值
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_action)
            target = rewards + (1 - dones) * self.gamma * target_q

        # 计算Critic损失
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)

        # 优化Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
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
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


