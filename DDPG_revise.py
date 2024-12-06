from dataclasses import dataclass
import time
import copy
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers

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
        indices = np.random.choice(self.size, batch_size, replace=False)
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


class Actor(models.Model):
    """Actor网络"""

    def __init__(self, state_dim: int, action_dim: int, action_bound: float) -> None:
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.layer1 = layers.Dense(400, activation="relu")
        self.layer2 = layers.Dense(300, activation="relu")
        self.layer3 = layers.Dense(action_dim, activation="tanh")

    def call(self, state: tf.Tensor) -> tf.Tensor:
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        return x * self.action_bound


class Critic(models.Model):
    """Critic网络"""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()
        self.layer1 = layers.Dense(400, activation="relu")
        self.layer2 = layers.Dense(300, activation="relu")
        self.layer3 = layers.Dense(1)

    def call(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        x = tf.concat([state, action], axis=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


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
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # 优化器
        self.actor_optimizer = optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = optimizers.Adam(learning_rate=1e-3)

        # 经验回放
        self.memory = ReplayBuffer(memory_capacity)

        # 训练超参数
        self.gamma: float = 0.99  # 折扣因子
        self.tau: float = 0.001  # 软更新参数
        self.epsilon: float = 1.0  # 探索噪声
        self.epsilon_decay: float = 0.995
        self.epsilon_min: float = 0.01

    def act(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0]
        noise = np.random.normal(0, self.epsilon, size=self.action_dim)
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

    def replay(self) -> tf.Tensor:
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return tf.constant(0.0)

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # 更新网络
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        self._update_actor(states)
        self._update_target_networks()

        # 衰减探索噪声
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return critic_loss


if __name__ == "__main__":
    env = Env()
    agent = Agent(env)
    checkpoint_path = "./checkpoints/ddpg_checkpoint"
    ckpt = tf.train.Checkpoint(actor=agent.actor, actor_target=agent.actor_target, critic=agent.critic,
                               critic_target=agent.critic_target)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    episodes = 10
    steps = 0
    for e in range(episodes):
        state = env.reset()
        start_time = time.time()

        loss = 0
        for time_t in range(100000):

            # 选择行为
            action = agent.act(state)

            # 在环境中施加行为推动游戏进行
            next_state, reward, done = env.step(action)
            # get_env.touch_in_step(action)

            # 记忆先前的状态，行为，回报与下一个状态
            agent.remember(state, action, reward, next_state, done)

            # 使下一个状态成为下一帧的新状态
            state = copy.deepcopy(next_state)

            loss += agent.replay()

            # 如果游戏结束done被置为ture
            # 除非agent没有完成目标
            if done:
                # 打印分数并且跳出游戏循环
                print("Epoch: {}/{}, use time: {}".format(e + 1, episodes, time.time() - start_time))
                break

        steps += (time_t + 1)
        loss /= (time_t + 1)

        print('Epoch:', e, 'step:', steps, 'epsilon:', agent.epsilon, 'loss:', loss.numpy())
        # ckpt_save_path = ckpt_manager.save()
        print("\n")
