import copy
import csv
import os
import pickle
import time

import torch

from config import Config
from DDPG import Agent
from env import Env


class DDPGTrainer:
    def __init__(self):
        self.config = Config()
        self.env = Env()
        self.agent = Agent(self.env)
        self.current_episode = 0
        self.setup_checkpointing()
        self.load_checkpoint()

    def setup_checkpointing(self):
        """设置检查点"""
        self.checkpoint_path = self.config.CHECKPOINT_PATH
        self.max_to_keep = 5

        # 确保检查点目录存在，如果不存在则创建
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
            print(f"Checkpoint directory {self.checkpoint_path} created.")

        # 尝试加载最新的检查点
        self.load_checkpoint()

    def load_checkpoint(self):
        """加载最新的检查点和经验回放"""
        # 加载模型检查点
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pth")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )

        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(
                os.path.join(self.checkpoint_path, latest_checkpoint)
            )

            self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.agent.critic_target.load_state_dict(
                checkpoint["critic_target_state_dict"]
            )
            self.agent.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.agent.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )

            # 获取当前episode
            self.current_episode = int(latest_checkpoint.split("_")[1].split(".")[0])
            print(f"Resuming from episode {self.current_episode}")
            print(f"Latest checkpoint {latest_checkpoint} restored!")
        else:
            print("No checkpoint found, starting fresh.")

        # 加载经验回放缓冲区
        if self.config.MEMORY_FILE.exists():
            try:
                with self.config.MEMORY_FILE.open("rb") as f:
                    self.agent.memory = pickle.load(f)
                print(f"Loaded {len(self.agent.memory)} experiences from memory file")
            except Exception as e:
                print(f"Failed to load memory file: {e}")
                print("Starting with empty memory")

    def save_progress(self, scores: list[int], episode: int, steps: int):
        save_data = (scores, episode, steps)

        # 保存 scores 和 memory
        with self.config.SCORES_FILE.open("wb") as f:
            pickle.dump(save_data, f)
        with self.config.MEMORY_FILE.open("wb") as f:
            pickle.dump(self.agent.memory, f)

        # 保存模型和优化器
        checkpoint = {
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "critic_target_state_dict": self.agent.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
        }

        # 保持最多 max_to_keep 个检查点
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pth")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if len(checkpoint_files) >= self.max_to_keep:
            os.remove(os.path.join(self.checkpoint_path, checkpoint_files[0]))

        checkpoint_file = os.path.join(
            self.checkpoint_path, f"checkpoint_{episode}.pth"
        )
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")

    def train(self):
        best_score = float("-inf")
        no_improve = 0
        scores = []
        start_time = time.time()
        total_steps = 0

        csv_file = "training_progress.csv"

        # 检测CSV文件是否存在，若不存在则创建并写入表头
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Episode",
                        "Total Steps",
                        "Total Reward",
                        "Episode Loss",
                        "Episode Score",
                        "Best Score",
                    ]
                )

        # 只在经验不足时进行预训练
        if len(self.agent.memory) < self.agent.batch_size:
            print(
                f"Insufficient experiences ({len(self.agent.memory)} < {self.agent.batch_size})"
            )
        else:
            print(f"Found {len(self.agent.memory)} experiences, skipping pretrain")

        # 从当前episode继续训练
        for episode in range(self.current_episode, self.config.EPISODES):
            (
                episode_score,
                episode_steps,
                episode_loss,
                total_reward,
            ) = self.run_training_episode()
            total_steps += episode_steps
            scores.append(episode_score)

            # 检查是否有改进
            if episode_score > best_score:
                best_score = episode_score
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"Episode: {episode}, Steps: {total_steps}, Epsilon: {self.agent.epsilon}, "
                f"Reward: {total_reward}, Loss: {episode_loss}, Score: {episode_score}, "
                f"Best: {best_score}"
            )

            # 将当前训练数据写入CSV文件
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        episode,
                        total_steps,
                        total_reward,
                        episode_loss,
                        episode_score,
                        best_score,
                    ]
                )

            if episode % self.config.SAVE_FREQUENCY == 0:
                self.save_progress(scores, episode, total_steps)
                print(f"Total training time: {time.time() - start_time:.2f}s")

    def run_training_episode(self):
        state = self.env.reset()
        episode_start_time = time.time()
        total_loss = 0
        total_reward = 0.0

        for step in range(self.config.MAX_STEPS):
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)

            self.agent.remember(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            total_reward += reward
            total_loss += self.agent.replay()

            if done:
                print(
                    f"Episode completed in {step} steps, "
                    f"Time: {time.time() - episode_start_time:.2f}s"
                )
                return step, step + 1, total_loss / (step + 1), total_reward

        return (
            self.config.MAX_STEPS - 1,
            self.config.MAX_STEPS,
            total_loss / self.config.MAX_STEPS,
            total_reward,
        )

    def test(self):
        # 设置为评估模式
        self.agent.eval()

        scores = []

        csv_file = "test_progress.csv"

        # 检测CSV文件是否存在，若不存在则创建并写入表头
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Test Episode", "Score", "Max Score"])

        for episode in range(self.config.TEST_EPISODES):
            episode_score = self.run_test_episode()
            scores.append(episode_score)

            print(
                f"Test Episode: {episode}, Score: {episode_score}, "
                f"Max Score: {max(scores)}"
            )

            # 将当前测试数据写入CSV文件
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode, episode_score, max(scores)])

    def run_test_episode(self):
        state = self.env.reset()

        for step in range(self.config.TEST_MAX_STEPS):
            # 直接使用actor网络输出，不需要调用agent.act()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.agent.actor(state_tensor).numpy()[0]
            next_state, reward, done = self.env.step(action)
            state = copy.deepcopy(next_state)

            if done:
                return step

        return self.config.TEST_MAX_STEPS - 1


def main():
    trainer = DDPGTrainer()

    # Uncomment the one you want to run
    trainer.train()
    # trainer.test()


if __name__ == "__main__":
    main()
