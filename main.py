import torch
import os
import pickle
import time
import copy
from config import Config
from DDPG import Agent
from env import Env


class DDPGTrainer:
    def __init__(self):
        self.config = Config()
        self.env = Env()
        self.agent = Agent(self.env)
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
        """加载最新的检查点"""
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
            self.agent.actor_target.load_state_dict(
                checkpoint["actor_target_state_dict"]
            )
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

            print(f"Latest checkpoint {latest_checkpoint} restored!")
        else:
            print("No checkpoint found, starting fresh.")

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
            "actor_target_state_dict": self.agent.actor_target.state_dict(),
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
        start_time = time.time()
        scores = []
        total_steps = 0

        for episode in range(self.config.EPISODES):
            episode_score, episode_steps, episode_loss = self.run_training_episode()
            total_steps += episode_steps
            scores.append(episode_score)

            print(
                f"Episode: {episode}, Steps: {total_steps}, Epsilon: {self.agent.epsilon}, "
                f"Loss: {episode_loss}, Max score: {max(scores)}"
            )

            if episode % self.config.SAVE_FREQUENCY == 0:
                self.save_progress(scores, episode, total_steps)
                print(f"Total training time: {time.time() - start_time:.2f}s")

    def run_training_episode(self):
        state = self.env.reset()
        episode_start_time = time.time()
        total_loss = 0

        for step in range(self.config.MAX_STEPS):
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)

            self.agent.remember(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            total_loss += self.agent.replay()

            if done:
                print(
                    f"Episode completed in {step} steps, "
                    f"Time: {time.time() - episode_start_time:.2f}s"
                )
                return step, step + 1, total_loss / (step + 1)

        return (
            self.config.MAX_STEPS - 1,
            self.config.MAX_STEPS,
            total_loss / self.config.MAX_STEPS,
        )

    def test(self):
        scores = []

        for episode in range(self.config.TEST_EPISODES):
            episode_score = self.run_test_episode()
            scores.append(episode_score)

            print(
                f"Test Episode: {episode}, Score: {episode_score}, "
                f"Max Score: {max(scores)}"
            )

    def run_test_episode(self):
        state = self.env.reset()

        for step in range(self.config.TEST_MAX_STEPS):
            action = self.agent.actor(state)
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
