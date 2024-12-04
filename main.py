import copy
import pickle
import time

import tensorflow as tf

from config import Config
from DDPG import Agent
from env import Env


class DDPGTrainer:
    def __init__(self):
        self.config = Config()
        self.env = Env()
        self.agent = Agent(self.env)
        self.setup_checkpointing()
        self.load_memory()

    def setup_checkpointing(self):
        """设置检查点"""
        self.ckpt = tf.train.Checkpoint(
            actor=self.agent.actor,
            actor_target=self.agent.actor_target,
            critic=self.agent.critic,
            critic_target=self.agent.critic_target,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.config.CHECKPOINT_PATH, max_to_keep=5
        )

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!")

    def load_memory(self):
        if self.config.MEMORY_FILE.exists():
            with open(self.config.MEMORY_FILE, "rb") as f:
                self.agent.memory = pickle.load(f)

    def save_progress(self, scores: list[int], episode: int, steps: int):
        save_data = (scores, episode, steps)
        with self.config.SCORES_FILE.open("wb") as f:
            pickle.dump(save_data, f)
        with self.config.MEMORY_FILE.open("wb") as f:
            pickle.dump(self.agent.memory, f)
        self.ckpt_manager.save()

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
        state = self.env.reset(is_show=False)
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
        state = self.env.reset(is_show=False)

        for step in range(self.config.TEST_MAX_STEPS):
            action = self.agent.actor(state)
            next_state, reward, done = self.env.touch_in_step(action)
            state = copy.deepcopy(next_state)

            if done:
                return step

        return self.config.TEST_MAX_STEPS - 1


def main():
    trainer = DDPGTrainer()

    # Uncomment the one you want to run
    # trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
