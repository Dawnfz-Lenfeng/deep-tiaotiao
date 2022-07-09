import tensorflow as tf
import time
import copy
import os
import pickle
from DDPG import Agent
from GetEnv import GetEnv

file_memory = "./data/ddpg_mem.p"
file_score = "./data/ddpg_score.p"

if __name__ == '__main__':
    get_env = GetEnv()

    agent = Agent(get_env)

    if os.path.exists(file_memory):
        with open(file_memory, "rb") as f:
            agent.memory = pickle.load(f)

    checkpoint_path = "./checkpoints/ddpg_checkpoint"
    ckpt = tf.train.Checkpoint(actor=agent.actor, actor_target=agent.actor_target, critic=agent.critic, critic_target=agent.critic_target)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    scores = []
    episodes = 1000
    steps = 0
    for e in range(episodes):
        state = get_env.reset(is_show=False)
        start_time = time.time()
        
        loss = 0
        for time_t in range(100000):

            # 选择行为
            action = agent.act(state)

            # 在环境中施加行为推动游戏进行
            next_state, reward, done = get_env.touch_in_step(action)
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
                print("Epoch: {}/{}, score: {}, use time: {}".format(e + 1, episodes, time_t, time.time() - start_time))
                scores.append(time_t)
                break

        steps += (time_t + 1)
        loss /= (time_t + 1)

        print('Epoch:', e, 'step:', steps, 'epsilon:', agent.epsilon, 'loss:', loss.numpy(), 'Max score', max(scores))
        print("\n")

        # if e % 10 == 0 and steps != 0:
        #     save_data = (scores, e, steps)
        #     pickle.dump(save_data, open(file_score,'wb'))
        #     pickle.dump(agent.memory, open(file_memory,'wb'))

        #     ckpt_save_path = ckpt_manager.save()