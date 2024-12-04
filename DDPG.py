import copy
import time

import numpy as np
import tensorflow as tf

from GetEnv_direct import GetEnv


class ActorNetwork(tf.keras.Model):
    """
    输出动作：在这里就是按压时长秒数
    """

    def __init__(self):
        super().__init__()

        self.average_polling_2d = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(2, 2), padding="same"
        )
        self.conv2d_1 = tf.keras.layers.Conv2D(
            32, (8, 8), strides=(2, 2), padding="same"
        )
        self.activation_1 = tf.keras.layers.Activation("relu")

        self.conv2d_2 = tf.keras.layers.Conv2D(
            32, (4, 4), strides=(2, 2), padding="same"
        )
        self.activation_2 = tf.keras.layers.Activation("relu")

        self.conv2d_3 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="same"
        )
        self.activation_3 = tf.keras.layers.Activation("relu")

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(200, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(200, activation="relu")

        self.outputs = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, inputs, training=False):

        x = inputs[:, :, :, :]

        x = self.average_polling_2d(x)

        x = self.conv2d_1(x)
        x = self.activation_1(x)

        x = self.conv2d_2(x)
        x = self.activation_2(x)

        x = self.conv2d_3(x)
        x = self.activation_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        outputs = self.outputs(x)

        return outputs


class CriticNetwork(tf.keras.Model):
    """
    输入：next_state, next_action
    输出：评估其状态的价值
    """

    def __init__(self):
        super().__init__()

        self.average_polling_2d = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(2, 2), padding="same"
        )
        self.conv2d_1 = tf.keras.layers.Conv2D(
            32, (8, 8), strides=(2, 2), padding="same"
        )
        self.activation_1 = tf.keras.layers.Activation("relu")

        self.conv2d_2 = tf.keras.layers.Conv2D(
            32, (4, 4), strides=(2, 2), padding="same"
        )
        self.activation_2 = tf.keras.layers.Activation("relu")

        self.conv2d_3 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="same"
        )
        self.activation_3 = tf.keras.layers.Activation("relu")

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(200, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(200, activation="relu")

        # input action
        self.dense_action_1 = tf.keras.layers.Dense(200, activation="relu")
        self.add_action_1 = tf.keras.layers.Add()

        # final
        self.outputs = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs_state, input_action, training=False):

        # input state
        x = inputs_state[:, :, :, :]

        x = self.average_polling_2d(x)

        x = self.conv2d_1(x)
        x = self.activation_1(x)

        x = self.conv2d_2(x)
        x = self.activation_2(x)

        x = self.conv2d_3(x)
        x = self.activation_3(x)

        x = self.flatten(x)
        x_s = self.dense_1(x)

        # input action
        x_a = self.dense_action_1(input_action)

        x = self.add_action_1([x_s, x_a])

        x = self.dense_2(x)
        outputs = self.outputs(x)

        return outputs


loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


def loss_function(real, pred):
    loss_ = loss_object(real, pred)

    return tf.reduce_mean(loss_)


class Agent:
    """
    function:
    初始化网络
        1.网络预测
        2.回放经验计算loss
    """

    def __init__(self, env):
        self.env = env
        # 记录
        self.memory = []

        self.gamma = 0.99
        self.TAU = 0.001  # Target Network HyperParameters
        self.lr_actor = 0.0001  # Learning rate for Actor
        self.lr_critic = 0.001  # Learning rate for Critic

        self.episode_count = 2000
        self.max_steps = 100000
        self.reward = 0
        self.done = False
        self.step = 0
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.indicator = 0

        self.batch_size = 32

        self.mem_len = 200

        # self.actor = ActorNetwork(self.sess, self.BATCH_SIZE, self.TAU, self.LRA)
        # self.critic = CriticNetwork(self.sess, self.BATCH_SIZE, self.TAU, self.LRC)

        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr_critic)

        # 创建4个网络
        self.actor = ActorNetwork()
        self.actor_target = ActorNetwork()
        self.critic = CriticNetwork()
        self.critic_target = CriticNetwork()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def add_noise(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)[0]

    def act(self, state):
        act = self.actor(state)

        noise = self.epsilon * self.add_noise(act.numpy()[0, 0], 0, 0.5, 0.5)
        # noise = 0

        print("act:", act.numpy(), "noise:", noise, "sum:", (act + noise).numpy())

        return act + noise

    def target_train(self, raw_model, target_model):
        """
        目标网络：从当前网络的参数进行复制更新
        """
        critic_weights = raw_model.get_weights()
        critic_target_weights = target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = (
                self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
            )
        target_model.set_weights(critic_target_weights)

    def replay(self):
        """
        经验回放
        """
        if len(self.memory) < 64:
            self.mem_len = 32
        elif len(self.memory) < 128:
            self.mem_len = 64
        elif len(self.memory) < 256:
            self.mem_len = 128
        elif len(self.memory) < 512:
            self.mem_len = 256
        else:
            self.mem_len = 512

        mem = self.memory[-self.mem_len :]

        n_batch = min(self.batch_size, len(mem))
        batches = np.random.choice(len(mem), n_batch)

        states = []
        actions = []
        rewards = []
        new_states = []
        dones = []

        for i, mem_idx in enumerate(batches):
            state, action, reward, next_state, done = mem[mem_idx]

            if done:
                next_state = np.zeros_like(state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(next_state)
            dones.append(done)

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.array(rewards)
        new_states = np.concatenate(new_states, axis=0)

        # Critic目标网络，使用所有的状态，计算每一步的动作和Q值
        target_q_values = self.critic_target(new_states, self.actor_target(new_states))

        y_t = []
        # 计算多轮以来的rewards奖励
        for k, d in enumerate(dones):
            if d:
                y_t.append(rewards[k])
            else:
                y_t.append(rewards[k] + self.gamma * target_q_values[k].numpy())

        y_t = tf.constant(np.float32(np.reshape(y_t, [-1, 1])))
        # print(y_t)
        # print(target_q_values)

        # 1.更新crtic网络
        loss = 0
        with tf.GradientTape() as critic_tape:
            ## 计算当前状态 与 下一状态的差值
            pre_t = self.critic(states, actions)
            loss = loss_function(y_t, pre_t)
        # print(pre_t)
        critic_trainable_variables = self.critic.trainable_variables
        gradients = critic_tape.gradient(loss, critic_trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, critic_trainable_variables)
        )

        # 2.更新actor网络
        with tf.GradientTape(persistent=True) as actor_tape1:
            a_for_grad = self.actor(states)
            critic_outputs = self.critic(states, a_for_grad)

        critic_grads = actor_tape1.gradient(critic_outputs, a_for_grad)
        critic_grads = [-i for i in critic_grads]
        # print(critic_outputs, a_for_grad)
        # print(critic_grads)
        critic_grads = tf.reshape(tf.concat(critic_grads, axis=0), [-1, 1])

        actor_trainable_variables = self.actor.trainable_variables
        params_grad = actor_tape1.gradient(
            a_for_grad, actor_trainable_variables, output_gradients=critic_grads
        )
        grads = zip(params_grad, actor_trainable_variables)
        self.actor_optimizer.apply_gradients(grads)

        # gradients = tf.gradients(critic_outputs, actor_trainable_variables, -self.action_gradient)
        # gradients = [-i for i in gradients]
        # self.actor_optimizer.apply_gradients(zip(gradients, actor_trainable_variables))

        # 3.更新actor目标网络和critic目标网络
        self.target_train(self.actor, self.actor_target)
        self.target_train(self.critic, self.critic_target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss
        # return 0


if __name__ == "__main__":
    get_env = GetEnv()

    # # 创建4个网络
    # actor = ActorNetwork()
    # actor_target = ActorNetwork()
    # critic = CriticNetwork()
    # critic_target = CriticNetwork()
    agent = Agent(get_env)

    # # Actor-critic网络预测
    # action = agent.actor(image_gray)
    # outputs = agent.critic(image_gray, action)
    # print(actor.summary())
    # print(critic.summary())

    # batch_size = 1
    # image_gray = tf.random.normal([batch_size, 200, 120, 1])
    # state = image_gray
    # action = tf.random.normal([batch_size, 1])
    # reward = 1
    # next_state = state
    # done = False

    checkpoint_path = "./checkpoints/ddpg_checkpoint"
    ckpt = tf.train.Checkpoint(
        actor=agent.actor,
        actor_target=agent.actor_target,
        critic=agent.critic,
        critic_target=agent.critic_target,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    scores = []
    episodes = 100
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
                print(
                    "Epoch: {}/{}, score: {}, use time: {}".format(
                        e + 1, episodes, time_t, time.time() - start_time
                    )
                )
                scores.append(time_t)
                break

        steps += time_t + 1
        loss /= time_t + 1

        print(
            "Epoch:",
            e,
            "step:",
            steps,
            "epsilon:",
            agent.epsilon,
            "loss:",
            loss.numpy(),
        )
        # ckpt_save_path = ckpt_manager.save()
        print("\n")
