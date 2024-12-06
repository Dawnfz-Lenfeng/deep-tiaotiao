import numpy as np

from window import GameWindow


class Env:
    """游戏环境"""

    def __init__(self):
        self.window = GameWindow()
        # 状态空间：游戏截图
        self.state_dim = 784  # 28*28
        # 动作空间：按压时间
        self.action_dim = 1
        self.action_bound = 1.0
        # 跳跃次数追踪
        self.jump_count = 0

    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重新定位游戏区域并重置游戏
        self.window.get_game_region()
        # 重置跳跃次数
        self.jump_count = 0
        # 获取初始状态
        state = self.window.get_state()
        return state

    def step(self, action: float) -> tuple[np.ndarray, float, bool]:
        """执行一步动作"""
        # 计算按压时间
        duration = self._to_duration(action)
        # 执行点击
        self.window.click_screen(duration)

        # 获取新状态和奖励
        next_state = self.window.get_state()
        done = self.window.check_done()

        self.jump_count += 1
        factor = 100 / self.jump_count  # 计算跳跃次数的因子
        if done:
            # 死亡时损失所有累积的分数
            reward = -10.0 * (1 + factor)  # 基础惩罚加上累积分数损失
            self.jump_count = 0
        else:
            # 奖励随跳跃次数增加而增加
            reward = 2 * (1 + factor)  # 每次跳跃的奖励会逐渐增加

        return next_state, reward, done

    def _to_duration(self, action: float) -> int:
        """将动作值转换为按压时间(ms)"""
        # 将[-1,1]映射到[300,1100]ms
        normalized = np.clip(action, -1, 1)
        return int(300 * normalized + 700)
