import time

import numpy as np
import pyautogui

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

    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重新定位游戏区域并重置游戏
        self.window.get_game_region()
        # 获取初始状态
        state = self.window.get_state()
        return state

    def step(self, action: float) -> tuple[np.ndarray, float, bool]:
        """执行一步动作
        Args:
            action: [0,1] 范围内的值，用于计算按压时间
        Returns:
            state: 游戏画面特征
            reward: 奖励值 (+1成功/-1失败)
            done: 是否结束
        """
        # 计算按压时间
        duration = self._to_duration(action)
        # 执行点击
        self.window.click_screen(duration)

        # 获取新状态和奖励
        next_state = self.window.get_state()
        done = self.window.check_done()
        reward = -1 if done else 1

        return next_state, reward, done

    def _to_duration(self, action: float) -> int:
        """将动作值转换为按压时间(ms)"""
        # 将[-1,1]映射到[300,1100]ms
        normalized = np.clip(action, -1, 1)
        return int(300 * normalized + 700)
