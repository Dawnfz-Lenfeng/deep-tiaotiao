import time

import cv2
import numpy as np
import pyautogui
import win32gui


class GameWindow:
    """游戏窗口管理"""

    def __init__(self):
        # 加载模板图片
        self.start_btn = cv2.imread("pic/start.png", cv2.IMREAD_GRAYSCALE)
        self.restart_btn = cv2.imread("pic/restart.png", cv2.IMREAD_GRAYSCALE)
        self.chequer = cv2.imread("pic/chequer.png", cv2.IMREAD_GRAYSCALE)
        self.game_region = None
        self.window_name = "跳一跳"  # 微信小程序窗口名称

    def get_game_region(self):
        """获取游戏区域并重置游戏"""
        # 查找游戏窗口
        hwnd = win32gui.FindWindow(None, self.window_name)
        if not hwnd:
            return

        # 设置窗口在最前面
        win32gui.SetForegroundWindow(hwnd)
        # 获取窗口位置
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        self.game_region = (left, top, right - left, bottom - top)

        # 先尝试点击重启按钮
        screen = self.get_screenshot()
        self._click_button(screen, self.restart_btn)

        # 再点击开始按钮
        screen = self.get_screenshot()
        self._click_button(screen, self.start_btn)

    def check_done(self) -> bool:
        """检查是否结束"""
        # 检测重启按钮是否出现
        screen = self.get_screenshot()
        result = cv2.matchTemplate(screen, self.restart_btn, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > 0.5

    def get_screenshot(self) -> np.ndarray:
        """获取游戏区域的截图"""
        screen = np.array(pyautogui.screenshot(region=self.game_region))
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)  # 转灰度
        return gray

    def click_screen(self, duration: float):
        """点击屏幕"""
        # 获得点击位置
        x = self.game_region[0] + self.game_region[2] // 2
        y = self.game_region[1] + self.game_region[3] // 2
        # 执行点击
        pyautogui.mouseDown(x, y)
        pyautogui.sleep(duration / 1000.0)
        pyautogui.mouseUp()
        # 等待动画结束
        time.sleep(4)

    def get_state(self) -> np.ndarray:
        """获取当前状态特征"""
        screen = self.get_screenshot()
        # 获取模板图像的高度和宽度
        result = cv2.matchTemplate(self.chequer, screen, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        _, y = max_loc
        h, _ = self.chequer.shape[:2]
        # 计算左下角坐标,从棋子位置向上裁剪3倍棋子高度
        screen = screen[y - 3 * h : y + h, :]
        # 缩放到28x28
        resized = cv2.resize(screen, (28, 28))
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(-1)

    def _click_button(self, screen: np.ndarray, template: np.ndarray):
        """点击按钮, 重启或开始"""
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.5:
            return

        x, y = max_loc
        click_x = self.game_region[0] + x + template.shape[1] // 2
        click_y = self.game_region[1] + y + template.shape[0] // 2
        pyautogui.click(click_x, click_y)
        time.sleep(1)  # 等待动画
