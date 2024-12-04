import time

import cv2
import numpy as np
import pyautogui
import win32com.client
import win32gui
from PIL import Image

# 缩放比例
pingmu_suofang = 1


class GetEnv:
    def __init__(self):
        self.jpg_file = "screenshot.png"
        self.jpg_file2 = "screenshot2.png"
        self.promgram_name = "跳一跳"
        self.start_btn_img = cv2.imread("./pic/start.png", cv2.IMREAD_GRAYSCALE)
        self.restart_btn_img = cv2.imread("./pic/restart.png", cv2.IMREAD_GRAYSCALE)

        # 跳一跳位置
        self.tiao_x = 0
        self.tiao_y = 0

        self.tiao_left = 0
        self.tiao_top = 0
        self.tiao_right = 0
        self.tiao_bot = 0

        # 开始游戏位置
        self.start_x = 0
        self.stat_y = 0

    def set_foreground(sef, hwnd):
        """
        将窗口设置为最前面
        :param hwnd: 窗口句柄 一个整数
        """
        if hwnd != win32gui.GetForegroundWindow():
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.SetForegroundWindow(hwnd)

    def _find_start_btn(self, screen_shot_im, find_shot_im):
        """
        找到开始游戏位置的图标
        """
        screen_shot_im = cv2.imread(screen_shot_im, cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(screen_shot_im, find_shot_im, cv2.TM_CCOEFF_NORMED)
        if result.max() > 0.8:
            y, x = np.unravel_index(result.argmax(), result.shape)
            y += find_shot_im.shape[0] // 2
            x += find_shot_im.shape[1] // 2
            return x, y
        else:
            return -1, -1

    def screen_pic(self, promgram_name, jpg_file, jpg_file2):
        """
        截屏窗口图
        """
        hWnd = win32gui.FindWindow(
            None, promgram_name
        )  # 窗口的类名可以用Visual Studio的SPY++工具获取
        # 设置窗口在最前面
        self.set_foreground(hWnd)
        left, top, right, bot = win32gui.GetWindowRect(hWnd)
        # print(right,left, bot,top)
        self.left = left
        self.top = top
        self.right = right
        self.bot = bot
        self.tiao_x = (right + left) // 2
        self.tiao_y = (bot + top) // 2

        # 截屏
        img = pyautogui.screenshot(region=None)  # x,y,w,h
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(jpg_file, img)

        # 识别“点击开始游戏”位置
        self.start_x, self.stat_y = self._find_start_btn(jpg_file, self.start_btn_img)
        pyautogui.click(x=self.start_x, y=self.stat_y, duration=0.25)

        time.sleep(1)

        # 重新截屏
        img = pyautogui.screenshot(region=None)  # x,y,w,h
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(jpg_file, img)

        # 剪裁
        rangle = (
            int(left) * pingmu_suofang,
            int(top) * pingmu_suofang,
            int(right) * pingmu_suofang,
            int(bot) * pingmu_suofang,
        )

        img = Image.open(jpg_file)
        jpg = img.convert("RGB")
        jpg = img.crop(rangle)
        jpg.save(jpg_file2)
        # jpg.show()

    def get_gray_image(self, jpg_file2, is_show=False):
        """
        转成灰度图，并进行归一化, numpy(200, 120)
        """
        im = cv2.imread(jpg_file2, cv2.IMREAD_GRAYSCALE)

        if is_show:
            cv2.imshow("name:img_gray", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        im = np.expand_dims(cv2.resize(im, (120, 200)) / 255.0, -1)
        im = np.float32(np.expand_dims(im, 0))
        return im

    def defeate_return():
        """
        失败返回
        """

    def reset(self, is_show=False):
        """
        初始化环境
        """

        # 截屏
        self.screen_pic(self.promgram_name, self.jpg_file, self.jpg_file2)

        # 转灰度图
        image_gray = self.get_gray_image(self.jpg_file2, is_show)
        return image_gray

    def dist(self, action):
        """
        计算按压时间:
        return: 最低按压0.3s，最高按压1.1s
        """
        d = action * 400 + 700
        if d < 300:
            d = 300
        elif d > 1100:
            d = 1100

        return d / 1000

    def touch_in_step(self, action):
        """
        按压时间: ms
        """

        time.sleep(0.5)

        # 计算按压时间
        dist = self.dist(action.numpy()[0, 0])
        print("dist:", dist)
        pyautogui.moveTo(x=self.tiao_x, y=self.tiao_y, duration=0.25)
        pyautogui.dragTo(x=self.tiao_x, y=self.tiao_y, duration=dist)

        time.sleep(3)

        # 截屏
        ## 重新截屏
        img = pyautogui.screenshot(region=None)  # x,y,w,h
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.jpg_file, img)

        ## 剪裁
        rangle = (
            int(self.left) * pingmu_suofang,
            int(self.top) * pingmu_suofang,
            int(self.right) * pingmu_suofang,
            int(self.bot) * pingmu_suofang,
        )
        img = Image.open(self.jpg_file)
        jpg = img.convert("RGB")
        jpg = img.crop(rangle)
        jpg.save(self.jpg_file2)

        # 查找是否失败
        restart_x, restart_y = self._find_start_btn(self.jpg_file, self.restart_btn_img)

        # The game has not ended yet
        if restart_x == -1:
            state = self.get_gray_image(self.jpg_file2)
            reward = 1
            done = False
            return state, reward, done
        else:
            # 失败返回
            pyautogui.click(x=restart_x, y=restart_y, duration=0.4)
            time.sleep(2)
            state = None
            reward = -1
            done = True
            return state, reward, done


if __name__ == "__main__":
    get_env = GetEnv()
    res = get_env.reset(is_show=False)
    start_time = time.time()
    for i in range(10):
        print("%d Use time %f" % (i, time.time() - start_time))
        next_state, reward, done = get_env.touch_in_step(10)

        if done:
            res = get_env.reset(is_show=False)
        else:
            break
