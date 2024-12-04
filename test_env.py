import time
import cv2
import numpy as np
from env import Env


def test_reset_and_step():
    """测试环境重置和动作执行"""
    env = Env()

    # 测试重置
    print("Testing reset...")
    state = env.reset()
    print(f"State shape: {state.shape}")

    # 测试不同大小的动作
    actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for action in actions:
        print(f"\nTesting action: {action}")
        print(f"Press duration: {env._to_duration(action)}ms")

        next_state, reward, done = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Next state shape: {next_state.shape}")

        if done:
            print("Game Over! Resetting...")


def test_window_detection():
    """测试窗口检测和按钮点击"""
    env = Env()

    # 测试游戏区域检测
    region = env.window.get_game_region()
    print(f"Game region: {region}")

    # 测试截图
    screen = env.window.get_screenshot()
    print(f"Screenshot shape: {screen.shape}")

    # 测试结束检测
    done = env.window.check_done()
    print(f"Game done: {done}")


def main():
    print("Starting environment tests...")

    # 测试窗口检测
    print("\n=== Testing Window Detection ===")
    test_window_detection()

    # 测试环境交互
    print("\n=== Testing Environment Interaction ===")
    test_reset_and_step()


if __name__ == "__main__":
    main()
