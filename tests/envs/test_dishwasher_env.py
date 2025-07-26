import time

import numpy as np
import cv2

from imitation_learning_lerobot.envs import DishwasherEnv
from imitation_learning_lerobot.teleoperation.joycon.aloha_joycon_handler import AlohaJoyconHandler
from loop_rate_limiters import RateLimiter

if __name__ == '__main__':
    env = DishwasherEnv(render_mode="human")
    env.reset()

    for camera in env.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    handler = AlohaJoyconHandler()
    handler.start()

    rate_limiter = RateLimiter(frequency=env.control_hz)

    while not handler.done:
        action = np.zeros(14)

        action[:] = handler.action

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        rate_limiter.sleep()

    handler.close()
    env.close()
