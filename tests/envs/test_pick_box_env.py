import numpy as np
import cv2

from imitation_learning_lerobot.envs import PickBoxEnv
from imitation_learning_lerobot.teleoperation import Handler, HandlerFactory
from loop_rate_limiters import RateLimiter

if __name__ == '__main__':
    env = PickBoxEnv(render_mode="human")
    env.reset()

    for camera in env.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    handler_type = "joycon"
    handler_cls = HandlerFactory.get_strategies(env.name + "_" + handler_type)
    handler = handler_cls()
    handler.start()
    print(handler.right_calibration_offset)

    rate_limiter = RateLimiter(frequency=env.control_hz)

    while not handler.done:
        action = handler.action

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        rate_limiter.sleep()

    handler.close()
    env.close()
