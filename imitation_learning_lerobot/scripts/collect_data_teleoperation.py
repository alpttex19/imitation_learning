import time
from pathlib import Path
import argparse

from loop_rate_limiters import RateLimiter
import numpy as np
import cv2

from imitation_learning_lerobot.envs import EnvFactory
from imitation_learning_lerobot.teleoperation import HandlerFactory


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='env type'
    )

    parser.add_argument(
        '--handler.type',
        type=str,
        dest='handler_type',
        required=True,
        help='handler type'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    env_type = args.env_type
    env_cls = EnvFactory.get_strategies(env_type)
    env = env_cls(render_mode="human")
    env.reset()

    for camera in env_cls.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    handler_cls = HandlerFactory.get_strategies(env.name + "_" + args.handler_type)
    handler = handler_cls()
    handler.start()

    rate_limiter = RateLimiter(frequency=env.control_hz)

    while not handler.done:
        action = handler.action
        # print(action)

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        rate_limiter.sleep()

    handler.close()
    env.close()


if __name__ == '__main__':
    main()
