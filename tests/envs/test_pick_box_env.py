import numpy as np
import cv2

from imitation_learning_lerobot.envs import PickBoxEnv
from imitation_learning_lerobot.utils.real_time_sync import RealTimeSync
from imitation_learning_lerobot.teleoperation.keyboard import PickBoxKeyListener

if __name__ == '__main__':
    env = PickBoxEnv(render_mode="human")

    listener = PickBoxKeyListener()
    listener.start()

    rt_sync = RealTimeSync(1.0 / env.control_hz)

    window_name1 = 'top'
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)

    window_name2 = 'hand'
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

    for i in range(5):

        observations = []
        actions = []

        listener.reset()
        observation, info = env.reset()
        rt_sync.reset()

        while not listener.done:

            if not listener.sync:
                rt_sync.reset()
                continue

            action = listener.action

            observations.append(observation)
            actions.append(action)

            observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            cv2.imshow(window_name1, cv2.cvtColor(observation["pixels"]['top'], cv2.COLOR_RGB2BGR))
            cv2.imshow(window_name2, cv2.cvtColor(observation["pixels"]['hand'], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            rt_sync.sync()

        env.close()

        if listener.save:
            pass
