import time

import numpy as np

from imitation_learning_lerobot.envs import DishwasherEnv
from imitation_learning_lerobot.teleoperation.joycon.aloha_joycon_handle import AlohaJoyconHandle

if __name__ == '__main__':
    env = DishwasherEnv(render_mode="human")
    # env.run()
    env.reset()

    handle = AlohaJoyconHandle()
    handle.start()
    while True:
        action = np.zeros(14)
        action[:6] = handle.action

        env.step(action)

        env.render()
        time.sleep(0.01)