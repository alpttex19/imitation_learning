from imitation_learning_lerobot.envs import DishWasherEnv

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = DishWasherEnv(render_mode="human")
    data = env.run()
    actions = np.array(data["actions"])
    plt.figure(1)
    for i in range(7):
        plt.plot(actions[:, i], label=str(i))
    plt.legend()

    plt.figure(2)
    for i in range(7):
        plt.plot(actions[:, 7 + i], label=str(i))
    plt.legend()
    plt.show()
