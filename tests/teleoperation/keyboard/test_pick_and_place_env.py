import numpy as np

from imitation_learning_lerobot.envs import PickAndPlaceEnv
from imitation_learning_lerobot.utils.real_time_sync import RealTimeSync
from imitation_learning_lerobot.teleoperation.keyboard.keyboard_listener import KeyboardListener


class MotionKeyListener(KeyboardListener):
    def __init__(self):
        super().__init__()
        self._action0 = np.array([1.292, 0.734, 1.083, 0.0])
        self._action = np.array([1.292, 0.734, 1.083, 0.0])

        self._sync = False
        self._done = False
        self._vel = 0.005

    def on_press(self, key):
        super().on_press(key)

        for key_char in self.active_keys:

            print(key_char)
            if key_char == 'Key.shift_r':
                self._sync = not self._sync

            if key_char == 'Key.space':
                self._done = True

            if not self._sync:
                break

            if key_char.lower() == '2':
                self._action[0] += self._vel

            if key_char.lower() == '8':
                self._action[0] -= self._vel

            if key_char.lower() == "6":
                self._action[1] += self._vel

            if key_char.lower() == '4':
                self._action[1] -= self._vel

            if key_char.lower() == '7':
                self._action[2] += self._vel

            if key_char.lower() == '1':
                self._action[2] -= self._vel

            if key_char.lower() == '9':
                self._action[3] += 0.05

            if key_char.lower() == '3':
                self._action[3] -= 0.05

    def reset(self):
        self._action[:] = self._action0
        self._sync = False
        self._done = False

    @property
    def action(self):
        return self._action.copy()

    @property
    def sync(self):
        return self._sync

    @property
    def done(self):
        return self._done


if __name__ == '__main__':
    env = PickAndPlaceEnv(render_mode="human")
    observation, info = env.reset()

    listener = MotionKeyListener()
    listener.start()

    listener.reset()

    rt_sync = RealTimeSync(1.0 / env.control_hz)
    while not listener.done:

        if not listener.sync:
            continue

        action = listener.action
        env.step(action)
        env.render()
        rt_sync.sync()

    env.close()
