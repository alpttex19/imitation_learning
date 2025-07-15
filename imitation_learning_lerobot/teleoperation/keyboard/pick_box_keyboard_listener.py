import numpy as np
from .keyboard_listener import KeyboardListener


class PickBoxKeyListener(KeyboardListener):
    def __init__(self):
        super().__init__()
        self._action0 = np.array([0.0, 0.0, 0.0, 0.0])
        self._action = np.array([0.0, 0.0, 0.0, 0.0])

        self._sync = False
        self._done = False
        self._save = True
        self._vel = 0.005

    def on_press(self, key):
        super().on_press(key)

        for key_char in self.active_keys:

            print(key_char)
            try:
                if key_char == 'Key.shift_r':
                    self._sync = not self._sync

                if key_char == "Key.backspace":
                    self._save = False

                if key_char == 'Key.space':
                    self._done = True

                if not self._sync:
                    break
                if key_char == '2':
                    self._action[2] += self._vel

                if key_char == '8':
                    self._action[2] -= self._vel

                if key_char == "6":
                    self._action[0] -= self._vel

                if key_char == '4':
                    self._action[0] += self._vel

                if key_char == '7':
                    self._action[1] += self._vel

                if key_char == '1':
                    self._action[1] -= self._vel

                if key_char == '9':
                    self._action[3] += 0.05

                if key_char == '3':
                    self._action[3] -= 0.05
            except AttributeError:
                pass

    def reset(self):
        self._action[:] = self._action0
        self._sync = False
        self._done = False
        self._save = True

    @property
    def action(self):
        return self._action.copy()

    @property
    def sync(self):
        return self._sync

    @property
    def done(self):
        return self._done

    @property
    def save(self):
        return self._save
