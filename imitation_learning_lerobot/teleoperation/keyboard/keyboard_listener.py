import time
import threading

from pynput import keyboard


class KeyboardListener:
    def __init__(self):
        super().__init__()

        self.active_keys = set()
        self._listener = None

    def on_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if key_char not in self.active_keys:
            self.active_keys.add(key_char)

    def on_release(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if key_char in self.active_keys:
            self.active_keys.remove(key_char)

        if key == keyboard.Key.esc:
            return False

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self._listener.start()
