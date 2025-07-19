import time

class JoyConListener:
    def __init__(self):
        super().__init__()

        self._active_keys = set()