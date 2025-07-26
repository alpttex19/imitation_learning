import threading

import numpy as np
from loop_rate_limiters import RateLimiter

from .right_joycon import RightJoycon
from ..handler import Handler

class PickBoxJoyconHandler(Handler):
    def __init__(self):
        super().__init__()

        self._timestep = 0.01

        self._state = np.zeros(4)
        self._action = np.zeros(4)
        self._last_action = np.zeros(4)
        self._filter_action = np.zeros(4)

        self._right_joycon = RightJoycon

        self._right_sync = False
        self._done = False

        self._thread: threading.Thread = None
        self._running = True

    def start(self):
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        pass