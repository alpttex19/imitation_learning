import abc
import threading
import time

import numpy as np


class OrientationEstimation(abc.ABC):

    def __init__(self):
        super().__init__()

        self._phi = 0.0
        self._theta = 0.0
        self._psi = 0.0
        # self._euler_angles = np.zeros(3)
        self._timestep = 0.01

        self._thread: threading.Thread = None
        self._running = True

    def start(self):
        self._thread = threading.Thread(target=self.solve_loop, daemon=True)
        self._thread.start()

    def solve_loop(self):
        while self._running:
            self.update()
            time.sleep(0.01)

    @abc.abstractmethod
    def update(self):
        pass

    @property
    def euler_angles(self):
        return self._phi, self._theta, self._psi
