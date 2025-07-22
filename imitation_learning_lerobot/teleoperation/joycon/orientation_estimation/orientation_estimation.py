import abc
import threading
import time

import numpy as np

from .imu import Imu


class OrientationEstimation(abc.ABC):

    def __init__(self):
        super().__init__()

        self._phi = 0.0
        self._theta = 0.0
        self._psi = 0.0
        self._timestep = 0.01

        self._imu = Imu()

        self._thread: threading.Thread = None
        self._running = True

    def start(self):
        self._imu.start()

        self._thread = threading.Thread(target=self.update_loop, daemon=True)
        self._thread.start()

    def update_loop(self):
        while self._running:
            self.update()
            time.sleep(self._timestep)

    @abc.abstractmethod
    def update(self):
        pass

    @property
    def euler_angles(self):
        return self._phi, self._theta, self._psi
