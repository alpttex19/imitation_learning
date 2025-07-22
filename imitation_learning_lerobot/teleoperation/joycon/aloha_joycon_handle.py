import threading
import time
import numpy as np
import spatialmath as sm

from pyjoycon import JoyCon, get_R_id

from ..handle import Handle
from .orientation_estimation import ComplimentaryOrientationEstimation


class AlohaJoyconHandle(Handle):
    def __init__(self):
        super().__init__()

        self._timestep = 0.01

        self._joycon_R = JoyCon(*get_R_id())

        self._state = np.zeros(6)
        self._action = np.zeros(6)
        self._last_action = np.zeros(6)

        self._orientation_estimation = ComplimentaryOrientationEstimation()

        self._sync = False
        self._done = False
        self._save = False

        self._thread: threading.Thread = None
        self._running = True

    def calibrate(self):
        num_samples = 100
        right_samples = []
        for _ in range(num_samples):
            status_R = self._joycon_R.get_status()
            accel_R = status_R['accel']
            rot_R = status_R['gyro']
            joystick_R = status_R['analog-sticks']['right']

            right_samples.append(
                [accel_R['x'], accel_R['y'], accel_R['z'], rot_R['x'], rot_R['y'], rot_R['z'], joystick_R['horizontal'],
                 joystick_R['vertical']])
            time.sleep(0.01)

        self.right_calibration_offset = np.mean(right_samples, axis=0)

    def start(self):
        self._orientation_estimation.start()

        self.calibrate()

        self._thread = threading.Thread(target=self.update_loop, daemon=True)
        self._thread.start()

    def update_loop(self):
        while self._running:
            self._update()
            time.sleep(self._timestep)

    def _update(self):
        status = self._joycon_R.get_status()

        if not self._sync:
            if status['buttons']['right']['a']:
                self._state[:3] = 0
                self._state[3:6] = self._orientation_estimation.euler_angles
                self._last_action[3:6] = self._action[3:6]
                self._sync = True
                print("sync  ")
        else:
            if status['buttons']['right']['y']:
                self._sync = False
                print("not sync")
        if status['buttons']['right']['sr']:
            self._done = True

        if not self._sync:
            return

        rotation = status['gyro']
        button_lower = status['buttons']['right']['zr']
        button_higher = status['buttons']['right']['r']
        joystick = status['analog-sticks']['right']
        up = status['buttons']['right']['x']
        down = status['buttons']['right']['b']

        R0 = sm.SO3.RPY(self._state[3:6])
        R_global = sm.SO3.RPY(self._orientation_estimation.euler_angles)
        delta_R: sm.SO3 = R0.inv() * R_global



        # self._action[0] -= (joystick['horizontal'] - self.right_calibration_offset[6]) * 0.00001
        # self._action[1] += (joystick['vertical'] - self.right_calibration_offset[7]) * 0.00001
        # self._action[2] += 0.005 if button_lower == 1 else -0.005 if button_higher == 1 else 0
        # self._action[3] = 1.0 if up == 1 else 0.0 if down == 1 else self._action[3]
        self._action[3:6] = (sm.SO3.RPY(self._last_action[3:6]) * delta_R).rpy()

    @property
    def action(self):
        return self._action.copy()
