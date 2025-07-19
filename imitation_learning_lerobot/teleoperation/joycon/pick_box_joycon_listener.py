import time
import numpy as np
from loop_rate_limiters import RateLimiter
from pyjoycon import JoyCon, get_R_id


class PickBoxJoyconListener:
    def __init__(self):
        super().__init__()

        self._joycon_R = JoyCon(*get_R_id())

        self._action = np.zeros(4)

        self._sync = False
        self._done = False
        self._save = False

    def control(self):
        status = self._joycon_R.get_status()

        if status['buttons']['right']['a']:
            self._sync = True
        if status['buttons']['right']['y']:
            self._sync = False
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

        self._action[0] -= (joystick['horizontal'] - self.right_calibration_offset[6]) * 0.00001
        self._action[1] += (joystick['vertical'] - self.right_calibration_offset[7]) * 0.00001
        self._action[2] += 0.005 if button_lower == 1 else -0.005 if button_higher == 1 else 0
        self._action[3] = 1.0 if up == 1 else 0.0 if down == 1 else self._action[3]

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

    def reset(self):
        self.calibrate()

        self._action[:] = 0
        self._sync = False
        self._done = False
        self._save = False

    def start(self):
        pass

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
