import time

import numpy as np

from pyjoycon import GyroTrackingJoyCon, get_R_id

from .orientation_estimation import OrientationEstimation


class ComplimentaryOrientationEstimation(OrientationEstimation):

    def __init__(self):
        super().__init__()
        self._imu = GyroTrackingJoyCon(*get_R_id())
        self._imu.calibrate()
        time.sleep(3)
        print()

    def update(self):
        accel = self._imu.accel_in_g
        gx, gy, gz = self._imu.gyro_in_rad # it may need to be multiplied by two

        # self._phi = np.arctan2(ay, az)
        # self._theta = np.arctan2(-ax, np.linalg.norm([ay, az]))

        ax, ay, az = accel[0]
        self._phi = np.arctan2(-ay, -az)
        self._theta = np.arctan2(-ax, np.linalg.norm([ay, az]))
        self._psi = 0.0

        # phi_dot = p + np.sin(phi_hat) * tan(theta_hat) * q + np.cos(phi_hat) * tan(theta_hat) * r
        # theta_dot = np.cos(phi_hat) * q - np.sin(phi_hat) * r