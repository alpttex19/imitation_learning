import time

from imitation_learning_lerobot.teleoperation.joycon.orientation_estimation.imu import Imu

if __name__ == '__main__':
    imu = Imu()
    while True:
        print("............")
        print(imu.get_acc())
        print(imu.get_gyro())
        time.sleep(0.1)
