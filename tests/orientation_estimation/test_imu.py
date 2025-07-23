import time

from imitation_learning_lerobot.teleoperation.joycon.right_joycon import RightJoycon

if __name__ == '__main__':
    imu = RightJoycon()
    while True:
        print("............")
        print(imu.get_acc())
        print(imu.get_gyro())
        time.sleep(0.1)
