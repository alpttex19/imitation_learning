import time

from imitation_learning_lerobot.teleoperation.joycon.orientation_estimation.complimentary_orientation_estimation import \
    ComplimentaryOrientationEstimation

if __name__ == '__main__':
    orientation_estimation = ComplimentaryOrientationEstimation()
    orientation_estimation.start()
    while True:
        print(orientation_estimation.euler_angles)
        time.sleep(1)
