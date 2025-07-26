import time

from imitation_learning_lerobot.teleoperation.joycon.aloha_joycon_handler import AlohaJoyconHandler

if __name__ == '__main__':
    handle = AlohaJoyconHandler()
    handle.start()

    while True:
        print(handle.action)
        time.sleep(1)
