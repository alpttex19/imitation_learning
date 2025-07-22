import time

from imitation_learning_lerobot.teleoperation.joycon.aloha_joycon_handle import AlohaJoyconHandle

if __name__ == '__main__':
    handle = AlohaJoyconHandle()
    handle.start()

    while True:
        print(handle.action)
        time.sleep(1)
