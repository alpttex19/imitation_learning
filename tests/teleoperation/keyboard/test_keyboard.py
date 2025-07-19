import keyboard
import time

if __name__ == '__main__':
    while True:
        if keyboard.is_pressed('q'):
            print("You pressed 'q")
            break
        if keyboard.is_pressed('a'):
            print('a was pressed')
        time.sleep(0.1)