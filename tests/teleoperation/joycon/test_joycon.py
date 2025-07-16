from pyjoycon import JoyCon, get_L_id, get_R_id
import time

# 自动检测连接的Joy-Con
# joycon_id = get_L_id() or get_R_id()
# if not joycon_id:
#     raise Exception("未检测到Joy-Con")

# joycon = JoyCon(*joycon_id)
joycon = JoyCon(*get_R_id())

try:
    while True:
        # 获取当前状态
        status = joycon.get_status()

        # 打印按钮状态
        print("\n--- Joy-Con 状态 ---")
        print(
            f"摇杆: 左X={status['analog-sticks']['left']['horizontal']}, 左Y={status['analog-sticks']['left']['vertical']}")
        print(f"按钮: A={status['buttons']['right']['a']}, B={status['buttons']['right']['b']}")
        print(f"加速度计: X={status['accel']['x']}, Y={status['accel']['y']}, Z={status['accel']['z']}")
        print(f"陀螺仪: X={status['gyro']['x']}, Y={status['gyro']['y']}, Z={status['gyro']['z']}")

        time.sleep(0.1)  # 控制读取频率

except KeyboardInterrupt:
    print("\n停止读取Joy-Con状态")
