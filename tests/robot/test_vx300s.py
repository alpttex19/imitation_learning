import numpy as np
import spatialmath as sm
import placo
from placo_utils.tf import tf

from imitation_learning_lerobot.arm.robot import VX300S, RobotWrapper

if __name__ == '__main__':
    robot = VX300S()
    robot.set_tool(sm.SE3.RPY(0.0, -np.pi / 2, np.pi) * sm.SE3.Trans([0.13, 0.0, -0.003]))
    q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) + 0.8
    print(robot.fkine(q0))

    Te = robot.fkine(q0)
    print(Te)
    qe1 = robot.ikine(Te)
    print(qe1)

    robot_wrapper = RobotWrapper()
    robot_wrapper.set_tool(sm.SE3.Trans([0.13, 0.0, -0.003]))
    print(robot_wrapper.fkine(q0))
    qe2 = robot_wrapper.ikine(Te)
    print(qe2)
    print(robot_wrapper.fkine(qe2))
