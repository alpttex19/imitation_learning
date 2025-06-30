import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

from imitation_learning_lerobot.arm.robot import VX300S

if __name__ == '__main__':
    mj_model = mujoco.MjModel.from_xml_path("../../imitation_learning_lerobot/assets/aloha/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    q0 = mj_data.qpos.copy()
    q0[:6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    q0[8:14] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    mujoco.mj_setState(mj_model, mj_data, q0, mujoco.mjtState.mjSTATE_QPOS)
    mujoco.mj_forward(mj_model, mj_data)

    left_gripper = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
    print(mj_data.site(left_gripper))
    right_gripper = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
    print(mj_data.site(right_gripper))

    # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    #     while viewer.is_running():
    #         mj_data.ctrl[7:13] = 0.2
    #         mujoco.mj_step(mj_model, mj_data)
    #
    #         with viewer.lock():
    #             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mj_data.time % 2)
    #
    #         viewer.sync()

    robot = VX300S()

    base = sm.SE3.Trans(-0.469, -0.019, 0.02)
    tool = sm.SE3.RPY(0.0, -np.pi / 2, np.pi) * sm.SE3.Trans(0.13, 0.0, -0.003)
    robot.set_base(base)
    robot.set_tool(tool)

    q = np.zeros(robot.dof)
    q[:6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T = robot.fkine(q)
    print(T)

    right_robot = VX300S()
    right_base = sm.SE3.Trans(0.469, -0.019, 0.02) * sm.SE3.Rz(np.pi)
    right_tool = sm.SE3.RPY(0.0, -np.pi / 2, np.pi) * sm.SE3.Trans(0.13, 0.0, -0.003)
    right_robot.set_base(right_base)
    right_robot.set_tool(right_tool)
    right_q = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    right_T = right_robot.fkine(right_q)
    print(right_T)

    q_test = np.random.rand(6) * 100
    T_test = right_robot.fkine(q_test)
    right_robot.set_joint(q_test - 0.0001)
    qe_test = right_robot.ikine(T_test)
    print(q_test)
    print(qe_test)