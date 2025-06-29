import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from imitation_learning_lerobot.arm.robot import Robot, VX300S
from imitation_learning_lerobot.arm.motion_planning import LinePositionParameter, OneAttitudeParameter, \
    CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from imitation_learning_lerobot.utils import mj


class DishWasherEnv:
    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500
        self._control_hz = 25

        self._render_mode = render_mode

        scene_path = Path(__file__).parent.parent / Path("assets/aloha/scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot = VX300S()
        self._left_robot_q = np.zeros(self._left_robot.dof)
        self._left_robot_joint_names = ["left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
                                        "left/wrist_angle", "left/wrist_rotate"]
        self._left_robot_T = sm.SE3()
        self._left_T0 = sm.SE3()

        self._right_robot = VX300S()
        self._right_robot_q = np.zeros(self._right_robot.dof)
        self._right_robot_joint_names = ["left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
                                         "left/wrist_angle", "left/wrist_rotate"]
        self._right_robot_T = sm.SE3()
        self._right_T0 = sm.SE3()

        self._height = 480
        self._width = 640
        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)

    def reset(self):
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot.disable_base()
        self._left_robot.disable_tool()

        self._right_robot.disable_base()
        self._right_robot.disable_tool()

        self._left_robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "left/base_link").t)
        self._left_robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._left_robot.set_joint(self._left_robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._left_robot_q[i]) for i, jn in
         enumerate(self._left_robot_joint_names)]
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._left_robot.set_tool(np.array([0.0, 0.0, 0.2]))
        self._left_robot_T = self._left_robot.fkine(self._left_robot_q)
        self._left_T0 = self._left_robot_T.copy()

        self._right_robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "right/base_link").t)
        self._right_robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._right_robot.set_joint(self._right_robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._right_robot_q[i]) for i, jn in
         enumerate(self._right_robot_joint_names)]
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._right_robot.set_tool(np.array([0.0, 0.0, 0.2]))
        self._right_robot_T = self._right_robot.fkine(self._right_robot_q)
        self._right_T0 = self._right_robot_T.copy()

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}
        return observation, info

    def step(self, action):
        n_steps = self._sim_hz // self._control_hz

        if action is not None:
            self._latest_action = action
            for i in range(n_steps):
                left_Ti = sm.SE3.Trans(action[0], action[1], action[2]) * sm.SE3(sm.SO3(self._left_T0.R))
                self._left_robot.move_cartesian(left_Ti)
                left_joint_position = self._left_robot.get_joint()
                self._mj_data.ctrl[:6] = left_joint_position
                action[6] = np.clip(action[6], 0, 1)
                self._mj_data.ctrl[6] = action[3] * 255.0

                right_Ti = sm.SE3.Trans(action[7], action[8], action[9]) * sm.SE3(sm.SO3(self._right_T0.R))
                self._right_robot.move_cartesian(right_Ti)
                right_joint_position = self._right_robot.get_joint()
                self._mj_data.ctrl[7:13] = right_joint_position
                action[13] = np.clip(action[13], 0, 1)
                self._mj_data.ctrl[13] = action[13] * 255.0

                mujoco.mj_step(self._mj_model, self._mj_data)

        observation = self._get_observation()
        reward = 0.0
        terminated = False

        self._step_num += 1

        truncated = False
        if self._step_num > 10000:
            truncated = True

        info = {"is_success": terminated}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "human":
            self._mj_viewer.sync()

    def close(self):
        if self._mj_viewer is not None:
            self._mj_viewer.close()
        if self._mj_renderer is not None:
            self._mj_renderer.close()

    def seed(self, seed=None):
        pass

    def _get_observation(self):
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # for i in range(len(self._ur5e_joint_names)):
        #     self._robot_q[i] = mj.get_joint_q(self._mj_model, self._mj_data, self._ur5e_joint_names[i])[0]
        # self._robot_T = self._robot.fkine(self._robot_q)
        agent_pos = np.zeros(14, dtype=np.float32)
        # agent_pos[:3] = self._robot_T.t
        # agent_pos[3] = np.linalg.norm(self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos)

        overhead_cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        worms_eye_cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "worms_eye_cam")
        teleoperator_pov_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
        collaborator_pov_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "collaborator_pov")
        wrist_cam_left_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam_left")
        wrist_cam_right_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam_right")

        self._mj_renderer.update_scene(self._mj_data, overhead_cam_id)
        image_overhead_cam = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, worms_eye_cam_id)
        image_worms_eye_cam = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, teleoperator_pov_id)
        image_teleoperator_pov = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, collaborator_pov_id)
        image_collaborator_pov = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, wrist_cam_left_id)
        image_wrist_cam_left = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, wrist_cam_right_id)
        image_wrist_cam_right = self._mj_renderer.render()

        obs = {
            'pixels': {
                'overhead_cam': image_overhead_cam,
                'worms_eye_cam': image_worms_eye_cam,
                'teleoperator_pov': image_teleoperator_pov,
                'collaborator_pov': image_collaborator_pov,
                'wrist_cam_left': image_wrist_cam_left,
                'wrist_cam_right': image_wrist_cam_right
            },
            'agent_pos': agent_pos
        }
        # self._render_cache = image_top
        return obs

    def run(self):
        observation, info = self.reset()

        observations = []
        actions = []

        while True:
            action = np.zeros(14)
            action[:3] = self._left_T0.t
            action[7:10] = self._right_T0.t

            observations.append(observation)
            actions.append(action.copy)

            observation, _, _, _, info = self.step(action)

            self.render()


if __name__ == '__main__':
    env = DishWasherEnv(render_mode="human")
    env.run()
