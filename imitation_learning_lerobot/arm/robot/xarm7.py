import numpy as np
from spatialmath import SE3
import tempfile
import os
from pathlib import Path

from .robot import Robot
from .xarm_manager import CasadiRobotManager


class XArm7(Robot):
    """
    XArm7 机械臂运动学类，使用Pinocchio库实现
    """

    def __init__(self, urdf_path: str = None) -> None:
        super().__init__()

        # XArm7基本参数
        self.robot_manager = CasadiRobotManager()
        self._dof = 7
        self.q0 = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]  # 从XML的keyframe获取

        # 关节名称 (从XML中提取)
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # 关节限制 (从XML中提取，弧度制)
        self.joint_limits_lower = np.array(
            [-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319]
        )
        self.joint_limits_upper = np.array(
            [6.28319, 2.0944, 6.28319, 3.927, 6.28319, 3.14159, 6.28319]
        )

        # 初始化配置
        self.q_current = np.array(self.q0)

    def fkine(self, q: np.ndarray) -> SE3:
        """
        正运动学：计算给定关节角度下的末端执行器位姿

        Args:
            q: 关节角度数组 (7,)

        Returns:
            SE3: 末端执行器的位姿矩阵
        """
        if len(q) != self._dof:
            raise ValueError(f"关节角度数组长度应为{self._dof}")

        # 转换为spatialmath的SE3格式
        return SE3(self.robot_manager.fkine(q))

    def ikine(self, Twt: SE3, q_init: np.ndarray = None) -> np.ndarray:
        """
        逆运动学：计算达到目标位姿所需的关节角度
        使用Pinocchio的数值迭代方法

        Args:
            Twt: 目标末端执行器位姿 (SE3)
            q_init: 初始关节角度猜测，如果为None则使用当前配置

        Returns:
            np.ndarray: 关节角度数组，如果求解失败返回空数组
        """
        if q_init is None:
            q_init = np.zeros(self._dof)
        else:
            q_init = np.array(q_init)

        # 检查初始关节角度是否在限制范围内
        q_init = np.clip(q_init, self.joint_limits_lower, self.joint_limits_upper)

        # 目标位姿矩阵
        sol_q, norm_err = self.robot_manager.ik(pos=Twt.t, rot=Twt.R)

        return sol_q, norm_err

    def move_cartesian(self, target_pose: SE3) -> bool:
        """
        移动到目标笛卡尔位姿

        Args:
            target_pose: 目标位姿

        Returns:
            bool: 是否成功移动
        """
        q_target = self.ikine(target_pose, self.q_current)
        if len(q_target) == 0:
            return False

        self.q_current = q_target
        return True

    def get_joint(self) -> np.ndarray:
        """获取当前关节角度"""
        return self.q_current.copy()

    def set_joint(self, q: np.ndarray) -> None:
        """设置关节角度"""
        if len(q) != self._dof:
            raise ValueError(f"关节角度数组长度应为{self._dof}")
        self.q_current = np.array(q)

    def get_cartesian(self) -> SE3:
        """获取当前末端执行器位姿"""
        return self.fkine(self.q_current)

    def check_joint_limits(self, q: np.ndarray) -> bool:
        """检查关节角度是否在限制范围内"""
        return np.all(q >= self.joint_limits_lower) and np.all(
            q <= self.joint_limits_upper
        )

    def random_configuration(self) -> np.ndarray:
        """生成随机有效关节配置"""
        return np.random.uniform(self.joint_limits_lower, self.joint_limits_upper)

    def distance_to_joint_limits(self, q: np.ndarray) -> float:
        """计算关节角度到限制边界的最小距离"""
        dist_lower = q - self.joint_limits_lower
        dist_upper = self.joint_limits_upper - q
        return np.min(np.minimum(dist_lower, dist_upper))


def create_xarm7_urdf_from_mjcf(mjcf_path: str, urdf_path: str = None) -> str:
    """
    从MuJoCo XML文件转换为URDF文件
    这是一个辅助函数，实际使用中可能需要手动转换或使用专门的转换工具

    Args:
        mjcf_path: MuJoCo XML文件路径
        urdf_path: 输出URDF文件路径，如果为None则创建临时文件

    Returns:
        str: URDF文件路径
    """
    if urdf_path is None:
        # 创建临时URDF文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
        urdf_path = temp_file.name
        temp_file.close()

    # 这里需要实现MJCF到URDF的转换
    # 可以使用mujoco.mjcf_to_urdf或其他转换工具
    # 简化起见，假设已有URDF文件
    print(f"请确保XArm7的URDF文件存在于: {urdf_path}")
    print("你可以使用以下命令从MuJoCo XML转换:")
    print(f"python -m mujoco.mjcf_to_urdf {mjcf_path} {urdf_path}")

    return urdf_path


# 使用示例
if __name__ == "__main__":
    # 需要提供XArm7的URDF文件路径
    urdf_path = "xarm7.urdf"  # 请替换为实际路径

    # 创建XArm7实例
    xarm = XArm7(urdf_path)

    # 测试正运动学
    q_test = np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_test = np.random.uniform(
        xarm.joint_limits_lower, xarm.joint_limits_upper, size=xarm._dof
    )
    T_test = xarm.fkine(q_test)

    # 测试逆运动学
    xarm.set_joint(q_test)
    q_ik, res_err = xarm.ikine(T_test)
    print(f"\n逆运动学结果:")
    print(f"原始关节角度: {q_test}")
    print(f"逆解关节角度: {q_ik}")
    print(f"误差: {res_err}")

    T_ik = xarm.fkine(q_ik)
    print("逆运动学结果:")
    print(T_ik)
    print("正运动学结果:")
    print(T_test)
