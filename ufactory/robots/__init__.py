"""Robot model registry, asset paths, and runtime profiles."""

from ufactory.robots.registry import (
    PROJECT_ROOT,
    ROBOT_PROFILES,
    RobotModelSpec,
    get_profile_key_for_robot_name,
    get_robot_profile,
    joint_names,
    robot_cli_choices,
)
from ufactory.robots.paths import (
    DYNAMICS_VALIDATION_POSES_YAML,
    bio_gripper_g2_movable_visual_urdf,
    gripper_g2_movable_visual_urdf,
    kinematics_user_dir,
    lite6_gripper_movable_visual_urdf,
    robot_assets,
    robot_urdf,
    robot_visual_glb_urdf,
    xarm5_urdf,
    xarm6_1305_urdf,
    xarm6_urdf,
    xarm7_urdf,
)
from ufactory.robots.runtime import (
    ABS_ERR_FRACTION,
    DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
    ArmControlParams,
    DynamicsValidationParams,
    GripperControlParams,
    RobotRuntimeProfile,
    TaskProfile,
    abs_err_limits_from_efforts,
    get_robot_runtime_profile,
    robot_runtime_cli_choices,
)

__all__ = [
    "PROJECT_ROOT",
    "ROBOT_PROFILES",
    "RobotModelSpec",
    "RobotRuntimeProfile",
    "get_robot_profile",
    "get_robot_runtime_profile",
    "robot_cli_choices",
    "robot_runtime_cli_choices",
    "robot_assets",
    "robot_urdf",
    "robot_visual_glb_urdf",
    "kinematics_user_dir",
]
