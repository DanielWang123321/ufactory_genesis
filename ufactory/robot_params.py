"""Typed runtime parameters for supported UFACTORY robot profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET
from typing import Any

from ufactory.robot_registry import RobotModelSpec, get_robot_profile, joint_names, robot_cli_choices


FloatTuple = tuple[float, ...]
NamedPoseTuple = tuple[tuple[str, FloatTuple], ...]

# ≈ 40°/s; all real-robot motion APIs use rad / rad/s (see real_robot_session).
DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S = math.radians(40.0)


@dataclass(frozen=True)
class ArmControlParams:
    """Runtime arm-control parameters derived from a robot profile."""

    joint_names: tuple[str, ...]
    ee_link: str
    home_qpos: FloatTuple
    default_qpos: FloatTuple
    kp: FloatTuple
    kv: FloatTuple
    force_lower: FloatTuple
    force_upper: FloatTuple
    effort_limits: FloatTuple


@dataclass(frozen=True)
class DynamicsValidationParams:
    """Per-robot dynamics validation parameters."""

    default_configs: NamedPoseTuple
    stress_configs: NamedPoseTuple = ()
    abs_err_limits: FloatTuple = ()
    l2_err_limit: float = 5.0
    rel_err_limit: float = 0.15
    default_z_min_mm: float = 50.0
    default_move_speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
    supports_hardware_validation: bool = False


@dataclass(frozen=True)
class GripperControlParams:
    """Optional gripper-control parameters for task and showcase examples."""

    drive_joint: str
    all_joint_names: tuple[str, ...]
    finger_link_names: tuple[str, str]
    open_pos: float
    close_pos: float
    kp: float
    kv: float
    force_lower: float
    force_upper: float
    damping: float
    frictionloss: float


@dataclass(frozen=True)
class TaskProfile:
    """Task-level defaults and capability flags."""

    reach_supported: bool = True
    grasp_place_supported: bool = False
    showcase_supported: bool = False
    reach_env_defaults: dict[str, Any] = field(default_factory=dict)
    grasp_place_env_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotRuntimeProfile:
    """Resolved profile plus runtime parameters used by examples and validators."""

    model: RobotModelSpec
    arm: ArmControlParams
    dynamics: DynamicsValidationParams
    gripper_g2: GripperControlParams | None = None
    task: TaskProfile = field(default_factory=TaskProfile)


# Dynamics validation poses sourced from absolute-accuracy calibration file
# (/home/uf/Desktop/xarm6_joint_pos.txt). 20 calib poses selected with EE y
# hemisphere balance (10 y+ / 10 y-) via scripts/select_dynamics_calib_poses.py;
# all pass Genesis settled/non-saturated checks and simulation collision pre-check.
XARM6_DEFAULT_DYNAMICS_CONFIGS: NamedPoseTuple = (
    ("home", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("calib_002", (0.593014, -0.423881, -1.697147, -0.454217, 0.561190, -0.000004)),
    ("calib_004", (1.123183, -0.790596, -1.011288, -1.495171, 0.762858, 0.136503)),
    ("calib_017", (0.494226, 0.005116, -1.621027, -1.090653, 0.167756, 0.647378)),
    ("calib_010", (1.591889, 0.220991, -1.565362, -1.795572, 1.447152, -0.035028)),
    ("calib_005", (0.984202, 0.617494, -1.341380, -2.370746, 1.384759, 0.227052)),
    ("calib_007", (0.226203, -0.886881, -0.947475, -2.788992, -0.096351, 0.227021)),
    ("calib_012", (1.318133, 0.818430, -1.325307, -1.973838, 1.637310, -0.079406)),
    ("calib_026", (2.378493, 0.409956, -1.527388, -1.418330, 2.194494, 1.928450)),
    ("calib_028", (1.862799, 0.118636, -2.242597, -1.813369, 1.453178, 2.287752)),
    ("calib_029", (1.693289, -1.244139, -0.996242, -1.618179, 1.645012, 2.287745)),
    ("calib_030", (-0.519561, -0.868379, -0.745066, 1.514022, 0.877686, 2.287719)),
    ("calib_037", (-0.925738, 0.115881, -1.689866, 1.626955, 1.369697, 2.243433)),
    ("calib_044", (-1.869600, -0.640124, -1.180910, 1.693657, 2.002397, 1.933161)),
    ("calib_040", (-0.560133, 0.151814, -0.922369, 2.044854, 1.473716, 1.740232)),
    ("calib_039", (-1.455897, -0.700937, -0.340808, 1.644372, 1.855563, 2.243433)),
    ("calib_043", (-2.127668, 0.162554, -1.663111, 1.551238, 2.568216, 2.172132)),
    ("calib_036", (-0.563515, 0.587003, -1.609372, 2.096422, 1.370123, 2.243449)),
    ("calib_031", (-0.957875, -0.868233, -0.973407, 1.391589, 1.270079, 2.061584)),
    ("calib_049", (-0.147554, 0.312715, -1.474090, 2.343635, 1.107538, 1.987774)),
    ("calib_034", (-0.328742, 0.477321, -1.647003, 2.212445, 1.139788, 2.061434)),
)

XARM6_STRESS_DYNAMICS_CONFIGS: NamedPoseTuple = (
    ("config_H", (-1.5, 0.8, -0.8, -1.5, 1.2, -2.0)),
)

XARM6_KP: FloatTuple = (3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0)
XARM6_KV: FloatTuple = (300.0, 300.0, 200.0, 200.0, 100.0, 100.0)
XARM6_EFFORT: FloatTuple = (50.0, 50.0, 32.0, 32.0, 32.0, 20.0)
XARM6_ABS_ERR_LIMITS: FloatTuple = (3.0, 3.0, 2.0, 2.0, 2.0, 2.0)

G2_GRIPPER_PARAMS = GripperControlParams(
    drive_joint="drive_joint",
    all_joint_names=(
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    ),
    finger_link_names=("left_finger", "right_finger"),
    open_pos=0.0,
    close_pos=0.85,
    kp=20.0,
    kv=5.0,
    force_lower=-5.0,
    force_upper=5.0,
    damping=0.1,
    frictionloss=0.0,
)


def _tuple(values) -> FloatTuple:
    return tuple(map(float, values))


def _fit(values: tuple[float, ...], n: int, *, fill: float = 0.0) -> FloatTuple:
    if len(values) >= n:
        return _tuple(values[:n])
    return _tuple((*values, *([fill] * (n - len(values)))))


def _parse_effort_limits(profile: RobotModelSpec) -> FloatTuple:
    urdf = profile.assets_dir / profile.default_urdf
    fallback = _fit(XARM6_EFFORT, profile.dof, fill=XARM6_EFFORT[-1])
    try:
        root = ET.parse(str(urdf)).getroot()
    except Exception:
        return fallback
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    efforts: list[float] = []
    for name in joint_names(profile):
        joint = by_name.get(name)
        limit = joint.find("limit") if joint is not None else None
        try:
            efforts.append(abs(float(limit.get("effort"))) if limit is not None else fallback[len(efforts)])
        except Exception:
            efforts.append(fallback[len(efforts)])
    return _tuple(efforts)


def _default_qpos(profile: RobotModelSpec) -> FloatTuple:
    if profile.key == "xarm6_1305":
        return (0.0, -0.5, 0.0, 0.0, 0.5, 0.0)
    if profile.key == "lite6":
        return (0.0, -0.6, 0.0, 0.0, 0.6, 0.0)
    return _fit((0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0), profile.dof)


def _generic_default_configs(profile: RobotModelSpec, default_qpos: FloatTuple) -> NamedPoseTuple:
    home = tuple(0.0 for _ in range(profile.dof))
    small_pos = list(home)
    small_neg = list(home)
    for idx in range(profile.dof):
        small_pos[idx] = 0.15 if idx % 2 == 0 else -0.15
        small_neg[idx] = -small_pos[idx]
    return (
        ("home", _tuple(home)),
        ("default", _tuple(default_qpos)),
        ("small_pos", _tuple(small_pos)),
        ("small_neg", _tuple(small_neg)),
    )


def _arm_params(profile: RobotModelSpec) -> ArmControlParams:
    efforts = _parse_effort_limits(profile)
    if profile.key == "xarm6_1305":
        kp = XARM6_KP
        kv = XARM6_KV
        force_upper = XARM6_EFFORT
    else:
        kp = _fit((3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0, 800.0), profile.dof, fill=800.0)
        kv = _fit((300.0, 300.0, 200.0, 200.0, 100.0, 100.0, 80.0), profile.dof, fill=80.0)
        force_upper = efforts
    return ArmControlParams(
        joint_names=joint_names(profile),
        ee_link=profile.ee_link,
        home_qpos=tuple(0.0 for _ in range(profile.dof)),
        default_qpos=_default_qpos(profile),
        kp=_fit(kp, profile.dof),
        kv=_fit(kv, profile.dof),
        force_lower=tuple(-abs(v) for v in force_upper[: profile.dof]),
        force_upper=_fit(force_upper, profile.dof),
        effort_limits=efforts,
    )


def _dynamics_params(profile: RobotModelSpec, arm: ArmControlParams) -> DynamicsValidationParams:
    if profile.key == "xarm6_1305":
        return DynamicsValidationParams(
            default_configs=XARM6_DEFAULT_DYNAMICS_CONFIGS,
            stress_configs=XARM6_STRESS_DYNAMICS_CONFIGS,
            abs_err_limits=XARM6_ABS_ERR_LIMITS,
            l2_err_limit=5.0,
            rel_err_limit=0.15,
            default_z_min_mm=50.0,
            supports_hardware_validation=True,
        )
    abs_limits = tuple(max(1.0, min(3.0, effort * 0.08)) for effort in arm.effort_limits)
    return DynamicsValidationParams(
        default_configs=_generic_default_configs(profile, arm.default_qpos),
        abs_err_limits=abs_limits,
        l2_err_limit=max(2.0, sum(v * v for v in abs_limits) ** 0.5),
        rel_err_limit=0.15,
        default_z_min_mm=50.0,
        supports_hardware_validation=False,
    )


def _task_profile(profile: RobotModelSpec) -> TaskProfile:
    reach_env_defaults = {
        "num_obs": profile.dof * 2 + 6,
        "num_actions": profile.dof,
        "action_scale": 0.05,
        "episode_length_s": 5.0,
        "ctrl_dt": 0.02,
        "target_pos_lower": [0.15, -0.3, 0.05],
        "target_pos_upper": [0.55, 0.3, 0.50],
    }
    grasp_defaults = {
        "num_obs": 22,
        "num_actions": 4,
        "action_scales": [0.05, 0.05, 0.05, 1.0],
        "episode_length_s": 10.0,
        "ctrl_dt": 0.02,
        "table_height": 0.4,
        "obj_size": [0.04, 0.04, 0.04],
        "obj_spawn_lower": [0.28, -0.05, 0.0],
        "obj_spawn_upper": [0.32, 0.05, 0.0],
        "target_spawn_lower": [0.40, -0.10, 0.0],
        "target_spawn_upper": [0.55, 0.10, 0.0],
        "substeps": 4,
    }
    return TaskProfile(
        reach_supported=True,
        grasp_place_supported=profile.key == "xarm6_1305",
        showcase_supported=profile.key == "xarm6_1305",
        reach_env_defaults=reach_env_defaults,
        grasp_place_env_defaults=grasp_defaults,
    )


def _build_runtime_profile(profile: RobotModelSpec) -> RobotRuntimeProfile:
    arm = _arm_params(profile)
    return RobotRuntimeProfile(
        model=profile,
        arm=arm,
        dynamics=_dynamics_params(profile, arm),
        gripper_g2=G2_GRIPPER_PARAMS if profile.supports_gripper_g2 else None,
        task=_task_profile(profile),
    )


def get_robot_runtime_profile(robot_key: str) -> RobotRuntimeProfile:
    """Resolve a robot key or alias to typed runtime parameters."""
    return _build_runtime_profile(get_robot_profile(robot_key))


def robot_runtime_cli_choices() -> list[str]:
    return robot_cli_choices()
