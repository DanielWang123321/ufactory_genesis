"""Typed runtime parameters for supported UFACTORY robot profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET

from ufactory.robots.registry import RobotModelSpec, get_robot_profile, joint_names, robot_cli_choices


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
    # Layered static analysis (L2/L3); L1 uses abs_err_limits / l2_err_limit.
    genesis_internal_abs_limits: FloatTuple = ()
    genesis_internal_l2_limit: float = 0.8
    pd_vs_pin_abs_limits: FloatTuple = ()
    pd_vs_pin_l2_limit: float = 5.0
    mass_rel_fro_limit_warn: float = 0.05
    mass_rel_fro_limit_fail: float = 0.15
    pin_vs_real_abs_scale: float = 1.5


@dataclass(frozen=True)
class GripperControlParams:
    """Optional gripper-control parameters for task and showcase examples.

    ``open_pos``/``close_pos`` are the Genesis drive-DOF values (native units:
    radians for Gripper G2's angled ``drive_joint``, metres for Lite6's
    prismatic ``finger_joint1``) at the fully-open/fully-closed physical
    two-finger gap. ``closed_gap_m``/``open_gap_m`` describe that physical gap
    range. The two families use opposite sign conventions (G2: open=0.0 <
    close=0.85; Lite6: close=0.0 < open=0.0089), so callers must always
    interpolate through ``open_pos``/``close_pos`` rather than assuming
    "larger drive = more closed".
    """

    family: str
    drive_joint: str
    all_joint_names: tuple[str, ...]
    finger_link_names: tuple[str, str]
    open_pos: float
    close_pos: float
    closed_gap_m: float
    open_gap_m: float
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
    pick_place_supported: bool = False
    showcase_supported: bool = False


@dataclass(frozen=True)
class RobotRuntimeProfile:
    """Resolved profile plus runtime parameters used by examples and validators."""

    model: RobotModelSpec
    arm: ArmControlParams
    dynamics: DynamicsValidationParams
    gripper: GripperControlParams | None = None
    task: TaskProfile = field(default_factory=TaskProfile)

    @property
    def gripper_g2(self) -> GripperControlParams | None:
        """Deprecated alias for :attr:`gripper` (kept for existing callers)."""
        return self.gripper


XARM6_KP: FloatTuple = (3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0)
XARM6_KV: FloatTuple = (300.0, 300.0, 200.0, 200.0, 100.0, 100.0)
XARM6_EFFORT: FloatTuple = (50.0, 50.0, 32.0, 32.0, 32.0, 20.0)
ABS_ERR_FRACTION = 0.10

_HARDWARE_DYNAMICS_KEYS = frozenset({"xarm5_1305", "xarm6_1305", "uf850", "lite6", "xarm7_1305"})

G2_GRIPPER_PARAMS = GripperControlParams(
    family="g2",
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
    closed_gap_m=0.0,
    open_gap_m=0.084,
    kp=20.0,
    kv=5.0,
    force_lower=-5.0,
    force_upper=5.0,
    damping=0.1,
    frictionloss=0.0,
)

LITE6_GRIPPER_PARAMS = GripperControlParams(
    family="lite6",
    drive_joint="finger_joint1",
    all_joint_names=("finger_joint1", "finger_joint2"),
    finger_link_names=("uflite_finger1", "uflite_finger2"),
    open_pos=0.0089,
    close_pos=0.0,
    closed_gap_m=0.020,
    open_gap_m=0.038,
    kp=500.0,
    kv=50.0,
    force_lower=-20.0,
    force_upper=20.0,
    damping=0.05,
    frictionloss=0.0,
)


def _tuple(values) -> FloatTuple:
    return tuple(map(float, values))


def abs_err_limits_from_efforts(
    efforts: FloatTuple,
    *,
    fraction: float = ABS_ERR_FRACTION,
) -> FloatTuple:
    """Per-joint absolute torque error limits (Nm) as a fraction of URDF effort."""
    return tuple(float(e) * fraction for e in efforts)


def _l2_err_limit_from_abs(abs_limits: FloatTuple) -> float:
    return max(2.0, sum(v * v for v in abs_limits) ** 0.5)


XARM6_ABS_ERR_LIMITS = abs_err_limits_from_efforts(XARM6_EFFORT)


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
        ("0", _tuple(home)),
        ("1", _tuple(default_qpos)),
        ("2", _tuple(small_pos)),
        ("3", _tuple(small_neg)),
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
    from ufactory.dynamics.poses_config import default_configs_named_tuple, stress_configs_for_robot

    internal_abs = tuple(0.5 for _ in arm.effort_limits)
    yaml_default = default_configs_named_tuple(profile.key)
    yaml_stress = stress_configs_for_robot(profile.key)
    default_configs = yaml_default or _generic_default_configs(profile, arm.default_qpos)
    stress_configs = yaml_stress
    abs_limits = abs_err_limits_from_efforts(arm.effort_limits)
    l2_limit = _l2_err_limit_from_abs(abs_limits)
    return DynamicsValidationParams(
        default_configs=default_configs,
        stress_configs=stress_configs,
        abs_err_limits=abs_limits,
        l2_err_limit=l2_limit,
        rel_err_limit=0.15,
        default_z_min_mm=50.0,
        supports_hardware_validation=profile.key in _HARDWARE_DYNAMICS_KEYS,
        genesis_internal_abs_limits=internal_abs,
        genesis_internal_l2_limit=0.8,
        pd_vs_pin_abs_limits=abs_limits,
        pd_vs_pin_l2_limit=l2_limit,
    )


def _task_profile(profile: RobotModelSpec) -> TaskProfile:
    return TaskProfile(
        reach_supported=True,
        pick_place_supported=profile.supports_gripper_g2 or profile.supports_lite6_gripper,
        showcase_supported=profile.supports_gripper_g2 or profile.supports_lite6_gripper,
    )


def _gripper_params_for(profile: RobotModelSpec) -> GripperControlParams | None:
    if profile.supports_gripper_g2:
        return G2_GRIPPER_PARAMS
    if profile.supports_lite6_gripper:
        return LITE6_GRIPPER_PARAMS
    return None


def _build_runtime_profile(profile: RobotModelSpec) -> RobotRuntimeProfile:
    arm = _arm_params(profile)
    return RobotRuntimeProfile(
        model=profile,
        arm=arm,
        dynamics=_dynamics_params(profile, arm),
        gripper=_gripper_params_for(profile),
        task=_task_profile(profile),
    )


def get_robot_runtime_profile(robot_key: str) -> RobotRuntimeProfile:
    """Resolve a robot key or alias to typed runtime parameters."""
    return _build_runtime_profile(get_robot_profile(robot_key))


def robot_runtime_cli_choices() -> list[str]:
    return robot_cli_choices()
