"""Project path helpers for UFACTORY URDF and configuration assets."""

from pathlib import Path

from ufactory.robots.assembly import compose_robot_urdf, get_end_effector, parse_assembly_ref
from ufactory.robots.registry import PROJECT_ROOT, get_robot_profile

DYNAMICS_VALIDATION_POSES_YAML = PROJECT_ROOT / "assets" / "configs" / "dynamics_validation_pose.yaml"


def _existing_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path.resolve())


def _urdf_path(assets: Path, name: str) -> str:
    return _existing_path(assets / name)


def xarm5_urdf(name: str = "xarm5_1305.urdf") -> str:
    """Return absolute path to an xArm 5 URDF file in project assets."""
    return _urdf_path(get_robot_profile("xarm5_1305").assets_dir, name)


def xarm6_urdf(name: str = "xarm6_1305.urdf") -> str:
    """Return absolute path to an xArm 6 URDF file in project assets."""
    return _urdf_path(get_robot_profile("xarm6_1305").assets_dir, name)


def xarm7_urdf(name: str = "xarm7_1305.urdf") -> str:
    """Return absolute path to an xArm 7 URDF file in project assets."""
    return _urdf_path(get_robot_profile("xarm7_1305").assets_dir, name)


def xarm6_1305_urdf() -> str:
    """Return absolute path to the xArm6 XI1305 simulation URDF."""
    return xarm6_urdf()


def robot_assets(robot_name: str) -> Path:
    return get_robot_profile(robot_name).assets_dir


def kinematics_user_dir(robot_name: str) -> Path:
    return get_robot_profile(robot_name).assets_dir / "kinematics" / "user"


def robot_urdf(robot_key: str, name: str | None = None) -> str:
    """Return the absolute path of a robot URDF by profile key or robot name.

    ``name`` is a file in the arm asset directory (default: the dynamics URDF) or an assembly reference such as
    ``assembly:gripper_g2`` (see :mod:`ufactory.robots.assembly`), which is composed on demand.
    """
    profile = get_robot_profile(robot_key)
    urdf_name = name or profile.default_urdf
    ref = parse_assembly_ref(urdf_name)
    if ref is not None:
        return compose_robot_urdf(profile.key, ref.end_effector, visual=ref.visual, gripper_visual=ref.gripper_visual)
    return _urdf_path(profile.assets_dir, urdf_name)


_ACCESSORY_FLAGS = (
    ("with_bio_gripper_g2", "bio_gripper_g2", "Bio Gripper G2"),
    ("with_gripper_g2", "gripper_g2", "Gripper G2"),
    ("with_vacuum_gripper", "vacuum_gripper", "Vacuum Gripper"),
    ("with_lite6_gripper", "lite6_gripper", "Lite6 Gripper"),
    ("with_lite6_vacuum_gripper", "lite6_vacuum_gripper", "Lite6 Vacuum Gripper"),
)


def robot_visual_glb_urdf(
    robot_key: str,
    *,
    with_bio_gripper_g2: bool = False,
    with_gripper_g2: bool = False,
    with_vacuum_gripper: bool = False,
    with_lite6_gripper: bool = False,
    with_lite6_vacuum_gripper: bool = False,
    movable: bool = False,
) -> str:
    """GLB-visual URDF of an arm, optionally composed with one end effector.

    Without an accessory this is the arm's ``*_visual.glb.urdf``. With one, the arm and part are assembled at
    runtime: ``movable=True`` keeps per-link gripper visuals, otherwise grippers use their static overlay.
    New code should call :func:`ufactory.robots.assembly.compose_robot_urdf` directly.
    """
    profile = get_robot_profile(robot_key)
    flags = {
        "with_bio_gripper_g2": with_bio_gripper_g2,
        "with_gripper_g2": with_gripper_g2,
        "with_vacuum_gripper": with_vacuum_gripper,
        "with_lite6_gripper": with_lite6_gripper,
        "with_lite6_vacuum_gripper": with_lite6_vacuum_gripper,
    }
    selected = [(name, label) for flag, name, label in _ACCESSORY_FLAGS if flags[flag]]
    if len(selected) > 1:
        raise ValueError(
            "with_bio_gripper_g2, with_gripper_g2, with_vacuum_gripper, "
            "with_lite6_gripper, and with_lite6_vacuum_gripper are mutually exclusive"
        )
    if movable and not (with_gripper_g2 or with_lite6_gripper or with_bio_gripper_g2):
        raise ValueError(
            "movable=True requires with_gripper_g2=True, with_lite6_gripper=True, or with_bio_gripper_g2=True"
        )
    if not selected:
        return _urdf_path(profile.assets_dir, profile.visual_glb_urdf)
    end_effector, label = selected[0]
    if not profile.supports(end_effector):
        raise ValueError(f"Robot {robot_key} does not support {label}")
    has_static = get_end_effector(end_effector).static_visual is not None
    gripper_visual = "movable" if movable or not has_static else "static"
    return compose_robot_urdf(profile.key, end_effector, visual="glb", gripper_visual=gripper_visual)


def lite6_gripper_movable_visual_urdf() -> str:
    """Standalone Lite6 gripper-only movable visual URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf")


def gripper_g2_movable_visual_urdf() -> str:
    """Standalone gripper-only movable visual URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf")


def bio_gripper_g2_urdf() -> str:
    """Standalone Bio Gripper G2 URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2.urdf")


def bio_gripper_g2_movable_visual_urdf() -> str:
    """Standalone Bio Gripper G2 movable visual URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf")


def vacuum_gripper_urdf() -> str:
    """Standalone xArm / UF850 vacuum gripper URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "vacuum_gripper" / "vacuum_gripper.urdf")


def microplate_sbs_96_well_urdf() -> str:
    """SBS 96-well microplate URDF for Bio Gripper G2 pick-and-place."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "microplate" / "sbs_96_well.urdf")
