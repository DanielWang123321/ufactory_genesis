"""Project path helpers for UFACTORY URDF and configuration assets."""

from pathlib import Path

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
    """Return absolute path to a robot URDF by profile key or robot name."""
    profile = get_robot_profile(robot_key)
    urdf_name = name or profile.default_urdf
    return _urdf_path(profile.assets_dir, urdf_name)


def robot_visual_glb_urdf(
    robot_key: str,
    *,
    with_bio_gripper_g2: bool = False,
    with_gripper_g2: bool = False,
    with_lite6_gripper: bool = False,
    with_lite6_vacuum_gripper: bool = False,
    movable: bool = False,
) -> str:
    profile = get_robot_profile(robot_key)
    accessory_flags = (
        with_bio_gripper_g2,
        with_gripper_g2,
        with_lite6_gripper,
        with_lite6_vacuum_gripper,
    )
    if sum(accessory_flags) > 1:
        raise ValueError(
            "with_bio_gripper_g2, with_gripper_g2, with_lite6_gripper, "
            "and with_lite6_vacuum_gripper are mutually exclusive"
        )
    if movable and not (with_gripper_g2 or with_lite6_gripper or with_bio_gripper_g2):
        raise ValueError(
            "movable=True requires with_gripper_g2=True, with_lite6_gripper=True, or with_bio_gripper_g2=True"
        )
    if with_gripper_g2:
        if not profile.supports_gripper_g2:
            raise ValueError(f"Robot {robot_key} does not support Gripper G2")
        if movable:
            if not profile.gripper_g2_movable_visual_urdf:
                raise ValueError(f"Robot {robot_key} has no Gripper G2 movable visual URDF")
            return _urdf_path(profile.assets_dir, profile.gripper_g2_movable_visual_urdf)
        if not profile.gripper_g2_visual_urdf:
            raise ValueError(f"Robot {robot_key} has no Gripper G2 visual URDF")
        return _urdf_path(profile.assets_dir, profile.gripper_g2_visual_urdf)
    if with_bio_gripper_g2:
        if not profile.supports_bio_gripper_g2 or not profile.bio_gripper_g2_visual_urdf:
            raise ValueError(f"Robot {robot_key} does not support Bio Gripper G2 visual URDF")
        if movable:
            if not profile.bio_gripper_g2_movable_visual_urdf:
                raise ValueError(f"Robot {robot_key} has no Bio Gripper G2 movable visual URDF")
            return _urdf_path(profile.assets_dir, profile.bio_gripper_g2_movable_visual_urdf)
        return _urdf_path(profile.assets_dir, profile.bio_gripper_g2_visual_urdf)
    if with_lite6_gripper:
        if not profile.supports_lite6_gripper:
            raise ValueError(f"Robot {robot_key} does not support Lite6 Gripper")
        if movable:
            if not profile.lite6_gripper_movable_visual_urdf:
                raise ValueError(f"Robot {robot_key} has no Lite6 Gripper movable visual URDF")
            return _urdf_path(profile.assets_dir, profile.lite6_gripper_movable_visual_urdf)
        if not profile.lite6_gripper_visual_urdf:
            raise ValueError(f"Robot {robot_key} has no Lite6 Gripper visual URDF")
        return _urdf_path(profile.assets_dir, profile.lite6_gripper_visual_urdf)
    if with_lite6_vacuum_gripper:
        if not profile.supports_lite6_vacuum_gripper or not profile.lite6_vacuum_gripper_visual_urdf:
            raise ValueError(f"Robot {robot_key} does not support Lite6 Vacuum Gripper visual URDF")
        return _urdf_path(profile.assets_dir, profile.lite6_vacuum_gripper_visual_urdf)
    return _urdf_path(profile.assets_dir, profile.visual_glb_urdf)


def lite6_gripper_movable_visual_urdf() -> str:
    """Standalone Lite6 gripper-only movable visual URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf")


def gripper_g2_movable_visual_urdf() -> str:
    """Standalone gripper-only movable visual URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf")


def bio_gripper_g2_movable_visual_urdf() -> str:
    """Standalone Bio Gripper G2 movable visual URDF (no arm)."""
    return _existing_path(PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf")
