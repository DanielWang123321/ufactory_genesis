"""Project path helpers for xArm URDF assets."""

from pathlib import Path

from ufactory.robot_registry import PROJECT_ROOT, get_profile_key_for_robot_name, get_robot_profile
XARM6_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "xarm6"
XARM5_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "xarm5"
XARM7_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "xarm7"
LITE6_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "lite6"
UF850_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "uf850"
BIO_GRIPPER_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "bio_gripper"
GRIPPER_G2_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "gripper_g2"
LITE6_GRIPPER_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper"
LITE6_VACUUM_GRIPPER_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper"

XARM6_KINEMATICS_USER_DIR = XARM6_ASSETS / "kinematics" / "user"
XARM5_KINEMATICS_USER_DIR = XARM5_ASSETS / "kinematics" / "user"
XARM7_KINEMATICS_USER_DIR = XARM7_ASSETS / "kinematics" / "user"
LITE6_KINEMATICS_USER_DIR = LITE6_ASSETS / "kinematics" / "user"
UF850_KINEMATICS_USER_DIR = UF850_ASSETS / "kinematics" / "user"

XARM6_1305_VISUAL_GLB_URDF = "xarm6_1305_visual.glb.urdf"
XARM6_1305_G2_VISUAL_URDF = "xarm6_1305_g2_visual.urdf"
XARM6_1305_G2_MOVABLE_VISUAL_URDF = "xarm6_1305_g2_movable_visual.urdf"

_KINEMATICS_DIRS = {
  "xarm5": XARM5_KINEMATICS_USER_DIR,
  "xarm6": XARM6_KINEMATICS_USER_DIR,
  "xarm7": XARM7_KINEMATICS_USER_DIR,
  "lite6": LITE6_KINEMATICS_USER_DIR,
  "uf850": UF850_KINEMATICS_USER_DIR,
}


def _urdf_path(assets: Path, name: str) -> str:
  path = assets / name
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def xarm6_urdf(name: str = "xarm6.urdf") -> str:
  """Return absolute path to an xArm 6 URDF file in project assets."""
  return _urdf_path(XARM6_ASSETS, name)


def xarm6_1305_urdf() -> str:
  """Return absolute path to the xArm6 XI1305 simulation URDF."""
  return xarm6_urdf("xarm6_1305.urdf")


def xarm6_1305_visual_glb_urdf(with_gripper_g2: bool = False, movable: bool = False) -> str:
  """Return URDF path for xArm6 1305 GLB visual model (optionally with Gripper G2)."""
  return robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=with_gripper_g2, movable=movable)


def robot_assets(robot_name: str) -> Path:
  profile = get_robot_profile(get_profile_key_for_robot_name(robot_name))
  return profile.assets_dir


def kinematics_user_dir(robot_name: str) -> Path:
  key = get_profile_key_for_robot_name(robot_name)
  profile = get_robot_profile(key)
  return profile.assets_dir / "kinematics" / "user"


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
      "movable=True requires with_gripper_g2=True, with_lite6_gripper=True, "
      "or with_bio_gripper_g2=True"
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


def xarm5_1305_urdf() -> str:
  return robot_urdf("xarm5_1305")


def xarm5_1305_visual_glb_urdf(with_bio_gripper_g2: bool = False) -> str:
  return robot_visual_glb_urdf("xarm5_1305", with_bio_gripper_g2=with_bio_gripper_g2)


def xarm7_1305_urdf() -> str:
  return robot_urdf("xarm7_1305")


def xarm7_1305_visual_glb_urdf(with_bio_gripper_g2: bool = False) -> str:
  return robot_visual_glb_urdf("xarm7_1305", with_bio_gripper_g2=with_bio_gripper_g2)


def lite6_urdf() -> str:
  return robot_urdf("lite6")


def lite6_visual_glb_urdf(
  *,
  with_lite6_gripper: bool = False,
  with_lite6_vacuum_gripper: bool = False,
  movable: bool = False,
) -> str:
  return robot_visual_glb_urdf(
    "lite6",
    with_lite6_gripper=with_lite6_gripper,
    with_lite6_vacuum_gripper=with_lite6_vacuum_gripper,
    movable=movable,
  )


def lite6_with_gripper_urdf() -> str:
  profile = get_robot_profile("lite6")
  if not profile.lite6_with_gripper_urdf:
    raise ValueError("lite6 has no with_gripper physics URDF")
  return _urdf_path(profile.assets_dir, profile.lite6_with_gripper_urdf)


def lite6_with_vacuum_gripper_urdf() -> str:
  profile = get_robot_profile("lite6")
  if not profile.lite6_with_vacuum_gripper_urdf:
    raise ValueError("lite6 has no with_vacuum_gripper physics URDF")
  return _urdf_path(profile.assets_dir, profile.lite6_with_vacuum_gripper_urdf)


def lite6_gripper_movable_visual_urdf() -> str:
  """Standalone Lite6 gripper-only movable visual URDF (no arm)."""
  path = LITE6_GRIPPER_ASSETS / "lite6_gripper_movable_visual.urdf"
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def uf850_urdf() -> str:
  return robot_urdf("uf850")


def uf850_visual_glb_urdf(with_bio_gripper_g2: bool = False) -> str:
  return robot_visual_glb_urdf("uf850", with_bio_gripper_g2=with_bio_gripper_g2)


def bio_gripper_glb(ee_link: str = "link6") -> str:
  path = BIO_GRIPPER_ASSETS / "meshes" / "visual" / f"bio_gripper_g2_visual_{ee_link}.glb"
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def gripper_g2_static_glb(ee_link: str = "link6") -> str:
  path = GRIPPER_G2_ASSETS / "meshes" / "visual" / f"gripper_g2_static_{ee_link}.glb"
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def gripper_g2_base_glb(ee_link: str = "link6") -> str:
  path = GRIPPER_G2_ASSETS / "meshes" / "visual" / "visual_glb" / ee_link / "base.glb"
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def gripper_g2_shared_glb(name: str) -> str:
  path = GRIPPER_G2_ASSETS / "meshes" / "visual" / "visual_glb" / name
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def gripper_g2_movable_visual_urdf() -> str:
  """Standalone gripper-only movable visual URDF (no arm)."""
  path = GRIPPER_G2_ASSETS / "gripper_g2_movable_visual.urdf"
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


def bio_gripper_movable_visual_urdf() -> str:
  """Standalone Bio Gripper G2 movable visual URDF (no arm)."""
  path = BIO_GRIPPER_ASSETS / "bio_gripper_movable_visual.urdf"
  if not path.exists():
    raise FileNotFoundError(path)
  return str(path.resolve())


