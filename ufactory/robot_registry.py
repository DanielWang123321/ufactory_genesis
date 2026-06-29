"""Robot model metadata for multi-arm support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RobotModelSpec:
  """Metadata for a supported UFACTORY arm variant."""

  key: str
  robot_name: str
  variant: str
  dof: int
  ee_link: str
  assets_dir: Path
  mesh_variant: str
  default_urdf: str
  visual_glb_urdf: str
  kinematics_prefix: str
  supports_bio_gripper_g2: bool = True
  bio_gripper_g2_visual_urdf: str = ""
  bio_gripper_g2_movable_visual_urdf: str = ""
  supports_gripper_g2: bool = True
  gripper_g2_visual_urdf: str = ""
  gripper_g2_movable_visual_urdf: str = ""
  supports_lite6_gripper: bool = False
  lite6_gripper_visual_urdf: str = ""
  lite6_gripper_movable_visual_urdf: str = ""
  supports_lite6_vacuum_gripper: bool = False
  lite6_vacuum_gripper_visual_urdf: str = ""
  lite6_with_gripper_urdf: str = ""
  lite6_with_vacuum_gripper_urdf: str = ""
  glb_src_base_name: str = "link0.glb"
  glb_out_base_name: str = "link_base.glb"
  stl_base_name: str = "link_base.stl"


def _assets(robot: str) -> Path:
  return PROJECT_ROOT / "assets" / "urdf" / robot


def _xarm_profile(dof: int, *, glb_src_base_name: str = "link0.glb") -> RobotModelSpec:
  robot = f"xarm{dof}"
  key = f"{robot}_1305"
  return RobotModelSpec(
    key=key,
    robot_name=robot,
    variant="1305",
    dof=dof,
    ee_link=f"link{dof}",
    assets_dir=_assets(robot),
    mesh_variant=key,
    default_urdf=f"{key}.urdf",
    visual_glb_urdf=f"{key}_visual.glb.urdf",
    kinematics_prefix=robot,
    bio_gripper_g2_visual_urdf=f"{key}_bio_gripper_g2_visual.glb.urdf",
    bio_gripper_g2_movable_visual_urdf=f"{key}_bio_gripper_g2_movable_visual.glb.urdf",
    gripper_g2_visual_urdf=f"{key}_g2_visual.urdf",
    gripper_g2_movable_visual_urdf=f"{key}_g2_movable_visual.urdf",
    glb_src_base_name=glb_src_base_name,
    glb_out_base_name="link_base.glb",
  )


ROBOT_PROFILES: dict[str, RobotModelSpec] = {
  "xarm5_1305": _xarm_profile(5),
  "xarm7_1305": _xarm_profile(7),
  "lite6": RobotModelSpec(
    key="lite6",
    robot_name="lite6",
    variant="",
    dof=6,
    ee_link="link6",
    assets_dir=_assets("lite6"),
    mesh_variant="lite6",
    default_urdf="lite6.urdf",
    visual_glb_urdf="lite6_visual.glb.urdf",
    kinematics_prefix="lite6",
    supports_bio_gripper_g2=False,
    bio_gripper_g2_visual_urdf="",
    supports_gripper_g2=False,
    gripper_g2_visual_urdf="",
    gripper_g2_movable_visual_urdf="",
    supports_lite6_gripper=True,
    lite6_gripper_visual_urdf="lite6_gripper_visual.glb.urdf",
    lite6_gripper_movable_visual_urdf="lite6_gripper_movable_visual.glb.urdf",
    supports_lite6_vacuum_gripper=True,
    lite6_vacuum_gripper_visual_urdf="lite6_vacuum_gripper_visual.glb.urdf",
    lite6_with_gripper_urdf="lite6_with_gripper.urdf",
    lite6_with_vacuum_gripper_urdf="lite6_with_vacuum_gripper.urdf",
  ),
  "uf850": RobotModelSpec(
    key="uf850",
    robot_name="uf850",
    variant="",
    dof=6,
    ee_link="link6",
    assets_dir=_assets("uf850"),
    mesh_variant="uf850",
    default_urdf="uf850.urdf",
    visual_glb_urdf="uf850_visual.glb.urdf",
    kinematics_prefix="uf850",
    bio_gripper_g2_visual_urdf="uf850_bio_gripper_g2_visual.glb.urdf",
    bio_gripper_g2_movable_visual_urdf="uf850_bio_gripper_g2_movable_visual.glb.urdf",
    gripper_g2_visual_urdf="uf850_g2_visual.urdf",
    gripper_g2_movable_visual_urdf="uf850_g2_movable_visual.urdf",
  ),
  "xarm6_1305": _xarm_profile(6, glb_src_base_name="link_base.glb"),
}


def get_robot_profile(key: str) -> RobotModelSpec:
  return ROBOT_PROFILES[get_profile_key_for_robot_name(key)]


def link_glb_stl_pairs(profile: RobotModelSpec) -> tuple[tuple[str, str], ...]:
  """Map source GLB filenames to STL reference names for relocalization."""
  return ((profile.glb_src_base_name, profile.stl_base_name), *((f"link{i}.glb", f"link{i}.stl") for i in range(1, profile.dof + 1)))


def glb_output_name(profile: RobotModelSpec, src_glb: str) -> str:
  return profile.glb_out_base_name if src_glb == profile.glb_src_base_name else src_glb


def arm_link_names(profile: RobotModelSpec) -> tuple[str, ...]:
  return ("link_base", *(f"link{i}" for i in range(1, profile.dof + 1)))


def joint_names(profile: RobotModelSpec) -> tuple[str, ...]:
  return tuple(f"joint{i}" for i in range(1, profile.dof + 1))


def get_profile_key_for_robot_name(robot_name: str) -> str:
  """Resolve a robot name or profile key to a canonical profile key.

  Accepts both profile keys (``xarm6_1305``) and short robot names (``xarm6``).
  Short xArm names resolve to the XI1305 profile (``xarm6`` -> ``xarm6_1305``).
  """
  if robot_name in ROBOT_PROFILES:
    return robot_name
  preferred = f"{robot_name}_1305"
  if preferred in ROBOT_PROFILES:
    return preferred
  matches = [key for key, profile in ROBOT_PROFILES.items() if profile.robot_name == robot_name]
  if matches:
    return sorted(matches)[0]
  raise KeyError(f"Unknown robot: {robot_name}")


def robot_cli_choices() -> list[str]:
  """Sorted ``--robot`` choices: profile keys plus short robot-name aliases."""
  keys = set(ROBOT_PROFILES)
  aliases = {profile.robot_name for profile in ROBOT_PROFILES.values()}
  return sorted(keys | aliases)
