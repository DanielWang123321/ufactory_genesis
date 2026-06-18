"""Robot model metadata for multi-arm support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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


ROBOT_PROFILES: Dict[str, RobotModelSpec] = {
  "xarm5_1305": RobotModelSpec(
    key="xarm5_1305",
    robot_name="xarm5",
    variant="1305",
    dof=5,
    ee_link="link5",
    assets_dir=_assets("xarm5"),
    mesh_variant="xarm5_1305",
    default_urdf="xarm5_1305.urdf",
    visual_glb_urdf="xarm5_1305_visual.glb.urdf",
    kinematics_prefix="xarm5",
    bio_gripper_g2_visual_urdf="xarm5_1305_bio_gripper_g2_visual.glb.urdf",
    bio_gripper_g2_movable_visual_urdf="xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
    gripper_g2_visual_urdf="xarm5_1305_g2_visual.urdf",
    gripper_g2_movable_visual_urdf="xarm5_1305_g2_movable_visual.urdf",
  ),
  "xarm7_1305": RobotModelSpec(
    key="xarm7_1305",
    robot_name="xarm7",
    variant="1305",
    dof=7,
    ee_link="link7",
    assets_dir=_assets("xarm7"),
    mesh_variant="xarm7_1305",
    default_urdf="xarm7_1305.urdf",
    visual_glb_urdf="xarm7_1305_visual.glb.urdf",
    kinematics_prefix="xarm7",
    bio_gripper_g2_visual_urdf="xarm7_1305_bio_gripper_g2_visual.glb.urdf",
    bio_gripper_g2_movable_visual_urdf="xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
    gripper_g2_visual_urdf="xarm7_1305_g2_visual.urdf",
    gripper_g2_movable_visual_urdf="xarm7_1305_g2_movable_visual.urdf",
  ),
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
  "xarm6_1305": RobotModelSpec(
    key="xarm6_1305",
    robot_name="xarm6",
    variant="1305",
    dof=6,
    ee_link="link6",
    assets_dir=_assets("xarm6"),
    mesh_variant="xarm6_1305",
    default_urdf="xarm6_1305.urdf",
    visual_glb_urdf="xarm6_1305_visual.glb.urdf",
    kinematics_prefix="xarm6",
    bio_gripper_g2_visual_urdf="xarm6_1305_bio_gripper_g2_visual.glb.urdf",
    bio_gripper_g2_movable_visual_urdf="xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
    gripper_g2_visual_urdf="xarm6_1305_g2_visual.urdf",
    gripper_g2_movable_visual_urdf="xarm6_1305_g2_movable_visual.urdf",
    glb_src_base_name="link_base.glb",
    glb_out_base_name="link_base.glb",
  ),
}


def get_robot_profile(key: str) -> RobotModelSpec:
  profile = ROBOT_PROFILES.get(key)
  if profile is None:
    raise KeyError(f"Unknown robot profile: {key}. Known: {sorted(ROBOT_PROFILES)}")
  return profile


def link_glb_stl_pairs(profile: RobotModelSpec) -> Tuple[Tuple[str, str], ...]:
  """Map source GLB filenames to STL reference names for relocalization."""
  pairs: list[Tuple[str, str]] = []
  pairs.append((profile.glb_src_base_name, profile.stl_base_name))
  for i in range(1, profile.dof + 1):
    pairs.append((f"link{i}.glb", f"link{i}.stl"))
  return tuple(pairs)


def glb_output_name(profile: RobotModelSpec, src_glb: str) -> str:
  if src_glb == profile.glb_src_base_name:
    return profile.glb_out_base_name
  return src_glb


def arm_link_names(profile: RobotModelSpec) -> Tuple[str, ...]:
  return tuple(["link_base"] + [f"link{i}" for i in range(1, profile.dof + 1)])


def joint_names(profile: RobotModelSpec) -> Tuple[str, ...]:
  return tuple(f"joint{i}" for i in range(1, profile.dof + 1))


def get_profile_key_for_robot_name(robot_name: str) -> str:
  """Resolve a robot name or profile key to a canonical profile key.

  Accepts both profile keys (``xarm6_1305``) and short robot names (``xarm6``).
  """
  if robot_name in ROBOT_PROFILES:
    return robot_name
  for key, profile in ROBOT_PROFILES.items():
    if profile.robot_name == robot_name:
      return key
  raise KeyError(f"Unknown robot: {robot_name}")
