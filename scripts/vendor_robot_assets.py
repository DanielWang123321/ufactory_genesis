#!/usr/bin/env python3
"""Vendor URDF and STL/OBJ meshes from xarm_ros2 xarm_description into project assets."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import PROJECT_ROOT
from ufactory.robot_registry import ROBOT_PROFILES, RobotModelSpec

XARM_ROS2_REPO = "https://github.com/xArm-Developer/xarm_ros2.git"

VENDOR_SPECS = {
  "xarm5_1305": {
    "xacro_subdir": "xarm5",
    "xacro_file": "xarm5.urdf.xacro",
    "macro": "xarm5_urdf",
    "model_num": "1305",
    "mesh_ros": "xarm5_1305",
    "default_kinematics": "xarm5_default_kinematics.yaml",
  },
  "xarm7_1305": {
    "xacro_subdir": "xarm7",
    "xacro_file": "xarm7.urdf.xacro",
    "macro": "xarm7_urdf",
    "model_num": "1305",
    "mesh_ros": "xarm7_1305",
    "default_kinematics": "xarm7_default_kinematics.yaml",
  },
  "lite6": {
    "xacro_subdir": "lite6",
    "xacro_file": "lite6.urdf.xacro",
    "macro": "lite6_urdf",
    "model_num": None,
    "mesh_ros": "lite6",
    "default_kinematics": "lite6_default_kinematics.yaml",
  },
  "uf850": {
    "xacro_subdir": "uf850",
    "xacro_file": "uf850.urdf.xacro",
    "macro": "uf850_urdf",
    "model_num": None,
    "mesh_ros": "uf850",
    "default_kinematics": "uf850_default_kinematics.yaml",
  },
}


def _ensure_xarm_ros2(source: Path | None) -> Path:
  if source is not None and (source / "xarm_description").is_dir():
    return source / "xarm_description"
  cache = Path(tempfile.gettempdir()) / "xarm_ros2_vendor_cache" / "xarm_description"
  if cache.is_dir():
    return cache
  repo_root = cache.parent
  if not (repo_root / ".git").is_dir():
    subprocess.run(
      ["git", "clone", "--depth", "1", XARM_ROS2_REPO, str(repo_root)],
      check=True,
    )
  return cache


def _preprocess_xarm_description(src: Path, work: Path) -> Path:
  if work.exists():
    shutil.rmtree(work)
  shutil.copytree(src, work)

  for path in work.rglob("*.xacro"):
    text = path.read_text(encoding="utf-8")
    text = text.replace("$(find xarm_description)", str(work))
    path.write_text(text, encoding="utf-8")
  return work


def _default_kinematics_yaml(xarm_desc: Path, spec: dict) -> Path:
  name = spec["default_kinematics"]
  path = xarm_desc / "config" / "kinematics" / "default" / name
  if not path.is_file():
    raise FileNotFoundError(f"Missing default kinematics: {path}")
  return path


def _write_wrapper_xacro(
  work_desc: Path,
  spec: dict,
  kinematics_yaml: Path,
  out_xacro: Path,
) -> None:
  sub = spec["xacro_subdir"]
  macro = spec["macro"]
  model_arg = ""
  if spec["model_num"]:
    model_arg = f' model_num="{spec["model_num"]}"'
  content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="gen">
  <xacro:property name="is_ros2" value="true"/>
  <xacro:property name="use_xacro_load_yaml" value="true"/>
  <xacro:property name="use_len" value="true"/>
  <xacro:property name="mesh_suffix" value="stl"/>
  <xacro:property name="mesh_path" value="meshes"/>
  <xacro:include filename="{work_desc / 'urdf/common/common.material.xacro'}"/>
  <xacro:common_material prefix=""/>
  <xacro:include filename="{work_desc / 'urdf/common/common.link.xacro'}"/>
  <xacro:include filename="{work_desc / 'urdf' / sub / spec['xacro_file']}"/>
  <xacro:{macro} prefix=""{model_arg} kinematics_params_filename="{kinematics_yaml}"/>
</robot>
"""
  out_xacro.write_text(content, encoding="utf-8")


def _run_xacro(wrapper: Path) -> str:
  result = subprocess.run(
    ["xacro", str(wrapper)],
    check=True,
    capture_output=True,
    text=True,
    env={**dict(subprocess.os.environ), "XACRO_INORDER": "1"},
  )
  return result.stdout


def _add_world_link(urdf_text: str) -> str:
  if "<link name=\"world\"" in urdf_text or "<link name='world'" in urdf_text:
    return urdf_text
  insert = (
    '  <link name="world" />\n\n'
    '  <joint name="world_joint" type="fixed">\n'
    '    <parent link="world" />\n'
    '    <child link="link_base" />\n'
    '    <origin xyz="0 0 0" rpy="0 0 0" />\n'
    "  </joint>\n\n"
  )
  return urdf_text.replace("<link name=\"link_base\">", insert + '  <link name="link_base">', 1)


def _normalize_mesh_paths(urdf_text: str, mesh_variant: str) -> str:
  urdf_text = re.sub(
    r"meshes/[a-zA-Z0-9_]+/",
    f"meshes/{mesh_variant}/",
    urdf_text,
  )
  return urdf_text


def _copy_meshes(work_desc: Path, profile: RobotModelSpec, mesh_ros: str) -> None:
  src_visual = work_desc / "meshes" / mesh_ros / "visual"
  src_collision = work_desc / "meshes" / mesh_ros / "collision"
  dst_root = profile.assets_dir / "meshes" / profile.mesh_variant
  dst_visual = dst_root / "visual"
  dst_collision = dst_root / "collision"
  dst_visual.mkdir(parents=True, exist_ok=True)
  dst_collision.mkdir(parents=True, exist_ok=True)

  if src_visual.is_dir():
    for path in src_visual.glob("*.stl"):
      shutil.copy2(path, dst_visual / path.name.lower())
    for path in src_visual.glob("*.STL"):
      shutil.copy2(path, dst_visual / path.name.lower())
  if src_collision.is_dir():
    for path in src_collision.glob("*"):
      if path.suffix.lower() in {".obj", ".stl"}:
        shutil.copy2(path, dst_collision / path.name.lower())


def _make_visual_glb_urdf(base_urdf: Path, profile: RobotModelSpec) -> Path:
  tree = ET.parse(str(base_urdf))
  root = tree.getroot()
  glb_dir = f"meshes/{profile.mesh_variant}/visual_glb"
  for link in root.findall("link"):
    for tag in ("visual",):
      geom_parent = link.find(tag)
      if geom_parent is None:
        continue
      mesh_el = geom_parent.find("geometry/mesh")
      if mesh_el is None:
        continue
      fn = mesh_el.get("filename", "")
      if not fn.endswith(".stl"):
        continue
      stem = Path(fn).stem
      mesh_el.set("filename", f"{glb_dir}/{stem}.glb")
  out = profile.assets_dir / profile.visual_glb_urdf
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def vendor_robot(key: str, xarm_desc: Path, work_root: Path) -> None:
  profile = ROBOT_PROFILES[key]
  spec = VENDOR_SPECS[key]
  work_desc = _preprocess_xarm_description(xarm_desc, work_root / key / "xarm_description")
  default_kin = _default_kinematics_yaml(xarm_desc, spec)
  kin_dest = profile.assets_dir / "kinematics" / "default" / spec["default_kinematics"]
  kin_dest.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(default_kin, kin_dest)
  wrapper = work_root / key / "gen.urdf.xacro"
  _write_wrapper_xacro(work_desc, spec, default_kin, wrapper)
  urdf_text = _run_xacro(wrapper)
  urdf_text = _normalize_mesh_paths(urdf_text, profile.mesh_variant)
  urdf_text = _add_world_link(urdf_text)

  profile.assets_dir.mkdir(parents=True, exist_ok=True)
  (profile.assets_dir / "kinematics" / "user").mkdir(parents=True, exist_ok=True)
  (profile.assets_dir / "kinematics" / "user" / ".gitkeep").touch(exist_ok=True)

  base_urdf = profile.assets_dir / profile.default_urdf
  base_urdf.write_text(urdf_text, encoding="utf-8")
  _copy_meshes(work_desc, profile, spec["mesh_ros"])
  glb_urdf = _make_visual_glb_urdf(base_urdf, profile)
  print(f"[{key}] wrote {base_urdf}")
  print(f"[{key}] wrote {glb_urdf}")


def vendor_bio_gripper(xarm_desc: Path, work_root: Path) -> None:
  work_desc = _preprocess_xarm_description(xarm_desc, work_root / "bio" / "xarm_description")
  wrapper = work_root / "bio" / "gen.urdf.xacro"
  content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="bio_gripper">
  <xacro:property name="is_ros2" value="true"/>
  <xacro:property name="use_xacro_load_yaml" value="true"/>
  <xacro:property name="use_len" value="true"/>
  <xacro:property name="mesh_suffix" value="stl"/>
  <xacro:property name="mesh_path" value="meshes"/>
  <xacro:include filename="{work_desc / 'urdf/common/common.material.xacro'}"/>
  <xacro:common_material prefix=""/>
  <xacro:include filename="{work_desc / 'urdf/common/common.link.xacro'}"/>
  <xacro:include filename="{work_desc / 'urdf/gripper/bio_gripper.urdf.xacro'}"/>
  <xacro:bio_gripper_urdf prefix="" attach_to="link_tool" attach_xyz="0 0 0" attach_rpy="0 0 0"/>
</robot>
"""
  wrapper.write_text(content, encoding="utf-8")
  urdf_text = _run_xacro(wrapper)
  assets = PROJECT_ROOT / "assets" / "urdf" / "bio_gripper"
  assets.mkdir(parents=True, exist_ok=True)
  out = assets / "bio_gripper.urdf"
  out.write_text(urdf_text, encoding="utf-8")

  src_visual = work_desc / "meshes" / "gripper" / "bio"
  dst_visual = assets / "meshes" / "visual"
  dst_visual.mkdir(parents=True, exist_ok=True)
  if src_visual.is_dir():
    for path in src_visual.rglob("*.stl"):
      shutil.copy2(path, dst_visual / path.name.lower())
  print(f"[bio_gripper] wrote {out}")


def vendor_gripper_g2(xarm_desc: Path, work_root: Path) -> None:
  work_desc = _preprocess_xarm_description(xarm_desc, work_root / "g2" / "xarm_description")
  wrapper = work_root / "g2" / "gen.urdf.xacro"
  content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="gripper_g2">
  <xacro:property name="is_ros2" value="true"/>
  <xacro:property name="use_xacro_load_yaml" value="true"/>
  <xacro:property name="use_len" value="true"/>
  <xacro:property name="mesh_suffix" value="stl"/>
  <xacro:property name="mesh_path" value="meshes"/>
  <xacro:include filename="{work_desc / 'urdf/common/common.material.xacro'}"/>
  <xacro:common_material prefix=""/>
  <xacro:include filename="{work_desc / 'urdf/common/common.link.xacro'}"/>
  <xacro:include filename="{work_desc / 'urdf/gripper/xarm_gripper.urdf.xacro'}"/>
  <xacro:xarm_gripper_urdf prefix="" attach_to="link_tool" attach_xyz="0 0 0" attach_rpy="0 0 0"/>
</robot>
"""
  wrapper.write_text(content, encoding="utf-8")
  urdf_text = _run_xacro(wrapper)
  assets = PROJECT_ROOT / "assets" / "urdf" / "gripper_g2"
  assets.mkdir(parents=True, exist_ok=True)
  out = assets / "gripper_g2.urdf"
  out.write_text(urdf_text, encoding="utf-8")

  src_collision = work_desc / "meshes" / "gripper"
  dst_collision = assets / "meshes" / "collision"
  dst_collision.mkdir(parents=True, exist_ok=True)
  if src_collision.is_dir():
    for path in src_collision.rglob("*.stl"):
      shutil.copy2(path, dst_collision / path.name.lower())
    for path in src_collision.rglob("*.STL"):
      shutil.copy2(path, dst_collision / path.name.lower())
  print(f"[gripper_g2] wrote {out}")


def migrate_sim_glbs(sim_root: Path) -> None:
  mapping = {
    "xarm5_1305": sim_root / "xarm5_1305",
    "xarm7_1305": sim_root / "xarm7_1305",
    "lite6": sim_root / "lite6",
    "uf850": sim_root / "850",
  }
  for key, src in mapping.items():
    if not src.is_dir():
      print(f"[skip] missing sim dir: {src}")
      continue
    profile = ROBOT_PROFILES[key]
    dst = profile.assets_dir / "meshes" / profile.mesh_variant / "visual_glb_src"
    dst.mkdir(parents=True, exist_ok=True)
    for glb in sorted(src.glob("*.glb")):
      shutil.copy2(glb, dst / glb.name)
    print(f"[{key}] copied GLB src -> {dst}")

  bio_src = sim_root / "bio_gripper_g2" / "bio_gripper_g2.glb"
  if bio_src.is_file():
    bio_dst = PROJECT_ROOT / "assets" / "urdf" / "bio_gripper" / "meshes" / "visual" / "visual_glb_src"
    bio_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bio_src, bio_dst / "bio_gripper_g2.glb")
    print(f"[bio_gripper] copied {bio_src} -> {bio_dst}")

  g2_src = sim_root / "gripper_g2"
  if g2_src.is_dir():
    g2_dst = PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "meshes" / "visual" / "visual_glb_src"
    g2_dst.mkdir(parents=True, exist_ok=True)
    for name in ("gripper_g2.glb", "gripper_g2_movable.glb"):
      src = g2_src / name
      if src.is_file():
        shutil.copy2(src, g2_dst / name)
        print(f"[gripper_g2] copied {src} -> {g2_dst}")

  lite6_gripper_src = sim_root / "lite6_gripper" / "lite_gripper.glb"
  if lite6_gripper_src.is_file():
    lite6_g_dst = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "meshes" / "visual" / "visual_glb_src"
    lite6_g_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lite6_gripper_src, lite6_g_dst / "lite_gripper.glb")
    print(f"[lite6_gripper] copied {lite6_gripper_src} -> {lite6_g_dst}")

  lite6_vac_src = sim_root / "lite6_vacuum_gripper" / "lite_vacuum_gripper.glb"
  if lite6_vac_src.is_file():
    lite6_v_dst = PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "meshes" / "visual" / "visual_glb_src"
    lite6_v_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lite6_vac_src, lite6_v_dst / "lite_vacuum_gripper.glb")
    print(f"[lite6_vacuum_gripper] copied {lite6_vac_src} -> {lite6_v_dst}")


def _normalize_gripper_mesh_paths(urdf_text: str, assets_subdir: str) -> str:
  """Rewrite xarm_description mesh paths to project-relative collision paths."""

  def _collision_path(match: re.Match[str]) -> str:
    stem = Path(match.group(1)).stem
    return f'filename="meshes/collision/{stem}.stl"'

  urdf_text = re.sub(
    r'filename="[^"]*gripper/lite/visual/([^"]+)"',
    _collision_path,
    urdf_text,
  )
  urdf_text = re.sub(
    r'filename="[^"]*vacuum_gripper/lite/visual/([^"]+)"',
    _collision_path,
    urdf_text,
  )
  urdf_text = re.sub(
    r'filename="[^"]*vacuum_gripper/lite/collision/([^"]+)"',
    _collision_path,
    urdf_text,
  )
  return urdf_text


def vendor_lite6_gripper(xarm_desc: Path, work_root: Path) -> None:
  work_desc = _preprocess_xarm_description(xarm_desc, work_root / "lite6g" / "xarm_description")
  wrapper = work_root / "lite6g" / "gen.urdf.xacro"
  content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="lite6_gripper">
  <xacro:property name="is_ros2" value="true"/>
  <xacro:property name="use_xacro_load_yaml" value="true"/>
  <xacro:property name="use_len" value="true"/>
  <xacro:property name="mesh_suffix" value="stl"/>
  <xacro:property name="mesh_path" value="meshes"/>
  <xacro:include filename="{work_desc / 'urdf/common/common.material.xacro'}"/>
  <xacro:common_material prefix=""/>
  <xacro:include filename="{work_desc / 'urdf/common/common.link.xacro'}"/>
  <xacro:include filename="{work_desc / 'urdf/gripper/uflite_gripper.urdf.xacro'}"/>
  <xacro:uflite_gripper_urdf prefix="" attach_to="link6" attach_xyz="0 0 0" attach_rpy="0 0 0"/>
</robot>
"""
  wrapper.write_text(content, encoding="utf-8")
  urdf_text = _run_xacro(wrapper)
  urdf_text = _normalize_gripper_mesh_paths(urdf_text, "lite6_gripper")
  assets = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper"
  assets.mkdir(parents=True, exist_ok=True)
  out = assets / "lite6_gripper.urdf"
  out.write_text(urdf_text, encoding="utf-8")

  src_visual = work_desc / "meshes" / "gripper" / "lite" / "visual"
  dst_collision = assets / "meshes" / "collision"
  dst_collision.mkdir(parents=True, exist_ok=True)
  if src_visual.is_dir():
    for path in src_visual.rglob("*.stl"):
      shutil.copy2(path, dst_collision / path.name.lower())
    for path in src_visual.rglob("*.STL"):
      shutil.copy2(path, dst_collision / path.name.lower())
  print(f"[lite6_gripper] wrote {out}")


def vendor_lite6_vacuum_gripper(xarm_desc: Path, work_root: Path) -> None:
  work_desc = _preprocess_xarm_description(xarm_desc, work_root / "lite6v" / "xarm_description")
  wrapper = work_root / "lite6v" / "gen.urdf.xacro"
  content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="lite6_vacuum_gripper">
  <xacro:property name="is_ros2" value="true"/>
  <xacro:property name="use_xacro_load_yaml" value="true"/>
  <xacro:property name="use_len" value="true"/>
  <xacro:property name="mesh_suffix" value="stl"/>
  <xacro:property name="mesh_path" value="meshes"/>
  <xacro:include filename="{work_desc / 'urdf/common/common.material.xacro'}"/>
  <xacro:common_material prefix=""/>
  <xacro:include filename="{work_desc / 'urdf/common/common.link.xacro'}"/>
  <xacro:include filename="{work_desc / 'urdf/vacuum_gripper/lite_vacuum_gripper.urdf.xacro'}"/>
  <xacro:uflite_vacuum_gripper_urdf prefix="" attach_to="link6" attach_xyz="0 0 0" attach_rpy="0 0 0"/>
</robot>
"""
  wrapper.write_text(content, encoding="utf-8")
  urdf_text = _run_xacro(wrapper)
  urdf_text = _normalize_gripper_mesh_paths(urdf_text, "lite6_vacuum_gripper")
  assets = PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper"
  assets.mkdir(parents=True, exist_ok=True)
  out = assets / "lite6_vacuum_gripper.urdf"
  out.write_text(urdf_text, encoding="utf-8")

  for sub in ("visual", "collision"):
    src = work_desc / "meshes" / "vacuum_gripper" / "lite" / sub
    dst_collision = assets / "meshes" / "collision"
    dst_collision.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
      for path in src.rglob("*.stl"):
        shutil.copy2(path, dst_collision / path.name.lower())
  print(f"[lite6_vacuum_gripper] wrote {out}")


def main() -> int:
  parser = argparse.ArgumentParser(description="Vendor robot URDF/meshes from xarm_ros2")
  parser.add_argument("--xarm-description", type=Path, default=None, help="Path to xarm_description root")
  parser.add_argument("--sim-root", type=Path, default=None, help="Path to source CAD GLB files (e.g. sim repo root)")
  parser.add_argument("--robots", nargs="*", default=list(VENDOR_SPECS.keys()))
  parser.add_argument("--skip-urdf", action="store_true")
  parser.add_argument("--skip-glb-migrate", action="store_true")
  parser.add_argument("--bio-gripper", action="store_true", default=True)
  parser.add_argument("--gripper-g2", action="store_true", default=True)
  parser.add_argument("--lite6-gripper", action="store_true", default=True)
  parser.add_argument("--lite6-vacuum-gripper", action="store_true", default=True)
  args = parser.parse_args()

  xarm_desc = _ensure_xarm_ros2(args.xarm_description)
  work_root = Path(tempfile.mkdtemp(prefix="ufactory_vendor_"))

  if not args.skip_glb_migrate and args.sim_root is not None and args.sim_root.is_dir():
    migrate_sim_glbs(args.sim_root)
  elif not args.skip_glb_migrate and args.sim_root is None:
    print("[vendor] --sim-root not provided; skipping GLB migration. "
          "Pass --sim-root <path> to migrate source CAD GLBs.")

  if not args.skip_urdf:
    for key in args.robots:
      if key not in VENDOR_SPECS:
        raise SystemExit(f"Unknown robot key: {key}")
      vendor_robot(key, xarm_desc, work_root)
    if args.bio_gripper:
      vendor_bio_gripper(xarm_desc, work_root)
    if args.gripper_g2:
      vendor_gripper_g2(xarm_desc, work_root)
    if args.lite6_gripper:
      vendor_lite6_gripper(xarm_desc, work_root)
    if args.lite6_vacuum_gripper:
      vendor_lite6_vacuum_gripper(xarm_desc, work_root)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
