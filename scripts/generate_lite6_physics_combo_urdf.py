#!/usr/bin/env python3
"""Generate Lite6 physics combo URDFs (arm STL + gripper/vacuum subtree)."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import LITE6_GRIPPER_ASSETS, LITE6_VACUUM_GRIPPER_ASSETS
from ufactory.robot_registry import ROBOT_PROFILES

EE_LINK = "link6"
_SKIP_JOINTS_GRIPPER = frozenset({"gripper_fix"})
_SKIP_JOINTS_VACUUM = frozenset({"vacuum_gripper_fix"})


def _load_subtree(template: Path, start_link: str, skip_joints: frozenset[str]) -> list[ET.Element]:
  if not template.is_file():
    raise FileNotFoundError(f"Missing template: {template}")
  root = ET.parse(template).getroot()
  elems: list[ET.Element] = []
  capture = False
  for child in root:
    if child.tag == "link" and child.get("name") == start_link:
      capture = True
    if not capture:
      continue
    if child.tag == "joint" and child.get("name") in skip_joints:
      continue
    elems.append(copy.deepcopy(child))
  if not elems:
    raise RuntimeError(f"{template} has no {start_link} subtree")
  return elems


def _rewrite_mesh_paths(root: ET.Element, prefix: str) -> None:
  for mesh in root.iter("mesh"):
    fn = mesh.get("filename", "")
    if fn.startswith("meshes/collision/"):
      mesh.set("filename", f"{prefix}/{Path(fn).name}")


def _append_accessory(
  arm_root: ET.Element,
  template: Path,
  start_link: str,
  skip_joints: frozenset[str],
  mesh_prefix: str,
  fix_joint_name: str,
) -> None:
  joint = ET.SubElement(arm_root, "joint", {"name": fix_joint_name, "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": EE_LINK})
  ET.SubElement(joint, "child", {"link": start_link})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

  for elem in _load_subtree(template, start_link, skip_joints):
    if elem.tag == "link":
      link = copy.deepcopy(elem)
      _rewrite_mesh_paths(link, mesh_prefix)
      arm_root.append(link)
    else:
      arm_root.append(copy.deepcopy(elem))


def generate_with_gripper() -> Path:
  profile = ROBOT_PROFILES["lite6"]
  if not profile.lite6_with_gripper_urdf:
    raise ValueError("lite6 missing with_gripper URDF name")
  arm_path = profile.assets_dir / profile.default_urdf
  tree = ET.parse(str(arm_path))
  root = tree.getroot()
  _append_accessory(
    root,
    LITE6_GRIPPER_ASSETS / "lite6_gripper.urdf",
    "uflite_gripper_link",
    _SKIP_JOINTS_GRIPPER,
    "../lite6_gripper/meshes/collision",
    "gripper_fix",
  )
  root.set("name", Path(profile.lite6_with_gripper_urdf).stem)
  out = profile.assets_dir / profile.lite6_with_gripper_urdf
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def generate_with_vacuum_gripper() -> Path:
  profile = ROBOT_PROFILES["lite6"]
  if not profile.lite6_with_vacuum_gripper_urdf:
    raise ValueError("lite6 missing with_vacuum_gripper URDF name")
  arm_path = profile.assets_dir / profile.default_urdf
  tree = ET.parse(str(arm_path))
  root = tree.getroot()
  _append_accessory(
    root,
    LITE6_VACUUM_GRIPPER_ASSETS / "lite6_vacuum_gripper.urdf",
    "uflite_vacuum_gripper_link",
    _SKIP_JOINTS_VACUUM,
    "../lite6_vacuum_gripper/meshes/collision",
    "vacuum_gripper_fix",
  )
  root.set("name", Path(profile.lite6_with_vacuum_gripper_urdf).stem)
  out = profile.assets_dir / profile.lite6_with_vacuum_gripper_urdf
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def main() -> int:
  gripper_out = generate_with_gripper()
  vacuum_out = generate_with_vacuum_gripper()
  print(f"[lite6] wrote {gripper_out}")
  print(f"[lite6] wrote {vacuum_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
