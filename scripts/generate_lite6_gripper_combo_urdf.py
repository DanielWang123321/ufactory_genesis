#!/usr/bin/env python3
"""Generate Lite6 arm + Lite6 gripper visual combo URDFs (static and movable)."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import LITE6_GRIPPER_ASSETS
from ufactory.robot_registry import ROBOT_PROFILES

GRIPPER_COLLISION = "../lite6_gripper/meshes/collision"
GRIPPER_VISUAL = "../lite6_gripper/meshes/visual"

GRIPPER_LINK_MESHES: dict[str, tuple[str | None, str]] = {
  "uflite_gripper_link": ("shell.glb", "shell.stl"),
  "uflite_finger1": ("finger1.glb", "finger1.stl"),
  "uflite_finger2": ("finger2.glb", "finger2.stl"),
}

_TEMPLATE_URDF = LITE6_GRIPPER_ASSETS / "lite6_gripper.urdf"
_SKIP_JOINTS = frozenset({"gripper_fix"})


def _load_gripper_subtree() -> list[ET.Element]:
  if not _TEMPLATE_URDF.is_file():
    raise FileNotFoundError(f"Missing gripper template: {_TEMPLATE_URDF}")
  root = ET.parse(_TEMPLATE_URDF).getroot()
  elems: list[ET.Element] = []
  capture = False
  for child in root:
    if child.tag == "link" and child.get("name") == "uflite_gripper_link":
      capture = True
    if not capture:
      continue
    if child.tag == "joint" and child.get("name") in _SKIP_JOINTS:
      continue
    elems.append(copy.deepcopy(child))
  if not elems:
    raise RuntimeError("lite6_gripper.urdf has no uflite_gripper_link subtree")
  return elems


def _set_mesh(geom_parent: ET.Element, filename: str) -> None:
  geom = geom_parent.find("geometry")
  if geom is None:
    geom = ET.SubElement(geom_parent, "geometry")
  mesh = geom.find("mesh")
  if mesh is None:
    mesh = ET.SubElement(geom, "mesh")
  mesh.set("filename", filename)


def _append_gripper_static_overlay(root: ET.Element, ee_link: str) -> None:
  tool = ET.SubElement(root, "link", {"name": "lite6_gripper_visual"})
  visual = ET.SubElement(tool, "visual")
  ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
  _set_mesh(visual, f"{GRIPPER_VISUAL}/lite6_gripper_static_{ee_link}.glb")
  ET.SubElement(visual, "material", {"name": "White"})

  joint = ET.SubElement(root, "joint", {"name": "lite6_gripper_visual_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "lite6_gripper_visual"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})


def _rewrite_gripper_link(link: ET.Element, *, movable: bool) -> ET.Element:
  link_name = link.get("name", "")
  meshes = GRIPPER_LINK_MESHES.get(link_name)
  out = ET.Element("link", {"name": link_name})
  for child in link:
    if child.tag == "inertial":
      out.append(copy.deepcopy(child))
    elif child.tag == "collision":
      if meshes is None:
        out.append(copy.deepcopy(child))
        continue
      _, stl_name = meshes
      col = ET.SubElement(out, "collision")
      origin = child.find("origin")
      if origin is not None:
        ET.SubElement(col, "origin", dict(origin.attrib))
      else:
        ET.SubElement(col, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
      _set_mesh(col, f"{GRIPPER_COLLISION}/{stl_name}")
    elif child.tag == "visual":
      if not movable or meshes is None:
        continue
      glb_name, _ = meshes
      if glb_name is None:
        continue
      vis = ET.SubElement(out, "visual")
      origin = child.find("origin")
      if origin is not None:
        ET.SubElement(vis, "origin", dict(origin.attrib))
      else:
        ET.SubElement(vis, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
      _set_mesh(vis, f"{GRIPPER_VISUAL}/visual_glb/{glb_name}")
  return out


def _append_gripper_kinematics(root: ET.Element, ee_link: str, *, movable: bool) -> None:
  joint = ET.SubElement(root, "joint", {"name": "gripper_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "uflite_gripper_link"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, movable=movable))
    else:
      root.append(copy.deepcopy(elem))


def generate_for_profile(*, movable: bool) -> Path:
  profile = ROBOT_PROFILES["lite6"]
  if not profile.supports_lite6_gripper:
    raise ValueError("lite6 does not support Lite6 Gripper")
  urdf_name = (
    profile.lite6_gripper_movable_visual_urdf
    if movable
    else profile.lite6_gripper_visual_urdf
  )
  if not urdf_name:
    raise ValueError("lite6 missing Lite6 Gripper URDF name")

  src = profile.assets_dir / profile.visual_glb_urdf
  tree = ET.parse(str(src))
  root = tree.getroot()
  if not movable:
    _append_gripper_static_overlay(root, profile.ee_link)
  _append_gripper_kinematics(root, profile.ee_link, movable=movable)
  root.set("name", Path(urdf_name).stem)

  out = profile.assets_dir / urdf_name
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def generate_standalone_movable() -> Path:
  root = ET.Element("robot", {"name": "lite6_gripper_movable_visual"})
  ET.SubElement(root, "link", {"name": "world"})
  world_joint = ET.SubElement(root, "joint", {"name": "world_joint", "type": "fixed"})
  ET.SubElement(world_joint, "parent", {"link": "world"})
  ET.SubElement(world_joint, "child", {"link": "uflite_gripper_link"})
  ET.SubElement(world_joint, "origin", {"xyz": "0 0 0.12", "rpy": "0 0 0"})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, movable=True))
    else:
      root.append(copy.deepcopy(elem))

  out = LITE6_GRIPPER_ASSETS / "lite6_gripper_movable_visual.urdf"
  tree = ET.ElementTree(root)
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def main() -> int:
  static_out = generate_for_profile(movable=False)
  movable_out = generate_for_profile(movable=True)
  standalone = generate_standalone_movable()
  print(f"[lite6] wrote {static_out}")
  print(f"[lite6] wrote {movable_out}")
  print(f"[standalone] wrote {standalone}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
