#!/usr/bin/env python3
"""Generate arm + Gripper G2 visual combo URDFs (static and movable)."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import GRIPPER_G2_ASSETS, PROJECT_ROOT
from ufactory.robot_registry import ROBOT_PROFILES

G2_COLLISION = "../gripper_g2/meshes/collision"
G2_VISUAL = "../gripper_g2/meshes/visual"

GRIPPER_LINK_MESHES: dict[str, tuple[str | None, str]] = {
  "xarm_gripper_base_link": ("base.glb", "base_link.stl"),
  "left_outer_knuckle": ("left_outer_knuckle.glb", "left_outer_knuckle.stl"),
  "left_finger": ("left_finger.glb", "left_finger.stl"),
  "left_inner_knuckle": ("left_inner_knuckle.glb", "left_inner_knuckle.stl"),
  "right_outer_knuckle": ("right_outer_knuckle.glb", "right_outer_knuckle.stl"),
  "right_finger": ("right_finger.glb", "right_finger.stl"),
  "right_inner_knuckle": ("right_inner_knuckle.glb", "right_inner_knuckle.stl"),
}

_TEMPLATE_URDF = GRIPPER_G2_ASSETS / "gripper_g2.urdf"
_SKIP_JOINTS = frozenset({"gripper_attach", "gripper_fix"})


def _load_gripper_subtree() -> list[ET.Element]:
  if not _TEMPLATE_URDF.is_file():
    raise FileNotFoundError(f"Missing gripper template: {_TEMPLATE_URDF}")
  root = ET.parse(_TEMPLATE_URDF).getroot()
  elems: list[ET.Element] = []
  capture = False
  for child in root:
    if child.tag == "link" and child.get("name") == "xarm_gripper_base_link":
      capture = True
    if not capture:
      continue
    if child.tag == "joint" and child.get("name") in _SKIP_JOINTS:
      continue
    elems.append(copy.deepcopy(child))
  if not elems:
    raise RuntimeError("gripper_g2.urdf has no xarm_gripper_base_link subtree")
  return elems


def _set_mesh(geom_parent: ET.Element, filename: str) -> None:
  geom = geom_parent.find("geometry")
  if geom is None:
    geom = ET.SubElement(geom_parent, "geometry")
  mesh = geom.find("mesh")
  if mesh is None:
    mesh = ET.SubElement(geom, "mesh")
  mesh.set("filename", filename)


def _collision_path(stl_name: str) -> str:
  return f"{G2_COLLISION}/{stl_name}"


def _movable_visual_path(ee_link: str, glb_name: str) -> str:
  if glb_name == "base.glb":
    return f"{G2_VISUAL}/visual_glb/{ee_link}/base.glb"
  return f"{G2_VISUAL}/visual_glb/{glb_name}"


def _static_visual_path(ee_link: str) -> str:
  return f"{G2_VISUAL}/gripper_g2_static_{ee_link}.glb"


def _append_gripper_static_overlay(root: ET.Element, ee_link: str) -> None:
  tool = ET.SubElement(root, "link", {"name": "gripper_g2_visual"})
  visual = ET.SubElement(tool, "visual")
  ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
  _set_mesh(visual, _static_visual_path(ee_link))
  ET.SubElement(visual, "material", {"name": "White"})

  joint = ET.SubElement(root, "joint", {"name": "gripper_g2_visual_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "gripper_g2_visual"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})


def _rewrite_gripper_link(link: ET.Element, ee_link: str, *, movable: bool) -> ET.Element:
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
      _set_mesh(col, _collision_path(stl_name))
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
      _set_mesh(vis, _movable_visual_path(ee_link, glb_name))
  return out


def _append_gripper_kinematics(root: ET.Element, ee_link: str, *, movable: bool) -> None:
  joint = ET.SubElement(root, "joint", {"name": "gripper_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "xarm_gripper_base_link"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, ee_link, movable=movable))
    else:
      root.append(copy.deepcopy(elem))


def generate_for_profile(key: str, *, movable: bool) -> Path:
  profile = ROBOT_PROFILES[key]
  if not profile.supports_gripper_g2:
    raise ValueError(f"{key} does not support G2 gripper")
  urdf_name = profile.gripper_g2_movable_visual_urdf if movable else profile.gripper_g2_visual_urdf
  if not urdf_name:
    raise ValueError(f"{key} missing G2 URDF name")

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


def generate_standalone_movable(ee_link: str = "link6") -> Path:
  """Gripper-only movable visual URDF (no arm) for open/close debugging."""
  root = ET.Element("robot", {"name": "gripper_g2_movable_visual"})
  ET.SubElement(root, "link", {"name": "world"})
  world_joint = ET.SubElement(root, "joint", {"name": "world_joint", "type": "fixed"})
  ET.SubElement(world_joint, "parent", {"link": "world"})
  ET.SubElement(world_joint, "child", {"link": "xarm_gripper_base_link"})
  ET.SubElement(world_joint, "origin", {"xyz": "0 0 0.12", "rpy": "0 0 0"})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, ee_link, movable=True))
    else:
      root.append(copy.deepcopy(elem))

  out = GRIPPER_G2_ASSETS / "gripper_g2_movable_visual.urdf"
  tree = ET.ElementTree(root)
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def main() -> int:
  keys = [
    k
    for k, p in ROBOT_PROFILES.items()
    if p.supports_gripper_g2 and p.gripper_g2_visual_urdf and p.gripper_g2_movable_visual_urdf
  ]
  for key in keys:
    static_out = generate_for_profile(key, movable=False)
    movable_out = generate_for_profile(key, movable=True)
    print(f"[{key}] wrote {static_out}")
    print(f"[{key}] wrote {movable_out}")
  standalone = generate_standalone_movable()
  print(f"[standalone] wrote {standalone}")
  if not _TEMPLATE_URDF.is_file():
    print(f"Warning: missing {_TEMPLATE_URDF}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
