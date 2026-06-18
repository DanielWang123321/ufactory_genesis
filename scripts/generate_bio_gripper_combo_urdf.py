#!/usr/bin/env python3
"""Generate arm + Bio Gripper G2 visual combo URDFs (static and movable)."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import BIO_GRIPPER_ASSETS
from ufactory.robot_registry import ROBOT_PROFILES

GRIPPER_VISUAL = "../bio_gripper/meshes/visual"
BIO_GRIPPER_MOUNT_RPY = "3.14159265 0 0"

GRIPPER_LINK_MESHES: dict[str, tuple[str | None, str]] = {
  "bio_gripper_base_link": ("bio_gripper_base.glb", "link_base.stl"),
  "bio_gripper_left_finger": ("bio_left_finger.glb", "left_finger.stl"),
  "bio_gripper_right_finger": ("bio_right_finger.glb", "right_finger.stl"),
}

_TEMPLATE_URDF = BIO_GRIPPER_ASSETS / "bio_gripper.urdf"
_SKIP_JOINTS = frozenset({"bio_gripper_fix"})


def _load_gripper_subtree() -> list[ET.Element]:
  if not _TEMPLATE_URDF.is_file():
    raise FileNotFoundError(f"Missing gripper template: {_TEMPLATE_URDF}")
  root = ET.parse(_TEMPLATE_URDF).getroot()
  elems: list[ET.Element] = []
  capture = False
  for child in root:
    if child.tag == "link" and child.get("name") == "bio_gripper_base_link":
      capture = True
    if not capture:
      continue
    if child.tag == "joint" and child.get("name") in _SKIP_JOINTS:
      continue
    elems.append(copy.deepcopy(child))
  if not elems:
    raise RuntimeError("bio_gripper.urdf has no bio_gripper_base_link subtree")
  return elems


def _set_mesh(geom_parent: ET.Element, filename: str) -> None:
  geom = geom_parent.find("geometry")
  if geom is None:
    geom = ET.SubElement(geom_parent, "geometry")
  mesh = geom.find("mesh")
  if mesh is None:
    mesh = ET.SubElement(geom, "mesh")
  mesh.set("filename", filename)


def _bio_gripper_glb_path(ee_link: str) -> str:
  return f"{GRIPPER_VISUAL}/bio_gripper_g2_visual_{ee_link}.glb"


def _movable_visual_path(ee_link: str, glb_name: str) -> str:
  return f"{GRIPPER_VISUAL}/visual_glb/{ee_link}/{glb_name}"


def _append_bio_gripper_static_overlay(root: ET.Element, ee_link: str) -> None:
  tool = ET.SubElement(root, "link", {"name": "bio_gripper_visual"})
  visual = ET.SubElement(tool, "visual")
  ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": BIO_GRIPPER_MOUNT_RPY})
  _set_mesh(visual, _bio_gripper_glb_path(ee_link))

  joint = ET.SubElement(root, "joint", {"name": "bio_gripper_visual_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "bio_gripper_visual"})
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
      _set_mesh(col, f"{GRIPPER_VISUAL}/{stl_name}")
    elif child.tag == "visual":
      if not movable or meshes is None:
        continue
      glb_name, _ = meshes
      if glb_name is None:
        continue
      vis = ET.SubElement(out, "visual")
      origin = child.find("origin")
      xyz = "0 0 0"
      if origin is not None and origin.get("xyz"):
        xyz = origin.get("xyz", "0 0 0")
      finger_links = frozenset({"bio_gripper_left_finger", "bio_gripper_right_finger"})
      vis_rpy = "0 0 0" if link_name in finger_links else BIO_GRIPPER_MOUNT_RPY
      ET.SubElement(vis, "origin", {"xyz": xyz, "rpy": vis_rpy})
      _set_mesh(vis, _movable_visual_path(ee_link, glb_name))
  return out


def _append_gripper_kinematics(root: ET.Element, ee_link: str, *, movable: bool) -> None:
  joint = ET.SubElement(root, "joint", {"name": "bio_gripper_attach", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "bio_gripper_base_link"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, ee_link, movable=movable))
    else:
      root.append(copy.deepcopy(elem))


def generate_for_profile(key: str, *, movable: bool) -> Path:
  profile = ROBOT_PROFILES[key]
  if not profile.supports_bio_gripper_g2:
    raise ValueError(f"{key} does not support bio gripper combo URDF")
  urdf_name = (
    profile.bio_gripper_g2_movable_visual_urdf
    if movable
    else profile.bio_gripper_g2_visual_urdf
  )
  if not urdf_name:
    raise ValueError(f"{key} missing Bio Gripper G2 URDF name")

  src = profile.assets_dir / profile.visual_glb_urdf
  tree = ET.parse(str(src))
  root = tree.getroot()
  if not movable:
    _append_bio_gripper_static_overlay(root, profile.ee_link)
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
  root = ET.Element("robot", {"name": "bio_gripper_movable_visual"})
  ET.SubElement(root, "link", {"name": "world"})
  world_joint = ET.SubElement(root, "joint", {"name": "world_joint", "type": "fixed"})
  ET.SubElement(world_joint, "parent", {"link": "world"})
  ET.SubElement(world_joint, "child", {"link": "bio_gripper_base_link"})
  # Elevate gripper; per-link visual origins apply BIO_GRIPPER_MOUNT_RPY (same as arm combo movable).
  ET.SubElement(world_joint, "origin", {"xyz": "0 0 0.12", "rpy": "0 0 0"})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, ee_link, movable=True))
    else:
      root.append(copy.deepcopy(elem))

  out = BIO_GRIPPER_ASSETS / "bio_gripper_movable_visual.urdf"
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
    if p.supports_bio_gripper_g2
    and p.bio_gripper_g2_visual_urdf
    and p.bio_gripper_g2_movable_visual_urdf
  ]
  for key in keys:
    static_out = generate_for_profile(key, movable=False)
    movable_out = generate_for_profile(key, movable=True)
    print(f"[{key}] wrote {static_out}")
    print(f"[{key}] wrote {movable_out}")
  standalone = generate_standalone_movable()
  print(f"[standalone] wrote {standalone}")
  src_glb = BIO_GRIPPER_ASSETS / "meshes" / "visual" / "visual_glb_src" / "bio_gripper_g2.glb"
  if not src_glb.is_file():
    print(f"Warning: missing {src_glb}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
