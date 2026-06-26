#!/usr/bin/env python3
"""Generate arm + Bio Gripper G2 visual combo URDFs (static and movable)."""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import BIO_GRIPPER_G2_ASSETS
from ufactory.robot_registry import ROBOT_PROFILES

GRIPPER_VISUAL = "../bio_gripper_g2/meshes/visual"
METRICS_PATH = BIO_GRIPPER_G2_ASSETS / "meshes" / "visual" / "relocalize_metrics.json"
BIO_GRIPPER_G2_STANDALONE_RPY = "0 0 0"

GRIPPER_LINK_MESHES: dict[str, tuple[str | None, str]] = {
  "bio_gripper_g2_base_link": ("bio_gripper_g2_base.glb", "link_base.stl"),
  "bio_gripper_g2_left_finger": ("bio_gripper_g2_left_finger.glb", "left_finger.stl"),
  "bio_gripper_g2_right_finger": ("bio_gripper_g2_right_finger.glb", "right_finger.stl"),
}

_TEMPLATE_URDF = BIO_GRIPPER_G2_ASSETS / "bio_gripper_g2.urdf"
_SKIP_JOINTS = frozenset({"bio_gripper_g2_fix"})


def _load_attach_origins() -> dict[str, dict[str, str]]:
  """Attach origins from relocalize metrics (per robot_key, fallback ee_link)."""
  if not METRICS_PATH.is_file():
    raise FileNotFoundError(
      f"Missing {METRICS_PATH}; run scripts/relocalize_bio_gripper_g2_glb.py first"
    )
  report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
  out: dict[str, dict[str, str]] = {}
  for entry in report:
    xyz = entry.get("attach_xyz_str")
    rpy = entry.get("attach_rpy_str")
    if not xyz or not rpy:
      continue
    origin = {"xyz": xyz, "rpy": rpy}
    robot_key = entry.get("robot_key")
    ee = entry.get("ee_link")
    if robot_key:
      out[robot_key] = origin
    elif ee:
      out.setdefault(ee, origin)
  if not out:
    raise RuntimeError(f"No attach origins in {METRICS_PATH}")
  return out


def _attach_origin_for(
  attach_origins: dict[str, dict[str, str]],
  *,
  robot_key: str,
  ee_link: str,
) -> dict[str, str]:
  origin = attach_origins.get(robot_key) or attach_origins.get(ee_link)
  if origin is None:
    raise KeyError(f"No Bio Gripper G2 attach origin for {robot_key}/{ee_link} in {METRICS_PATH}")
  return origin


def _load_gripper_subtree() -> list[ET.Element]:
  if not _TEMPLATE_URDF.is_file():
    raise FileNotFoundError(f"Missing gripper template: {_TEMPLATE_URDF}")
  root = ET.parse(_TEMPLATE_URDF).getroot()
  elems: list[ET.Element] = []
  capture = False
  for child in root:
    if child.tag == "link" and child.get("name") == "bio_gripper_g2_base_link":
      capture = True
    if not capture:
      continue
    if child.tag == "joint" and child.get("name") in _SKIP_JOINTS:
      continue
    elems.append(copy.deepcopy(child))
  if not elems:
    raise RuntimeError("bio_gripper_g2.urdf has no bio_gripper_g2_base_link subtree")
  return elems


def _set_mesh(geom_parent: ET.Element, filename: str) -> None:
  geom = geom_parent.find("geometry")
  if geom is None:
    geom = ET.SubElement(geom_parent, "geometry")
  mesh = geom.find("mesh")
  if mesh is None:
    mesh = ET.SubElement(geom, "mesh")
  mesh.set("filename", filename)


def _bio_gripper_g2_glb_path(ee_link: str) -> str:
  return f"{GRIPPER_VISUAL}/bio_gripper_g2_visual_{ee_link}.glb"


def _movable_visual_path(ee_link: str, glb_name: str) -> str:
  return f"{GRIPPER_VISUAL}/visual_glb/{ee_link}/{glb_name}"


def _append_bio_gripper_g2_static_overlay(
  root: ET.Element,
  ee_link: str,
  origin: dict[str, str],
) -> None:
  tool = ET.SubElement(root, "link", {"name": "bio_gripper_g2_visual"})
  visual = ET.SubElement(tool, "visual")
  ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
  _set_mesh(visual, _bio_gripper_g2_glb_path(ee_link))

  joint = ET.SubElement(root, "joint", {"name": "bio_gripper_g2_visual_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "bio_gripper_g2_visual"})
  ET.SubElement(joint, "origin", {"xyz": origin["xyz"], "rpy": origin["rpy"]})


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
      ET.SubElement(vis, "origin", {"xyz": xyz, "rpy": "0 0 0"})
      _set_mesh(vis, _movable_visual_path(ee_link, glb_name))
  return out


def _append_gripper_kinematics(
  root: ET.Element,
  ee_link: str,
  *,
  movable: bool,
  origin: dict[str, str],
) -> None:
  joint = ET.SubElement(root, "joint", {"name": "bio_gripper_g2_attach", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "bio_gripper_g2_base_link"})
  ET.SubElement(joint, "origin", {"xyz": origin["xyz"], "rpy": origin["rpy"]})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, ee_link, movable=movable))
    else:
      root.append(copy.deepcopy(elem))


def generate_for_profile(key: str, *, movable: bool, attach_origins: dict[str, dict[str, str]] | None = None) -> Path:
  profile = ROBOT_PROFILES[key]
  if not profile.supports_bio_gripper_g2:
    raise ValueError(f"{key} does not support Bio Gripper G2 combo URDF")
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
  if attach_origins is None:
    attach_origins = _load_attach_origins()
  origin = _attach_origin_for(attach_origins, robot_key=key, ee_link=profile.ee_link)
  if not movable:
    _append_bio_gripper_g2_static_overlay(root, profile.ee_link, origin)
  _append_gripper_kinematics(
    root, profile.ee_link, movable=movable, origin=origin
  )
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
  root = ET.Element("robot", {"name": "bio_gripper_g2_movable_visual"})
  ET.SubElement(root, "link", {"name": "world"})
  world_joint = ET.SubElement(root, "joint", {"name": "world_joint", "type": "fixed"})
  ET.SubElement(world_joint, "parent", {"link": "world"})
  ET.SubElement(world_joint, "child", {"link": "bio_gripper_g2_base_link"})
  ET.SubElement(world_joint, "origin", {"xyz": "0 0 0.12", "rpy": BIO_GRIPPER_G2_STANDALONE_RPY})

  for elem in _load_gripper_subtree():
    if elem.tag == "link":
      root.append(_rewrite_gripper_link(elem, ee_link, movable=True))
    else:
      root.append(copy.deepcopy(elem))

  out = BIO_GRIPPER_G2_ASSETS / "bio_gripper_g2_movable_visual.urdf"
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
  attach_origins = _load_attach_origins()
  for key in keys:
    static_out = generate_for_profile(key, movable=False, attach_origins=attach_origins)
    movable_out = generate_for_profile(key, movable=True, attach_origins=attach_origins)
    print(f"[{key}] wrote {static_out}")
    print(f"[{key}] wrote {movable_out}")
  standalone = generate_standalone_movable()
  print(f"[standalone] wrote {standalone}")
  src_glb = BIO_GRIPPER_G2_ASSETS / "meshes" / "visual" / "visual_glb_src" / "bio_gripper_g2.glb"
  if not src_glb.is_file():
    print(f"Warning: missing {src_glb}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
