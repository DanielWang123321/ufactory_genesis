#!/usr/bin/env python3
"""Generate Lite6 arm + Lite6 vacuum gripper static visual combo URDF."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.robot_registry import ROBOT_PROFILES


def _vacuum_gripper_glb_path(ee_link: str) -> str:
  return f"../lite6_vacuum_gripper/meshes/visual/lite6_vacuum_gripper_visual_{ee_link}.glb"


def _append_vacuum_gripper_visual(root: ET.Element, ee_link: str) -> None:
  tool = ET.SubElement(root, "link", {"name": "lite6_vacuum_gripper_visual"})
  visual = ET.SubElement(tool, "visual")
  ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
  geom = ET.SubElement(visual, "geometry")
  ET.SubElement(geom, "mesh", {"filename": _vacuum_gripper_glb_path(ee_link)})
  ET.SubElement(visual, "material", {"name": "White"})

  joint = ET.SubElement(root, "joint", {"name": "lite6_vacuum_gripper_visual_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "lite6_vacuum_gripper_visual"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})


def generate_for_lite6() -> Path:
  profile = ROBOT_PROFILES["lite6"]
  if not profile.supports_lite6_vacuum_gripper or not profile.lite6_vacuum_gripper_visual_urdf:
    raise ValueError("lite6 does not support vacuum gripper combo URDF")

  src = profile.assets_dir / profile.visual_glb_urdf
  tree = ET.parse(str(src))
  root = tree.getroot()
  _append_vacuum_gripper_visual(root, profile.ee_link)
  root.set("name", Path(profile.lite6_vacuum_gripper_visual_urdf).stem)

  out = profile.assets_dir / profile.lite6_vacuum_gripper_visual_urdf
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def main() -> int:
  out = generate_for_lite6()
  print(f"[lite6] wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
