#!/usr/bin/env python3
"""Generate arm + static bio gripper GLB visual combo URDFs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import BIO_GRIPPER_ASSETS, PROJECT_ROOT
from ufactory.robot_registry import ROBOT_PROFILES

def _bio_gripper_glb_path(ee_link: str) -> str:
  return f"../bio_gripper/meshes/visual/bio_gripper_g2_visual_{ee_link}.glb"


def _append_bio_gripper_visual(root: ET.Element, ee_link: str) -> None:
  tool = ET.SubElement(root, "link", {"name": "bio_gripper_visual"})
  visual = ET.SubElement(tool, "visual")
  ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
  geom = ET.SubElement(visual, "geometry")
  ET.SubElement(geom, "mesh", {"filename": _bio_gripper_glb_path(ee_link)})
  ET.SubElement(visual, "material", {"name": "White"})

  joint = ET.SubElement(root, "joint", {"name": "bio_gripper_visual_fix", "type": "fixed"})
  ET.SubElement(joint, "parent", {"link": ee_link})
  ET.SubElement(joint, "child", {"link": "bio_gripper_visual"})
  ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})


def generate_for_profile(key: str) -> Path:
  profile = ROBOT_PROFILES[key]
  if not profile.supports_bio_gripper_g2 or not profile.bio_gripper_g2_visual_urdf:
    raise ValueError(f"{key} does not support bio gripper combo URDF")

  src = profile.assets_dir / profile.visual_glb_urdf
  tree = ET.parse(str(src))
  root = tree.getroot()
  _append_bio_gripper_visual(root, profile.ee_link)
  root.set("name", Path(profile.bio_gripper_g2_visual_urdf).stem)

  out = profile.assets_dir / profile.bio_gripper_g2_visual_urdf
  try:
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(out), encoding="utf-8", xml_declaration=True)
  return out


def main() -> int:
  keys = [k for k, p in ROBOT_PROFILES.items() if p.supports_bio_gripper_g2 and p.bio_gripper_g2_visual_urdf]
  for key in keys:
    out = generate_for_profile(key)
    print(f"[{key}] wrote {out}")
  src_glb = BIO_GRIPPER_ASSETS / "meshes" / "visual" / "visual_glb_src" / "bio_gripper_g2.glb"
  if not src_glb.is_file():
    print(f"Warning: missing {src_glb}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
