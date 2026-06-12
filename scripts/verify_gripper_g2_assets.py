#!/usr/bin/env python3
"""Validate Gripper G2 combo URDF mesh paths (no Genesis required)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import GRIPPER_G2_ASSETS, robot_visual_glb_urdf
from ufactory.robot_registry import ROBOT_PROFILES


def _mesh_paths(urdf_path: Path) -> list[str]:
  root = ET.parse(urdf_path).getroot()
  return [m.get("filename", "") for m in root.iter("mesh") if m.get("filename")]


def _resolve(urdf_path: Path, rel: str) -> Path:
  return (urdf_path.parent / rel).resolve()


def main() -> int:
  errors: list[str] = []

  try:
    robot_visual_glb_urdf("lite6", with_gripper_g2=True)
    errors.append("lite6 should reject with_gripper_g2=True")
  except ValueError:
    print("[ok] lite6 rejects G2")

  keys = [k for k, p in ROBOT_PROFILES.items() if p.supports_gripper_g2]
  for key in keys:
    for movable in (False, True):
      urdf = Path(robot_visual_glb_urdf(key, with_gripper_g2=True, movable=movable))
      label = f"{key} movable={movable}"
      if not urdf.is_file():
        errors.append(f"{label}: missing {urdf}")
        continue
      missing = []
      for rel in _mesh_paths(urdf):
        if not _resolve(urdf, rel).is_file():
          missing.append(rel)
      if missing:
        errors.append(f"{label}: missing meshes: {missing[:3]}{'...' if len(missing) > 3 else ''}")
      else:
        print(f"[ok] {label}: {urdf.name}")

  src = GRIPPER_G2_ASSETS / "meshes" / "visual" / "visual_glb_src" / "gripper_g2.glb"
  if not src.is_file():
    errors.append(f"missing source GLB: {src}")
  else:
    print(f"[ok] source GLB: {src.name}")

  if errors:
    for err in errors:
      print(f"[fail] {err}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
