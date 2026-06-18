#!/usr/bin/env python3
"""Validate Lite6 gripper/vacuum combo URDF mesh paths (no Genesis required)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ufactory.paths import (
    LITE6_GRIPPER_ASSETS,
    LITE6_VACUUM_GRIPPER_ASSETS,
    lite6_with_gripper_urdf,
    lite6_with_vacuum_gripper_urdf,
    robot_visual_glb_urdf,
)
from ufactory.robot_registry import ROBOT_PROFILES


def _mesh_paths(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [m.get("filename", "") for m in root.iter("mesh") if m.get("filename")]


def _resolve(urdf_path: Path, rel: str) -> Path:
    return (urdf_path.parent / rel).resolve()


def main() -> int:
    errors: list[str] = []

    for key in ("xarm6_1305", "uf850"):
        try:
            robot_visual_glb_urdf(key, with_lite6_gripper=True)
            errors.append(f"{key} should reject with_lite6_gripper=True")
        except ValueError:
            print(f"[ok] {key} rejects Lite6 gripper")

    for movable in (False, True):
        urdf = Path(robot_visual_glb_urdf("lite6", with_lite6_gripper=True, movable=movable))
        label = f"lite6 gripper movable={movable}"
        if not urdf.is_file():
            errors.append(f"{label}: missing {urdf}")
            continue
        missing = [rel for rel in _mesh_paths(urdf) if not _resolve(urdf, rel).is_file()]
        if missing:
            errors.append(f"{label}: missing meshes: {missing[:3]}")
        else:
            print(f"[ok] {label}: {urdf.name}")

    vac_urdf = Path(robot_visual_glb_urdf("lite6", with_lite6_vacuum_gripper=True))
    missing_vac = [rel for rel in _mesh_paths(vac_urdf) if not _resolve(vac_urdf, rel).is_file()]
    if missing_vac:
        errors.append(f"lite6 vacuum: missing meshes: {missing_vac}")
    else:
        print(f"[ok] lite6 vacuum: {vac_urdf.name}")

    for label, resolver in (
        ("lite6_with_gripper", lite6_with_gripper_urdf),
        ("lite6_with_vacuum_gripper", lite6_with_vacuum_gripper_urdf),
    ):
        urdf = Path(resolver())
        missing = [rel for rel in _mesh_paths(urdf) if not _resolve(urdf, rel).is_file()]
        if missing:
            errors.append(f"{label}: missing meshes: {missing[:3]}")
        else:
            print(f"[ok] {label}: {urdf.name}")

    src_gripper = LITE6_GRIPPER_ASSETS / "meshes" / "visual" / "visual_glb_src" / "lite_gripper.glb"
    src_vacuum = LITE6_VACUUM_GRIPPER_ASSETS / "meshes" / "visual" / "visual_glb_src" / "lite_vacuum_gripper.glb"
    for path in (src_gripper, src_vacuum):
        if not path.is_file():
            errors.append(f"missing source GLB: {path}")
        else:
            print(f"[ok] source GLB: {path.name}")

    profile = ROBOT_PROFILES["lite6"]
    if not profile.supports_lite6_gripper or not profile.supports_lite6_vacuum_gripper:
        errors.append("lite6 profile missing accessory support flags")

    if errors:
        for err in errors:
            print(f"[fail] {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
