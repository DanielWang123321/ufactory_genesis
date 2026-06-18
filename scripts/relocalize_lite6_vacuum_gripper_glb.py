#!/usr/bin/env python3
"""Relocalize Lite6 vacuum gripper GLB for static visual combo URDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from relocalize_arm_glb import (
    _init_genesis,
    _raw_material_pbr,
    bake_glb_genesis_parts,
    export_opaque_doublesided_glb,
    mean_surface_distance,
)
from relocalize_gripper_glb import MOUNT_AXIS
from ufactory.paths import LITE6_VACUUM_GRIPPER_ASSETS

VISUAL_DIR = LITE6_VACUUM_GRIPPER_ASSETS / "meshes" / "visual"
SRC_GLB = VISUAL_DIR / "visual_glb_src" / "lite_vacuum_gripper.glb"
METRICS_PATH = VISUAL_DIR / "relocalize_metrics.json"
EE_LINK = "link6"

CAD_TO_STL_ROT_Z_DEG = 0.0


def _stl_base_path() -> Path:
    collision = LITE6_VACUUM_GRIPPER_ASSETS / "meshes" / "collision"
    for name in ("vacuum_gripper_lite.stl", "vacuum_gripper_lite.STL"):
        path = collision / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing vacuum gripper STL under {collision}")


def _rotate_cad_to_stl_frame(parts: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    Rz = Rotation.from_euler("z", CAD_TO_STL_ROT_Z_DEG, degrees=True).as_matrix()
    out: list[trimesh.Trimesh] = []
    for part in parts:
        mesh = part.copy()
        mesh.vertices = mesh.vertices @ Rz.T
        out.append(mesh)
    return out


def _global_coarse_shifts(
    parts: list[trimesh.Trimesh],
    stl_ref: trimesh.Trimesh,
) -> tuple[np.ndarray, float]:
    """Rz + centroid + mount-face Z flush in uflite_vacuum_gripper_link frame."""
    rotated = _rotate_cad_to_stl_frame(parts)
    merged = trimesh.util.concatenate([p.copy() for p in rotated])
    centroid_shift = stl_ref.centroid - merged.centroid
    merged.vertices += centroid_shift
    dz = float(stl_ref.vertices[:, MOUNT_AXIS].min() - merged.vertices[:, MOUNT_AXIS].min())
    return centroid_shift, dz


def _apply_coarse_shifts(
    parts: list[trimesh.Trimesh],
    centroid_shift: np.ndarray,
    dz: float,
) -> list[trimesh.Trimesh]:
    out: list[trimesh.Trimesh] = []
    for part in _rotate_cad_to_stl_frame(parts):
        mesh = part.copy()
        mesh.vertices += centroid_shift
        mesh.vertices[:, MOUNT_AXIS] += dz
        out.append(mesh)
    return out


def _align_parts_to_stl(
    parts: list[trimesh.Trimesh],
    stl_ref: trimesh.Trimesh,
) -> tuple[list[trimesh.Trimesh], dict]:
    if not parts:
        return [], {"rejected": True}
    rotated = _rotate_cad_to_stl_frame(parts)
    merged = trimesh.util.concatenate([p.copy() for p in rotated])
    before_mm = mean_surface_distance(merged, stl_ref) * 1000
    centroid_shift, dz = _global_coarse_shifts(parts, stl_ref)
    out = _apply_coarse_shifts(parts, centroid_shift, dz)
    merged_after = trimesh.util.concatenate([p.copy() for p in out])
    after_mm = mean_surface_distance(merged_after, stl_ref) * 1000
    ratios = merged_after.extents / np.maximum(stl_ref.extents, 1e-9)
    return out, {
        "alignment": "stl_coarse",
        "rejected": False,
        "surf_before_mm": round(before_mm, 2),
        "surf_after_mm": round(after_mm, 2),
        "extent_ratio_min": round(float(ratios.min()), 3),
        "extent_ratio_max": round(float(ratios.max()), 3),
    }


def relocalize_lite6_vacuum_gripper(dry_run: bool = False) -> dict:
    if not SRC_GLB.is_file():
        raise FileNotFoundError(f"Missing source GLB: {SRC_GLB}")
    stl_ref = trimesh.load(_stl_base_path(), force="mesh")

    parts = bake_glb_genesis_parts(SRC_GLB)
    all_pbr = _raw_material_pbr(SRC_GLB)
    aligned_parts, align_metrics = _align_parts_to_stl(parts, stl_ref)
    if not aligned_parts:
        raise RuntimeError("No parts in lite_vacuum_gripper.glb")

    combined = trimesh.util.concatenate([p.copy() for p in aligned_parts])
    stl_surface_mm = mean_surface_distance(combined, stl_ref) * 1000
    out_name = f"lite6_vacuum_gripper_visual_{EE_LINK}.glb"
    entry = {
        "ee_link": EE_LINK,
        "output": out_name,
        "verts": int(combined.vertices.shape[0]),
        "submesh_count": len(aligned_parts),
        "stl_surface_mm": round(stl_surface_mm, 2),
        "align_metrics": align_metrics,
    }
    print(
        f"{out_name}: verts={entry['verts']} submeshes={entry['submesh_count']}, "
        f"stl_surface={entry['stl_surface_mm']:.2f}mm"
    )
    if not dry_run:
        VISUAL_DIR.mkdir(parents=True, exist_ok=True)
        out = VISUAL_DIR / out_name
        export_opaque_doublesided_glb(aligned_parts, out, all_pbr)
        METRICS_PATH.write_text(json.dumps([entry], indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {METRICS_PATH}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Relocalize Lite6 vacuum gripper GLB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _init_genesis()
    relocalize_lite6_vacuum_gripper(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
