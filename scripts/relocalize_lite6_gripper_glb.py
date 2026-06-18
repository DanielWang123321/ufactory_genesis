#!/usr/bin/env python3
"""Relocalize Lite6 parallel gripper GLB into per-link visuals + static assembly."""

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
from ufactory.paths import LITE6_GRIPPER_ASSETS

VISUAL_DIR = LITE6_GRIPPER_ASSETS / "meshes" / "visual"
GLB_OUT_DIR = VISUAL_DIR / "visual_glb"
SRC_GLB = VISUAL_DIR / "visual_glb_src" / "lite_gripper.glb"
METRICS_PATH = VISUAL_DIR / "relocalize_metrics.json"
EE_LINK = "link6"
FINGER_JOINT_ORIGIN_Z = 0.0543

CAD_TO_STL_ROT_Z_DEG = 0.0


def _load_finger_stl(name: str) -> trimesh.Trimesh:
    collision = LITE6_GRIPPER_ASSETS / "meshes" / "collision"
    for suffix in (".stl", ".STL"):
        path = collision / f"{name}{suffix}"
        if path.is_file():
            return trimesh.load(path, force="mesh")
    raise FileNotFoundError(f"Missing {name} STL under {collision}")


def _stl_assembly_gripper_link() -> trimesh.Trimesh:
    """Physics collision meshes in uflite_gripper_link frame (matches gripper_fix @ link6 origin)."""
    shell = _stl_shell_path()
    shell_mesh = trimesh.load(shell, force="mesh")
    f1 = _load_finger_stl("finger1")
    f2 = _load_finger_stl("finger2")
    f1 = f1.copy()
    f2 = f2.copy()
    f1.vertices[:, MOUNT_AXIS] += FINGER_JOINT_ORIGIN_Z
    f2.vertices[:, MOUNT_AXIS] += FINGER_JOINT_ORIGIN_Z
    return trimesh.util.concatenate([shell_mesh, f1, f2])


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
    stl_assembly: trimesh.Trimesh,
    stl_shell: trimesh.Trimesh,
) -> tuple[np.ndarray, float]:
    """Shared Rz0 + centroid + mount-face Z flush (ICP worsens Lite6 gripper fit)."""
    rotated = _rotate_cad_to_stl_frame(parts)
    merged = trimesh.util.concatenate([p.copy() for p in rotated])
    centroid_shift = stl_assembly.centroid - merged.centroid
    merged.vertices += centroid_shift
    dz = float(stl_shell.vertices[:, MOUNT_AXIS].min() - merged.vertices[:, MOUNT_AXIS].min())
    return centroid_shift, dz


def _apply_coarse_shifts(
    group_parts: list[trimesh.Trimesh],
    centroid_shift: np.ndarray,
    dz: float,
) -> list[trimesh.Trimesh]:
    if not group_parts:
        return []
    out: list[trimesh.Trimesh] = []
    for part in _rotate_cad_to_stl_frame(group_parts):
        mesh = part.copy()
        mesh.vertices += centroid_shift
        mesh.vertices[:, MOUNT_AXIS] += dz
        out.append(mesh)
    return out


def _align_parts_to_stl(
    parts: list[trimesh.Trimesh],
    stl_ref: trimesh.Trimesh,
    *,
    centroid_shift: np.ndarray | None = None,
    dz: float | None = None,
    stl_shell: trimesh.Trimesh | None = None,
) -> tuple[list[trimesh.Trimesh], dict]:
    """Coarse rigid CAD->STL alignment in gripper link frame (no ICP)."""
    if not parts:
        return [], {"rejected": True}
    if centroid_shift is None or dz is None:
        shell = stl_shell if stl_shell is not None else stl_ref
        centroid_shift, dz = _global_coarse_shifts(parts, stl_ref, shell)
    rotated = _rotate_cad_to_stl_frame(parts)
    merged = trimesh.util.concatenate([p.copy() for p in rotated])
    before_mm = mean_surface_distance(merged, stl_ref) * 1000
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


def _merge_aligned(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _to_finger_link_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Express finger visuals in uflite_finger* link frame (joint origin z=0.0543)."""
    out = mesh.copy()
    out.vertices[:, MOUNT_AXIS] -= FINGER_JOINT_ORIGIN_Z
    return out


def _partition_part_indices(parts: list[trimesh.Trimesh]) -> dict[str, list[int]]:
    if not parts:
        raise RuntimeError("lite_gripper.glb produced no mesh parts")
    counts = [len(p.vertices) for p in parts]
    shell_idx = int(np.argmax(counts))
    finger_indices = [i for i in range(len(parts)) if i != shell_idx]
    if not finger_indices:
        return {"shell": [shell_idx], "finger1": [], "finger2": []}
    finger_ys = [float(parts[i].centroid[1]) for i in finger_indices]
    median_y = float(np.median(finger_ys))
    finger1 = [i for i in finger_indices if float(parts[i].centroid[1]) >= median_y]
    finger2 = [i for i in finger_indices if float(parts[i].centroid[1]) < median_y]
    if not finger1 or not finger2:
        ordered = sorted(finger_indices, key=lambda i: float(parts[i].centroid[1]))
        mid = max(1, len(ordered) // 2)
        finger1 = ordered[mid:]
        finger2 = ordered[:mid]
    return {"shell": [shell_idx], "finger1": finger1, "finger2": finger2}


def _parts_by_indices(parts: list[trimesh.Trimesh], indices: list[int]) -> list[trimesh.Trimesh]:
    return [parts[i] for i in indices]


def _materials_for_indices(indices: list[int], all_pbr: list[dict]) -> list[dict]:
    return [all_pbr[i] for i in indices]


def _stl_shell_path() -> Path:
    collision = LITE6_GRIPPER_ASSETS / "meshes" / "collision"
    for name in ("shell.stl", "shell.STL"):
        path = collision / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing shell STL under {collision}")


def relocalize_lite6_gripper(dry_run: bool = False) -> list[dict]:
    if not SRC_GLB.is_file():
        raise FileNotFoundError(f"Missing source GLB: {SRC_GLB}")
    stl_path = _stl_shell_path()
    stl_shell = trimesh.load(stl_path, force="mesh")
    stl_assembly = _stl_assembly_gripper_link()

    parts = bake_glb_genesis_parts(SRC_GLB)
    all_pbr = _raw_material_pbr(SRC_GLB)
    groups = _partition_part_indices(parts)

    aligned_all, icp_metrics = _align_parts_to_stl(parts, stl_assembly, stl_shell=stl_shell)
    if not aligned_all:
        raise RuntimeError("No parts in lite_gripper.glb")

    centroid_shift, dz = _global_coarse_shifts(parts, stl_assembly, stl_shell)
    aligned_shell_parts = _apply_coarse_shifts(_parts_by_indices(parts, groups["shell"]), centroid_shift, dz)
    aligned_f1_gripper = _apply_coarse_shifts(_parts_by_indices(parts, groups["finger1"]), centroid_shift, dz)
    aligned_f2_gripper = _apply_coarse_shifts(_parts_by_indices(parts, groups["finger2"]), centroid_shift, dz)
    aligned_f1 = [_to_finger_link_frame(m) for m in aligned_f1_gripper]
    aligned_f2 = [_to_finger_link_frame(m) for m in aligned_f2_gripper]

    aligned_static = _merge_aligned(aligned_all)
    aligned_shell = _merge_aligned(aligned_shell_parts)
    if aligned_static is None or aligned_shell is None:
        raise RuntimeError("No shell parts after partitioning lite_gripper.glb")

    assembly_dist_mm = mean_surface_distance(aligned_static, stl_assembly) * 1000

    hole_fit = assembly_dist_mm
    report: list[dict] = []
    export_sets: dict[str, tuple[list[trimesh.Trimesh], list[dict]]] = {
        "shell.glb": (aligned_shell_parts, _materials_for_indices(groups["shell"], all_pbr)),
        "finger1.glb": (aligned_f1, _materials_for_indices(groups["finger1"], all_pbr)),
        "finger2.glb": (aligned_f2, _materials_for_indices(groups["finger2"], all_pbr)),
        f"lite6_gripper_static_{EE_LINK}.glb": (aligned_all, all_pbr),
    }
    for name, (mesh_parts, mesh_pbr) in export_sets.items():
        if not mesh_parts:
            continue
        combined = _merge_aligned(mesh_parts)
        if combined is None:
            continue
        if name == "shell.glb":
            stl_ref = stl_shell
        elif name.startswith("finger"):
            stl_ref = _load_finger_stl(name.split(".")[0])
        else:
            stl_ref = stl_assembly
        entry = {
            "file": name,
            "ee_link": EE_LINK,
            "verts": int(combined.vertices.shape[0]),
            "submesh_count": len(mesh_parts),
            "hole_fit_mm": round(hole_fit, 2) if name.startswith("lite6_gripper_static") else None,
            "assembly_surface_mm": round(assembly_dist_mm, 2) if name.startswith("lite6_gripper_static") else None,
            "icp_metrics": icp_metrics if name.startswith("lite6_gripper_static") else None,
            "stl_surface_mm": round(mean_surface_distance(combined, stl_ref) * 1000, 2),
            "partition_counts": {k: len(v) for k, v in groups.items()},
        }
        report.append(entry)
        print(f"{name}: verts={entry['verts']} submeshes={entry['submesh_count']}")

    if not dry_run:
        GLB_OUT_DIR.mkdir(parents=True, exist_ok=True)
        VISUAL_DIR.mkdir(parents=True, exist_ok=True)
        for name, (mesh_parts, mesh_pbr) in export_sets.items():
            if not mesh_parts:
                continue
            if name.startswith("lite6_gripper_static"):
                out = VISUAL_DIR / name
            else:
                out = GLB_OUT_DIR / name
            export_opaque_doublesided_glb(mesh_parts, out, mesh_pbr)
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {METRICS_PATH}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Relocalize Lite6 gripper GLB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _init_genesis()
    relocalize_lite6_gripper(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
