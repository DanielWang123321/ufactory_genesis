#!/usr/bin/env python3
"""Topology + pipeline diagnostics for Gripper G2 knuckle GLBs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relocalize_arm_glb import (
    _init_genesis,
    bake_glb_genesis_parts,
    export_opaque_doublesided_glb,
    mean_surface_distance,
)
from relocalize_gripper_glb import (
    G2_BLACK_MATERIAL,
    GRIPPER_STL_DIR,
    KNUCKLE_GLBS,
    MOVABLE_SRC,
    STATIC_KNUCKLE_PART_INDICES,
    STL_FOR_GLB,
    VISUAL_GLB_OUT,
    _bounds_volume,
    _gripper_link_poses_in_link6,
    _knuckle_candidates_from_scene,
    _knuckle_from_static_whole,
    _mesh_quality_metrics,
    _movable_from_static_parts,
    _rigid_align_mesh_to_ref,
    _split_assembly_to_link,
    _transform_points_row,
    _volume_ratio_vs_stl,
    extract_movable_groups,
    relocalize_static_assembly,
)

OUT_JSON = Path(__file__).resolve().parents[1] / ".cursor" / "knuckle_pipeline_diag.json"


def _mesh_stats(mesh: trimesh.Trimesh, stl_ref: trimesh.Trimesh | None = None) -> dict:
    out = {
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_mm": ((mesh.bounds * 1000).round(2).tolist()),
        "bounds_volume_mm3": round(_bounds_volume(mesh) * 1e9, 1),
        **_mesh_quality_metrics(mesh),
    }
    if stl_ref is not None:
        out["mean_surface_mm"] = round(mean_surface_distance(mesh, stl_ref) * 1000, 2)
        out["volume_ratio_vs_stl"] = round(_volume_ratio_vs_stl(mesh, stl_ref), 3)
    return out


def discover_static_knuckle_indices(static_parts: list[trimesh.Trimesh]) -> dict[str, list[int]]:
    """Pick large static baked parts (max extent >= 20mm) best matching each knuckle STL."""
    link_poses = _gripper_link_poses_in_link6()
    result: dict[str, list[int]] = {}
    for glb_name in sorted(KNUCKLE_GLBS):
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        t = link_poses[glb_name]
        stl_link6 = stl.copy()
        stl_link6.vertices = _transform_points_row(stl.vertices, t)
        scores: list[tuple[float, int, int, float]] = []
        for idx, part in enumerate(static_parts):
            if len(part.vertices) < 100:
                continue
            ext_mm = float((part.bounds[1] - part.bounds[0]).max() * 1000)
            if ext_mm < 20.0:
                continue
            local = _split_assembly_to_link(part.copy(), t)
            surf = mean_surface_distance(local, stl) * 1000
            scores.append((surf, idx, len(part.vertices), ext_mm))
        if not scores:
            raise RuntimeError(f"No large static part for {glb_name}")
        scores.sort(key=lambda x: x[0])
        best_idx = scores[0][1]
        result[glb_name] = [best_idx]
        print(f"{glb_name}: index={best_idx} surf={scores[0][0]:.2f}mm verts={scores[0][2]}")
    return result


def topology_table(static_parts: list[trimesh.Trimesh]) -> dict:
    link_poses = _gripper_link_poses_in_link6()
    baked = bake_glb_genesis_parts(MOVABLE_SRC)
    scene_knuckles = _knuckle_candidates_from_scene(baked, link_poses)
    static_whole = _knuckle_from_static_whole(static_parts, link_poses)
    static_vertex, _ = _movable_from_static_parts(static_parts, link_poses)
    groups = extract_movable_groups(MOVABLE_SRC)
    table: dict[str, dict] = {}
    for glb_name in sorted(KNUCKLE_GLBS):
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        scene_raw = trimesh.util.concatenate(groups[glb_name][1])
        on_disk = (
            trimesh.load(VISUAL_GLB_OUT / glb_name, force="mesh")
            if (VISUAL_GLB_OUT / glb_name).is_file()
            else None
        )
        row = {
            "static_part_index": STATIC_KNUCKLE_PART_INDICES[glb_name],
            "A_scene_raw": _mesh_stats(scene_raw, stl),
            "B_static_whole_link_local": _mesh_stats(static_whole[glb_name], stl),
            "C_scene_link_local": _mesh_stats(scene_knuckles[glb_name], stl),
            "D_on_disk": _mesh_stats(on_disk, stl) if on_disk else None,
            "E_stl_ref": _mesh_stats(stl),
        }
        table[glb_name] = row
        print(f"\n=== {glb_name} ===")
        for key in ("A_scene_raw", "B_static_whole_link_local", "C_scene_link_local", "D_on_disk"):
            if row.get(key):
                r = row[key]
                print(
                    f"  {key}: verts={r['verts']} vol_ratio={r.get('volume_ratio_vs_stl')} "
                    f"surf={r.get('mean_surface_mm')}mm"
                )
    return table


def pipeline_dump(static_parts: list[trimesh.Trimesh], dump_dir: Path) -> None:
    link_poses = _gripper_link_poses_in_link6()
    baked = bake_glb_genesis_parts(MOVABLE_SRC)
    groups = extract_movable_groups(MOVABLE_SRC)
    scene_knuckles = _knuckle_candidates_from_scene(baked, link_poses)
    static_whole = _knuckle_from_static_whole(static_parts, link_poses)
    static_vertex, _ = _movable_from_static_parts(static_parts, link_poses)
    dump_dir.mkdir(parents=True, exist_ok=True)
    mat = G2_BLACK_MATERIAL

    for glb_name in sorted(KNUCKLE_GLBS):
        sub = dump_dir / glb_name.replace(".glb", "")
        sub.mkdir(parents=True, exist_ok=True)
        scene_raw = trimesh.util.concatenate(groups[glb_name][1])
        aligned, _ = _rigid_align_mesh_to_ref(scene_knuckles[glb_name], static_vertex[glb_name])
        steps = {
            "01_scene_raw": scene_raw,
            "02_scene_link_local": scene_knuckles[glb_name],
            "03_after_procrustes": aligned,
            "04_static_vertex_ref": static_vertex[glb_name],
            "05_static_whole_link_local": static_whole[glb_name],
        }
        for name, mesh in steps.items():
            export_opaque_doublesided_glb([mesh], sub / f"{name}.glb", mat)
        print(f"dumped {sub}/ ({len(steps)} steps)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Knuckle pipeline diagnostics")
    parser.add_argument("--dump", type=Path, default=None, help="Export step GLBs to DIR")
    args = parser.parse_args()

    _init_genesis()
    _, static_parts = relocalize_static_assembly("link6", dry_run=True)
    discovered = discover_static_knuckle_indices(static_parts)
    if discovered != STATIC_KNUCKLE_PART_INDICES:
        print("WARNING: discovered indices differ from STATIC_KNUCKLE_PART_INDICES")
        print("discovered:", discovered)
        print("constants:", STATIC_KNUCKLE_PART_INDICES)
    table = topology_table(static_parts)
    payload = {
        "static_knuckle_part_indices": STATIC_KNUCKLE_PART_INDICES,
        "discovered_indices": discovered,
        "topology": table,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")

    if args.dump:
        pipeline_dump(static_parts, args.dump)


if __name__ == "__main__":
    main()
