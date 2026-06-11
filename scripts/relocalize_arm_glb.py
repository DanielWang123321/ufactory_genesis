#!/usr/bin/env python3
"""
Relocalize xArm6 1305 GLB visuals into URDF link-local frames.

Source GLBs are exported from CAD in a global assembly frame with internal scene-graph
offsets (decimetres). URDF expects per-link geometry in each link's local frame (like STL).

Uses Genesis glTF parser (same as runtime) to bake node transforms, scales ×0.1 to metres,
centroid-aligns then refines translation against the STL reference (UFACTORY DH link frame),
and writes flat GLBs with opaque double-sided materials.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pygltflib
import trimesh
from pygltflib import (
    Accessor,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Scene,
)
from trimesh.proximity import closest_point

from ufactory.paths import XARM6_ASSETS

CAD_TO_METRES = 0.1
DEFAULT_RGBA = (1.0, 1.0, 1.0, 1.0)
LINKS = (
    ("link_base.glb", "link_base.stl"),
    ("link1.glb", "link1.stl"),
    ("link2.glb", "link2.stl"),
    ("link3.glb", "link3.stl"),
    ("link4.glb", "link4.stl"),
    ("link5.glb", "link5.stl"),
    ("link6.glb", "link6.stl"),
)


def _init_genesis() -> None:
    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="error")


def _raw_material_pbr(glb_path: Path) -> list[dict]:
    glb = pygltflib.GLTF2().load(str(glb_path))
    materials: list[dict] = []
    for mat in glb.materials:
        pbr = mat.pbrMetallicRoughness
        if pbr is not None and pbr.baseColorFactor is not None:
            rgba = [float(x) for x in pbr.baseColorFactor[:4]]
        else:
            rgba = list(DEFAULT_RGBA)
        metallic = float(pbr.metallicFactor) if pbr is not None and pbr.metallicFactor is not None else 0.0
        roughness = float(pbr.roughnessFactor) if pbr is not None and pbr.roughnessFactor is not None else 0.5
        materials.append({"rgba": rgba, "metallic": metallic, "roughness": roughness})
    return materials or [{"rgba": list(DEFAULT_RGBA), "metallic": 0.0, "roughness": 0.5}]


def _raw_material_colors(glb_path: Path) -> list[list[float]]:
    return [m["rgba"] for m in _raw_material_pbr(glb_path)]


def bake_glb_genesis_parts(glb_path: Path, scale: float = CAD_TO_METRES) -> list[trimesh.Trimesh]:
    """Bake GLB scene graph using Genesis parser (matches runtime loading)."""
    from genesis.options import surfaces
    from genesis.utils import gltf as gltf_utils

    surface = surfaces.Default()
    meshes = gltf_utils.parse_mesh_glb(
        str(glb_path),
        group_by_material=False,
        scale=None,
        is_mesh_zup=True,
        surface=surface,
    )
    if not meshes:
        raise RuntimeError(f"No geometry in {glb_path}")
    parts = [m.trimesh.copy() for m in meshes]
    for part in parts:
        part.apply_scale(scale)
    return parts


def mean_surface_distance(source: trimesh.Trimesh, target: trimesh.Trimesh, samples: int = 1000) -> float:
    return float(np.mean(closest_point(target, source.sample(samples))[1]))


def refine_translation_to_stl(
    parts: list[trimesh.Trimesh],
    stl_ref: trimesh.Trimesh,
    initial_delta: np.ndarray,
    search_mm: float = 5.0,
    samples: int = 1500,
) -> np.ndarray:
    """Refine link-local translation so GLB surface matches DH-frame STL (not centroid only)."""
    from scipy.optimize import minimize

    combined = trimesh.util.concatenate([part.copy() for part in parts])
    combined.vertices += initial_delta
    base = combined.vertices.copy()
    bound = search_mm / 1000.0

    def cost(offset: np.ndarray) -> float:
        combined.vertices = base + offset
        return mean_surface_distance(combined, stl_ref, samples=samples)

    result = minimize(
        cost,
        np.zeros(3, dtype=np.float64),
        method="Powell",
        options={"xtol": 1e-5, "ftol": 1e-5, "maxiter": 200},
    )
    offset = np.asarray(result.x, dtype=np.float64)
    offset = np.clip(offset, -bound, bound)
    return initial_delta + offset


def align_parts_to_stl(parts: list[trimesh.Trimesh], stl_ref: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    combined = trimesh.util.concatenate(parts)
    centroid_delta = stl_ref.centroid - combined.centroid
    delta = refine_translation_to_stl(parts, stl_ref, centroid_delta)
    aligned = []
    for part in parts:
        out = part.copy()
        out.vertices += delta
        aligned.append(out)
    return aligned


def export_opaque_doublesided_glb(
    parts: list[trimesh.Trimesh],
    out_path: Path,
    pbr_materials: list[dict] | None = None,
) -> None:
    """Write GLB with opaque double-sided PBR materials (prevents backface holes)."""
    if pbr_materials is None:
        pbr_materials = [{"rgba": list(DEFAULT_RGBA), "metallic": 0.0, "roughness": 0.5}]

    blob = bytearray()
    buffer_views: list[BufferView] = []
    accessors: list[Accessor] = []
    materials: list[Material] = []
    meshes: list[Mesh] = []
    nodes: list[Node] = []

    for idx, mesh in enumerate(parts):
        pbr = pbr_materials[idx % len(pbr_materials)]
        rgba = list(pbr["rgba"])
        if len(rgba) == 3:
            rgba = rgba + [1.0]
        metallic = float(pbr.get("metallic", 0.0))
        roughness = float(pbr.get("roughness", 0.5))

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
        norms = np.asarray(mesh.vertex_normals, dtype=np.float32)

        v_bytes = verts.tobytes()
        n_bytes = norms.tobytes()
        i_bytes = faces.tobytes()
        v_off = len(blob)
        blob.extend(v_bytes)
        n_off = len(blob)
        blob.extend(n_bytes)
        i_off = len(blob)
        blob.extend(i_bytes)

        bv_base = len(buffer_views)
        buffer_views.extend(
            [
                BufferView(buffer=0, byteOffset=v_off, byteLength=len(v_bytes), target=34962),
                BufferView(buffer=0, byteOffset=n_off, byteLength=len(n_bytes), target=34962),
                BufferView(buffer=0, byteOffset=i_off, byteLength=len(i_bytes), target=34963),
            ]
        )
        acc_base = len(accessors)
        accessors.extend(
            [
                Accessor(
                    bufferView=bv_base,
                    byteOffset=0,
                    componentType=5126,
                    count=len(verts),
                    type="VEC3",
                    max=verts.max(axis=0).tolist(),
                    min=verts.min(axis=0).tolist(),
                ),
                Accessor(
                    bufferView=bv_base + 1,
                    byteOffset=0,
                    componentType=5126,
                    count=len(norms),
                    type="VEC3",
                ),
                Accessor(
                    bufferView=bv_base + 2,
                    byteOffset=0,
                    componentType=5125,
                    count=len(faces),
                    type="SCALAR",
                ),
            ]
        )
        materials.append(
            Material(
                doubleSided=True,
                alphaMode="OPAQUE",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=rgba,
                    metallicFactor=metallic,
                    roughnessFactor=roughness,
                ),
            )
        )
        meshes.append(
            Mesh(
                name=f"part_{idx}",
                primitives=[
                    Primitive(
                        attributes={"POSITION": acc_base, "NORMAL": acc_base + 1},
                        indices=acc_base + 2,
                        material=idx,
                        mode=4,
                    )
                ],
            )
        )
        nodes.append(Node(mesh=idx, name=f"part_{idx}"))

    glb = GLTF2(
        asset=Asset(version="2.0"),
        scene=0,
        scenes=[Scene(nodes=list(range(len(nodes))))],
        nodes=nodes,
        meshes=meshes,
        materials=materials,
        buffers=[Buffer(byteLength=len(blob))],
        bufferViews=buffer_views,
        accessors=accessors,
    )
    glb.set_binary_blob(bytes(blob))
    glb.save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Relocalize arm GLB meshes to link frames")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=XARM6_ASSETS / "meshes" / "xarm6_1305" / "visual_glb_raw",
        help="Source GLBs (CAD export). Defaults to visual_glb_raw backup.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=XARM6_ASSETS / "meshes" / "xarm6_1305" / "visual_glb_raw",
        help="Backup original GLBs before overwriting visual_glb/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report alignment metrics without writing files",
    )
    args = parser.parse_args()

    visual_glb = XARM6_ASSETS / "meshes" / "xarm6_1305" / "visual_glb"
    visual_stl = XARM6_ASSETS / "meshes" / "xarm6_1305" / "visual"

    if not args.dry_run:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        for glb_name, _ in LINKS:
            src = visual_glb / glb_name
            raw = args.src_dir / glb_name
            if raw.exists() and not (args.backup_dir / glb_name).exists():
                shutil.copy2(raw, args.backup_dir / glb_name)
            elif src.exists() and not (args.backup_dir / glb_name).exists():
                shutil.copy2(src, args.backup_dir / glb_name)

    report = []
    _init_genesis()
    for glb_name, stl_name in LINKS:
        src_path = args.src_dir / glb_name
        if not src_path.exists():
            src_path = visual_glb / glb_name
        stl_path = visual_stl / stl_name

        parts = bake_glb_genesis_parts(src_path)
        stl_ref = trimesh.load(stl_path, force="mesh")
        aligned_parts = align_parts_to_stl(parts, stl_ref)
        combined = trimesh.util.concatenate(aligned_parts)
        pbr_materials = _raw_material_pbr(src_path)
        surf_mm = mean_surface_distance(combined, stl_ref) * 1000
        centroid_mm = float(np.linalg.norm(combined.centroid - stl_ref.centroid) * 1000)
        entry = {
            "file": glb_name,
            "parts": len(aligned_parts),
            "materials_double_sided": True,
            "verts": len(combined.vertices),
            "mean_surface_mm": round(surf_mm, 2),
            "centroid_err_mm": round(centroid_mm, 2),
        }
        report.append(entry)
        print(
            f"{glb_name}: parts={entry['parts']}, mean_surface={entry['mean_surface_mm']:.1f}mm, "
            f"centroid_err={entry['centroid_err_mm']:.1f}mm"
        )
        if not args.dry_run:
            export_opaque_doublesided_glb(aligned_parts, visual_glb / glb_name, pbr_materials)

    if not args.dry_run:
        metrics_path = visual_glb / "relocalize_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote relocalized GLBs to {visual_glb}")
        print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
