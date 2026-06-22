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

from ufactory.robot_registry import (
    ROBOT_PROFILES,
    RobotModelSpec,
    get_profile_key_for_robot_name,
    get_robot_profile,
    glb_output_name,
    link_glb_stl_pairs,
    robot_cli_choices,
)
from ufactory.paths import XARM6_ASSETS

CAD_TO_METRES = 0.1
DEFAULT_RGBA = (1.0, 1.0, 1.0, 1.0)
DEFAULT_ROBOT_KEY = "xarm6"


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


def mean_normal_dot(source: trimesh.Trimesh, target: trimesh.Trimesh, samples: int = 2000) -> float:
    """Average normal agreement; low values indicate flipped / symmetric mis-registration."""
    pts = source.sample(samples)
    _, _, tri_id = closest_point(target, pts)
    tri = target.triangles[tri_id]
    target_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)
    _, vert_id = source.kdtree.query(pts)
    source_normals = source.vertex_normals[vert_id]
    return float(np.mean(np.sum(source_normals * target_normals, axis=1)))


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


RIGID_ALIGN_THRESHOLD_MM = 2.5
NORMAL_DOT_THRESHOLD = 0.35
ORIENTATION_SURF_MAX_MM = 3.0
# High normal_dot but mm-level surface gap (UF850 link5: ~3mm, ndot~0.9).
SURFACE_DISAMBIG_MIN_MM = 1.0
SURFACE_DISAMBIG_MAX_MM = 4.5
EXTENT_RATIO_MIN = 0.85
EXTENT_RATIO_MAX = 1.15


def _apply_transform(parts: list[trimesh.Trimesh], matrix: np.ndarray) -> list[trimesh.Trimesh]:
    aligned = []
    for part in parts:
        out = part.copy()
        out.apply_transform(matrix)
        aligned.append(out)
    return aligned


def _apply_translation(parts: list[trimesh.Trimesh], delta: np.ndarray) -> list[trimesh.Trimesh]:
    aligned = []
    for part in parts:
        out = part.copy()
        out.vertices += delta
        aligned.append(out)
    return aligned


def _extent_ratios(mesh: trimesh.Trimesh, stl_ref: trimesh.Trimesh) -> np.ndarray:
    denom = np.maximum(stl_ref.extents, 1e-9)
    return mesh.extents / denom


def _is_rigid_rotation(matrix: np.ndarray, tol: float = 0.02) -> bool:
    singular = np.linalg.svd(matrix[:3, :3])[1]
    return bool(np.allclose(singular, 1.0, atol=tol) and np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=tol))


def refine_rigid_to_stl(
    parts: list[trimesh.Trimesh],
    stl_ref: trimesh.Trimesh,
    samples: int = 500,
) -> tuple[list[trimesh.Trimesh], dict]:
    """Rigid rotation+translation via mesh_other (never scale)."""
    combined = trimesh.util.concatenate([part.copy() for part in parts])
    matrix, _cost = trimesh.registration.mesh_other(
        combined,
        stl_ref,
        samples=samples,
        scale=False,
        icp_first=10,
        icp_final=50,
    )
    if np.linalg.det(matrix[:3, :3]) < 0:
        matrix = matrix @ np.diag([-1.0, 1.0, 1.0, 1.0])
    aligned = _apply_transform(parts, matrix)
    combined = trimesh.util.concatenate(aligned)
    ratios = _extent_ratios(combined, stl_ref)
    meta = {
        "rigid": _is_rigid_rotation(matrix),
        "extent_ratio_min": float(ratios.min()),
        "extent_ratio_max": float(ratios.max()),
    }
    if meta["extent_ratio_min"] < EXTENT_RATIO_MIN or meta["extent_ratio_max"] > EXTENT_RATIO_MAX:
        return parts, {**meta, "rejected": True}
    return aligned, {**meta, "rejected": False}


def _orientation_score(parts: list[trimesh.Trimesh], stl_ref: trimesh.Trimesh) -> tuple[float, float]:
    combined = trimesh.util.concatenate(parts)
    return mean_normal_dot(combined, stl_ref), mean_surface_distance(combined, stl_ref) * 1000


def disambiguate_orientation(parts: list[trimesh.Trimesh], stl_ref: trimesh.Trimesh) -> tuple[list[trimesh.Trimesh], dict]:
    """Resolve near-symmetric links where surface distance alone picks the wrong flip."""
    best_parts = parts
    best_dot, best_surf = _orientation_score(parts, stl_ref)
    meta = {"disambiguated": False, "normal_dot_before": round(best_dot, 3)}

    candidates: list[list[trimesh.Trimesh]] = []
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        matrix = trimesh.transformations.rotation_matrix(np.pi, axis)
        candidates.append(_apply_transform(parts, matrix))

    rigid_parts, rigid_meta = refine_rigid_to_stl(parts, stl_ref)
    if not rigid_meta.get("rejected"):
        candidates.append(rigid_parts)

    for cand in candidates:
        combined = trimesh.util.concatenate(cand)
        delta = refine_translation_to_stl(cand, stl_ref, stl_ref.centroid - combined.centroid)
        cand = _apply_translation(cand, delta)
        dot, surf = _orientation_score(cand, stl_ref)
        if surf <= ORIENTATION_SURF_MAX_MM and dot > best_dot:
            best_parts, best_dot, best_surf = cand, dot, surf
            meta["disambiguated"] = True

    meta["normal_dot_after"] = round(best_dot, 3)
    meta["orientation_surf_mm"] = round(best_surf, 2)
    return best_parts, meta


def align_parts_to_stl(parts: list[trimesh.Trimesh], stl_ref: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    combined = trimesh.util.concatenate(parts)
    centroid_delta = stl_ref.centroid - combined.centroid
    delta = refine_translation_to_stl(parts, stl_ref, centroid_delta)
    aligned = _apply_translation(parts, delta)

    combined = trimesh.util.concatenate(aligned)
    surf_mm = mean_surface_distance(combined, stl_ref) * 1000
    normal_dot = mean_normal_dot(combined, stl_ref)
    orient_meta: dict = {"normal_dot": round(normal_dot, 3)}
    stuck_translation = SURFACE_DISAMBIG_MIN_MM < surf_mm <= SURFACE_DISAMBIG_MAX_MM
    if normal_dot < NORMAL_DOT_THRESHOLD or stuck_translation:
        aligned, orient_meta = disambiguate_orientation(aligned, stl_ref)
        combined = trimesh.util.concatenate(aligned)
        surf_mm = mean_surface_distance(combined, stl_ref) * 1000
        normal_dot = mean_normal_dot(combined, stl_ref)

    rigid_meta: dict = {"used_rigid": False}
    if surf_mm > RIGID_ALIGN_THRESHOLD_MM:
        rigid_aligned, rigid_meta = refine_rigid_to_stl(aligned, stl_ref)
        if not rigid_meta.get("rejected"):
            aligned = rigid_aligned
            rigid_meta["used_rigid"] = True
            combined = trimesh.util.concatenate(aligned)
            delta = refine_translation_to_stl(
                aligned, stl_ref, stl_ref.centroid - combined.centroid, search_mm=20.0
            )
            aligned = _apply_translation(aligned, delta)

    # #region agent log
    combined = trimesh.util.concatenate(aligned)
    _surf_after = mean_surface_distance(combined, stl_ref) * 1000
    _ratios = _extent_ratios(combined, stl_ref)
    try:
        import time as _time

        _log_path = Path(__file__).resolve().parents[1] / ".cursor" / "debug-abca17.log"
        _payload = {
            "sessionId": "abca17",
            "runId": "relocalize",
            "hypothesisId": "C",
            "location": "relocalize_arm_glb:align_parts_to_stl",
            "message": "alignment result",
            "data": {
                "surf_before_rigid_mm": round(surf_mm, 2),
                "surf_after_mm": round(_surf_after, 2),
                "normal_dot_after": round(mean_normal_dot(combined, stl_ref), 3),
                "extent_ratio_min": round(float(_ratios.min()), 3),
                "extent_ratio_max": round(float(_ratios.max()), 3),
                **orient_meta,
                **rigid_meta,
            },
            "timestamp": int(_time.time() * 1000),
        }
        with _log_path.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(_payload) + "\n")
    except OSError:
        pass
    # #endregion

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


def _mesh_dirs(profile: RobotModelSpec) -> tuple[Path, Path, Path, Path]:
    mesh_root = profile.assets_dir / "meshes" / profile.mesh_variant
    return (
        mesh_root / "visual_glb_src",
        mesh_root / "visual_glb",
        mesh_root / "visual",
        mesh_root / "visual_glb_raw",
    )


def relocalize_robot(profile: RobotModelSpec, src_dir: Path | None, backup_dir: Path | None, dry_run: bool) -> None:
    src_default, visual_glb, visual_stl, backup_default = _mesh_dirs(profile)
    src_dir = src_dir or src_default
    backup_dir = backup_dir or backup_default
    links = link_glb_stl_pairs(profile)

    if not dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        visual_glb.mkdir(parents=True, exist_ok=True)
        for src_glb, _ in links:
            out_glb = glb_output_name(profile, src_glb)
            src = visual_glb / out_glb
            raw = src_dir / src_glb
            if raw.exists() and not (backup_dir / out_glb).exists():
                shutil.copy2(raw, backup_dir / out_glb)
            elif src.exists() and not (backup_dir / out_glb).exists():
                shutil.copy2(src, backup_dir / out_glb)

    report = []
    _init_genesis()
    for src_glb, stl_name in links:
        out_glb = glb_output_name(profile, src_glb)
        src_path = src_dir / src_glb
        if not src_path.exists():
            src_path = visual_glb / out_glb
        stl_path = visual_stl / stl_name
        if not stl_path.exists():
            raise FileNotFoundError(f"Missing STL reference: {stl_path}")

        parts = bake_glb_genesis_parts(src_path)
        stl_ref = trimesh.load(stl_path, force="mesh")
        aligned_parts = align_parts_to_stl(parts, stl_ref)
        combined = trimesh.util.concatenate(aligned_parts)
        pbr_materials = _raw_material_pbr(src_path)
        surf_mm = mean_surface_distance(combined, stl_ref) * 1000
        centroid_mm = float(np.linalg.norm(combined.centroid - stl_ref.centroid) * 1000)
        ratios = _extent_ratios(combined, stl_ref)
        entry = {
            "src_file": src_glb,
            "file": out_glb,
            "parts": len(aligned_parts),
            "materials_double_sided": True,
            "verts": len(combined.vertices),
            "mean_surface_mm": round(surf_mm, 2),
            "centroid_err_mm": round(centroid_mm, 2),
            "normal_dot": round(mean_normal_dot(combined, stl_ref), 3),
            "extent_ratio_min": round(float(ratios.min()), 3),
            "extent_ratio_max": round(float(ratios.max()), 3),
        }
        report.append(entry)
        print(
            f"{out_glb}: parts={entry['parts']}, mean_surface={entry['mean_surface_mm']:.1f}mm, "
            f"centroid_err={entry['centroid_err_mm']:.1f}mm"
        )
        if not dry_run:
            export_opaque_doublesided_glb(aligned_parts, visual_glb / out_glb, pbr_materials)

    if not dry_run:
        metrics_path = visual_glb / "relocalize_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote relocalized GLBs to {visual_glb}")
        print(f"Metrics: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relocalize arm GLB meshes to link frames")
    parser.add_argument(
        "--robot",
        default=DEFAULT_ROBOT_KEY,
        choices=robot_cli_choices(),
        help="Robot profile key or short name (default: xarm6 -> xarm6_1305)",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Source GLBs (CAD export). Defaults to meshes/<variant>/visual_glb_src/",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup original GLBs before overwriting visual_glb/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report alignment metrics without writing files",
    )
    args = parser.parse_args()

    profile = get_robot_profile(args.robot)
    if get_profile_key_for_robot_name(args.robot) == "xarm6_1305" and args.src_dir is None:
        legacy_raw = XARM6_ASSETS / "meshes" / "xarm6_1305" / "visual_glb_raw"
        if legacy_raw.is_dir() and any(legacy_raw.glob("*.glb")):
            args.src_dir = legacy_raw
    relocalize_robot(profile, args.src_dir, args.backup_dir, args.dry_run)


if __name__ == "__main__":
    main()
