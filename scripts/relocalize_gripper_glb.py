#!/usr/bin/env python3
"""
Relocalize Gripper G2 GLB visuals into URDF link-local frames.

- gripper_g2.glb: semantic parts (case/splint/support/tie) -> 7 per-link GLBs in visual_glb/
- gripper_g2_movable.glb: high-res CAD assembly -> single gripper_g2_static.glb for fixed preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relocalize_arm_glb import (  # noqa: E402
    CAD_TO_METRES,
    _init_genesis,
    _raw_material_pbr,
    align_parts_to_stl,
    bake_glb_genesis_parts,
    export_opaque_doublesided_glb,
    mean_surface_distance,
    refine_rigid_to_stl,
)

from ufactory.paths import GRIPPER_G2_ASSETS, robot_visual_glb_urdf  # noqa: E402
from ufactory.robot_registry import ROBOT_PROFILES  # noqa: E402

GLB_TO_LINK: dict[str, str] = {
    "base.glb": "link6",  # xarm_gripper_base_link is merged into link6 in Genesis
    "left_outer_knuckle.glb": "left_outer_knuckle",
    "left_finger.glb": "left_finger",
    "left_inner_knuckle.glb": "left_inner_knuckle",
    "right_outer_knuckle.glb": "right_outer_knuckle",
    "right_finger.glb": "right_finger",
    "right_inner_knuckle.glb": "right_inner_knuckle",
}

GRIPPER_VISUAL_DIR = GRIPPER_G2_ASSETS / "meshes" / "visual"
GRIPPER_STL_DIR = GRIPPER_G2_ASSETS / "meshes" / "collision"
VISUAL_GLB_OUT = GRIPPER_VISUAL_DIR / "visual_glb"
SRC_DIR = GRIPPER_VISUAL_DIR / "visual_glb_src"

MOVABLE_SRC = SRC_DIR / "gripper_g2.glb"
STATIC_SRC = SRC_DIR / "gripper_g2_movable.glb"

EE_RING_R = (0.016, 0.032)
GRIPPER_G2_DIR = GRIPPER_VISUAL_DIR  # legacy alias for diagnose scripts

SHARED_MOVABLE_GLBS = tuple(
  name for name in (
    "left_outer_knuckle.glb",
    "left_finger.glb",
    "left_inner_knuckle.glb",
    "right_outer_knuckle.glb",
    "right_finger.glb",
    "right_inner_knuckle.glb",
  )
)

# gripper_g2.glb scene-graph node predicates -> output GLB (aligned to matching STL)
MOVABLE_GROUP_PREDICATES: dict[str, list] = {
    "base.glb": [lambda n: n.startswith("case")],
    "left_outer_knuckle.glb": [lambda n: n in ("support_l", "support_dot_l")],
    "left_finger.glb": [lambda n: n.startswith("splint_l") or n == "splint_dot_l"],
    "left_inner_knuckle.glb": [lambda n: n in ("tie_l", "tie_dot_l")],
    "right_outer_knuckle.glb": [lambda n: n in ("support_r", "support_dot_r")],
    "right_finger.glb": [lambda n: n.startswith("splint_r") or n == "splint_dot_r"],
    "right_inner_knuckle.glb": [lambda n: n in ("tie_r", "tie_dot_r")],
}
SKIP_NODES = frozenset({"world", "Gripper G2"})
# gripper_g2.glb Genesis bake order: case, dot, splint_l, dot, splint_r, dot, support_l, ...
GENESIS_PART_INDICES: dict[str, list[int]] = {
    "base.glb": [0],
    "left_finger.glb": [2],
    "right_finger.glb": [4],
    "left_outer_knuckle.glb": [6],
    "right_outer_knuckle.glb": [8],
    "left_inner_knuckle.glb": [10],
    "right_inner_knuckle.glb": [12],
}
# gripper_g2.glb Genesis bake order: solid, dot, solid, dot, ... (13 meshes)
GENESIS_INDEX_TO_GLB: dict[int, str] = {
    0: "base.glb",
    1: "base.glb",
    2: "left_finger.glb",
    3: "left_finger.glb",
    4: "right_finger.glb",
    5: "right_finger.glb",
    6: "left_outer_knuckle.glb",
    7: "left_outer_knuckle.glb",
    8: "right_outer_knuckle.glb",
    9: "right_outer_knuckle.glb",
    10: "left_inner_knuckle.glb",
    11: "left_inner_knuckle.glb",
    12: "right_inner_knuckle.glb",
}
MOVABLE_SOLID_GLBS = (
    "base.glb",
    "left_finger.glb",
    "right_finger.glb",
    "left_outer_knuckle.glb",
    "right_outer_knuckle.glb",
    "left_inner_knuckle.glb",
    "right_inner_knuckle.glb",
)
SEMANTIC_VISUAL_GLBS = frozenset(MOVABLE_SOLID_GLBS)
G2_WHITE_MATERIAL = [{"rgba": [1.0, 1.0, 1.0, 1.0], "metallic": 0.0, "roughness": 0.5}]
G2_BLACK_MATERIAL = [{"rgba": [0.07692310214042664, 0.07692310214042664, 0.07692310214042664, 1.0], "metallic": 0.0, "roughness": 0.55}]
KNUCKLE_GLBS = frozenset(
    {
        "left_outer_knuckle.glb",
        "right_outer_knuckle.glb",
        "left_inner_knuckle.glb",
        "right_inner_knuckle.glb",
    }
)
INNER_KNUCKLE_GLBS = frozenset({"left_inner_knuckle.glb", "right_inner_knuckle.glb"})
# gripper_g2_movable.glb Genesis bake indices for whole knuckle solids (link6 frame).
# Discovered via scripts/diagnose_knuckle_pipeline.py (large parts, max_ext >= 20mm).
STATIC_KNUCKLE_PART_INDICES: dict[str, list[int]] = {
    "left_outer_knuckle.glb": [6],
    "left_inner_knuckle.glb": [7],
    "right_outer_knuckle.glb": [4],
    "right_inner_knuckle.glb": [5],
}
# Legacy coarse yaw; used only when knuckle ICP is rejected.
LINK_LOCAL_VISUAL_RPY_DEG: dict[str, tuple[float, float, float]] = {
    "left_outer_knuckle.glb": (0.0, 0.0, 90.0),
    "right_outer_knuckle.glb": (0.0, 0.0, 90.0),
    "left_inner_knuckle.glb": (0.0, 0.0, 90.0),
    "right_inner_knuckle.glb": (0.0, 0.0, 90.0),
}
# Genesis semantic gripper bake uses X for left/right; URDF uses Y. Shell + fingers get Rz90.
SEMANTIC_FRAME_ROT = Rotation.from_euler("z", 90, degrees=True).as_matrix()

MOUNT_AXIS = 2  # URDF flange interface: EE/gripper_base share origin; arm body -z, tool +z
G2_RING_R = (0.015, 0.050)
G2_BOSS_R_MAX = 0.012

STL_FOR_GLB: dict[str, str] = {
    "base.glb": "base_link.STL",
    "left_outer_knuckle.glb": "left_outer_knuckle.STL",
    "left_finger.glb": "left_finger.STL",
    "left_inner_knuckle.glb": "left_inner_knuckle.STL",
    "right_outer_knuckle.glb": "right_outer_knuckle.STL",
    "right_finger.glb": "right_finger.STL",
    "right_inner_knuckle.glb": "right_inner_knuckle.STL",
}


def _node_world_mesh(scene: trimesh.Scene, node_name: str, scale: float) -> trimesh.Trimesh | None:
    """Extract a mesh for a scene-graph node with world transform applied."""
    try:
        transform, geom_name = scene.graph.get(node_name)
    except KeyError:
        return None
    if geom_name is None or geom_name not in scene.geometry:
        return None
    mesh = scene.geometry[geom_name].copy()
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.apply_transform(transform)
    mesh.apply_scale(scale)
    return mesh


def _nodes_for_group(scene: trimesh.Scene, predicates: list) -> list[str]:
    matched: list[str] = []
    for node_name in scene.graph.nodes:
        if node_name in SKIP_NODES:
            continue
        if any(pred(node_name) for pred in predicates):
            matched.append(node_name)
    return matched


def extract_movable_groups(
    glb_path: Path, scale: float = CAD_TO_METRES
) -> dict[str, tuple[list[str], list[trimesh.Trimesh]]]:
    """Group gripper_g2.glb parts by semantic name (parent + children merged per link)."""
    scene = trimesh.load(str(glb_path), force="scene")
    groups: dict[str, tuple[list[str], list[trimesh.Trimesh]]] = {}
    for out_name, predicates in MOVABLE_GROUP_PREDICATES.items():
        node_names = _nodes_for_group(scene, predicates)
        parts: list[trimesh.Trimesh] = []
        for node_name in node_names:
            mesh = _node_world_mesh(scene, node_name, scale)
            if mesh is not None:
                parts.append(mesh)
        if not parts:
            raise RuntimeError(f"No meshes found for {out_name} (nodes {node_names}) in {glb_path}")
        groups[out_name] = (node_names, parts)
    return groups


def _annular_arm_facing_faces(
    mesh: trimesh.Trimesh,
    r_inner: float,
    r_outer: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Faces pointing toward arm (-z normal) within an annulus (excludes center boss/pocket)."""
    fn = mesh.face_normals
    fc = mesh.triangles_center
    fa = mesh.area_faces
    r = np.linalg.norm(fc[:, :2], axis=1)
    mask = (fn[:, MOUNT_AXIS] < -0.85) & (r >= r_inner) & (r <= r_outer)
    return fc[mask], fa[mask]


def _area_peak_z(z: np.ndarray, areas: np.ndarray, n_bins: int = 30) -> float:
    """Z of the dominant area cluster (outer ring plane, not center boss tip)."""
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < 1e-5:
        return z_min
    edges = np.linspace(z_min, z_max, n_bins + 1)
    best_a, best_z = 0.0, z_min
    for i in range(n_bins):
        bin_mask = (z >= edges[i]) & (z < edges[i + 1])
        a = float(areas[bin_mask].sum())
        if a > best_a:
            best_a = a
            best_z = 0.5 * (edges[i] + edges[i + 1])
    return best_z


def _ring_plane_z_and_xy(
    mesh: trimesh.Trimesh,
    r_inner: float,
    r_outer: float,
    z_pick: str,
) -> tuple[float, np.ndarray]:
    """Mating plane on annular flange. z_pick='max' for link6 rim; 'area_peak' for G2 outer ring."""
    fc, fa = _annular_arm_facing_faces(mesh, r_inner, r_outer)
    if len(fc) == 0:
        raise RuntimeError(f"No annular arm-facing faces in r=[{r_inner}, {r_outer}]")
    z = fc[:, MOUNT_AXIS]
    if z_pick == "max":
        z_plane = float(z.max())
    elif z_pick == "area_peak":
        z_plane = _area_peak_z(z, fa)
    else:
        raise ValueError(z_pick)
    near = np.abs(z - z_plane) < 1e-3
    xy = fc[near, :2].mean(axis=0) if near.any() else fc[:, :2].mean(axis=0)
    return z_plane, xy


def _center_boss_tip_z(mesh: trimesh.Trimesh) -> float | None:
    """Center locating boss tip z (small radius, arm-facing); for metrics only."""
    fn = mesh.face_normals
    fc = mesh.triangles_center
    r = np.linalg.norm(fc[:, :2], axis=1)
    mask = (fn[:, MOUNT_AXIS] < -0.85) & (r < G2_BOSS_R_MAX)
    if not mask.any():
        return None
    return float(fc[mask, MOUNT_AXIS].min())


def _supported_ee_links() -> list[str]:
    return sorted(
        {
            p.ee_link
            for p in ROBOT_PROFILES.values()
            if p.supports_gripper_g2 and p.gripper_g2_visual_urdf
        }
    )


def _ee_glb_path(ee_link: str) -> Path:
    for profile in ROBOT_PROFILES.values():
        if profile.ee_link == ee_link and profile.supports_gripper_g2:
            return profile.assets_dir / "meshes" / profile.mesh_variant / "visual_glb" / f"{ee_link}.glb"
    raise KeyError(f"No profile with ee_link={ee_link}")


def align_parts_to_ee_flange(
    parts: list[trimesh.Trimesh],
    ee_glb_path: Path,
    ee_link: str,
) -> tuple[list[trimesh.Trimesh], dict]:
    """Coplanar mate: G2 outer ring flush with EE outer ring (not center boss tip)."""
    ee_ref = trimesh.load(ee_glb_path, force="mesh")
    combined = trimesh.util.concatenate([p.copy() for p in parts])
    z_ee, xy_ee = _ring_plane_z_and_xy(ee_ref, *EE_RING_R, z_pick="max")
    z_base_ring, xy_base = _ring_plane_z_and_xy(combined, *G2_RING_R, z_pick="area_peak")
    z_boss_before = _center_boss_tip_z(combined)
    delta = np.zeros(3, dtype=np.float64)
    delta[MOUNT_AXIS] = z_ee - z_base_ring
    delta[0] = xy_ee[0] - xy_base[0]
    delta[1] = xy_ee[1] - xy_base[1]
    aligned = []
    for part in parts:
        out = part.copy()
        out.vertices += delta
        aligned.append(out)
    combined_after = trimesh.util.concatenate(aligned)
    z_boss_after = _center_boss_tip_z(combined_after)
    metrics = {
        "ee_link": ee_link,
        "alignment": "outer_ring_mating_face",
        "z_ee_outer_ring_m": round(z_ee, 6),
        "z_g2_outer_ring_before_m": round(z_base_ring, 6),
        "z_g2_center_boss_before_m": round(z_boss_before, 6) if z_boss_before is not None else None,
        "z_g2_center_boss_after_m": round(z_boss_after, 6) if z_boss_after is not None else None,
        "z_shift_mm": round(float(delta[MOUNT_AXIS]) * 1000, 3),
        "xy_shift_mm": [round(float(delta[0]) * 1000, 3), round(float(delta[1]) * 1000, 3)],
        "ee_ring_xy_mm": (xy_ee * 1000).round(3).tolist(),
        "ring_gap_after_mm": round(abs(z_ee - _ring_plane_z_and_xy(combined_after, *G2_RING_R, "area_peak")[0]) * 1000, 3),
    }
    return aligned, metrics


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Genesis link quaternion (w, x, y, z) -> rotation matrix."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _pose_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    t = np.asarray(pos, dtype=np.float64).reshape(3)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_to_matrix(quat)
    mat[:3, 3] = t
    return mat


def _transform_points_row(vertices: np.ndarray, mat: np.ndarray) -> np.ndarray:
    ones = np.ones((len(vertices), 1), dtype=np.float64)
    homog = np.hstack([vertices, ones])
    return (homog @ mat.T)[:, :3]


def _gripper_link_poses_in_link6() -> dict[str, np.ndarray]:
    """URDF link poses at drive_joint=0: 4x4 maps link-local -> link6 frame."""
    import genesis as gs

    try:
        gs.init(backend=gs.cpu, logging_level="error")
    except gs.GenesisException:
        pass
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    robot = scene.add_entity(
        gs.morphs.URDF(file=robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True, movable=True), fixed=True)
    )
    scene.build()
    links = {link.name.split("/")[-1]: link for link in robot.links}
    link6 = links["link6"]
    link6_pos = np.asarray(link6.get_pos().cpu().numpy()).reshape(3)
    link6_quat = np.asarray(link6.get_quat().cpu().numpy()).reshape(4)
    t_world_link6 = _pose_matrix(link6_pos, link6_quat)
    t_link6_world = np.linalg.inv(t_world_link6)
    out: dict[str, np.ndarray] = {}
    for glb_name, link_name in GLB_TO_LINK.items():
        if glb_name == "base.glb":
            continue
        link = links.get(link_name)
        if link is None:
            raise RuntimeError(f"Gripper link missing in URDF build: {link_name} ({glb_name})")
        link_pos = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        link_quat = np.asarray(link.get_quat().cpu().numpy()).reshape(4)
        t_world_link = _pose_matrix(link_pos, link_quat)
        out[glb_name] = t_link6_world @ t_world_link
    return out


def _split_assembly_to_link(mesh_link6: trimesh.Trimesh, t_link6_link: np.ndarray) -> trimesh.Trimesh:
    """Express a link6-frame assembly mesh in a child link's local frame."""
    out = mesh_link6.copy()
    t_link_link6 = np.linalg.inv(t_link6_link)
    out.vertices = _transform_points_row(out.vertices, t_link_link6)
    return out


def _genesis_movable_aligned_link6() -> list[trimesh.Trimesh]:
    """Low-res gripper_g2.glb parts in link6 frame (Genesis Z-up bake + flange align)."""
    parts = bake_glb_genesis_parts(MOVABLE_SRC)
    ee_glb = _ee_glb_path("link6")
    aligned, _ = align_parts_to_ee_flange(parts, ee_glb, "link6")
    return aligned


def _apply_rigid_to_parts(
    parts: list[trimesh.Trimesh],
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> list[trimesh.Trimesh]:
    """Apply a shared rigid transform to copied meshes."""
    out: list[trimesh.Trimesh] = []
    for part in parts:
        mesh = part.copy()
        if rotation is not None:
            mesh.vertices = mesh.vertices @ rotation.T
        if translation is not None:
            mesh.vertices += translation
        out.append(mesh)
    return out


def _align_centroid_to_stl(mesh: trimesh.Trimesh, stl_ref: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate a link-local visual mesh so its centroid matches the STL link frame."""
    out = mesh.copy()
    out.vertices += stl_ref.centroid - out.centroid
    return out


def _icp_rigid_once(candidate: trimesh.Trimesh, stl_ref: trimesh.Trimesh) -> tuple[trimesh.Trimesh, float, dict]:
    """Single rigid ICP step (no scale); inner knuckle semantic mesh can exceed extent guard."""
    matrix, _cost = trimesh.registration.mesh_other(
        candidate,
        stl_ref,
        samples=2000,
        scale=False,
        icp_first=10,
        icp_final=50,
    )
    if np.linalg.det(matrix[:3, :3]) < 0:
        matrix = matrix @ np.diag([-1.0, 1.0, 1.0, 1.0])
    out = candidate.copy()
    out.apply_transform(matrix)
    surf_mm = mean_surface_distance(out, stl_ref) * 1000
    ratios = out.extents / np.maximum(stl_ref.extents, 1e-9)
    meta = {
        "extent_ratio_min": float(ratios.min()),
        "extent_ratio_max": float(ratios.max()),
        "det_R": float(np.linalg.det(matrix[:3, :3])),
    }
    return out, surf_mm, meta


def _align_knuckle_icp(
    mesh: trimesh.Trimesh,
    stl_ref: trimesh.Trimesh,
    glb_name: str,
    static_ref: trimesh.Trimesh | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Rigid ICP for knuckle links: fixes ~26deg semantic bake rotation residual."""
    before_mm = mean_surface_distance(mesh, stl_ref) * 1000
    best_mesh = mesh
    best_mm = before_mm
    best_score = before_mm
    best_meta: dict = {"rejected": True}
    best_seed = "centroid"
    best_mode = "none"
    best_target = "stl"

    seeds: list[tuple[str, trimesh.Trimesh]] = [("centroid", mesh)]
    if glb_name in INNER_KNUCKLE_GLBS:
        seeds.append(("link_local_z90", _apply_link_local_visual_rotation(mesh, glb_name)))

    icp_targets: list[tuple[str, trimesh.Trimesh, str]] = [("stl", stl_ref, "stl")]
    if static_ref is not None:
        icp_targets.insert(0, ("static_cad", static_ref, "static_cad"))

    for seed_name, candidate in seeds:
        for target_name, icp_target, metric_name in icp_targets:
            aligned, meta = refine_rigid_to_stl([candidate], icp_target)
            if meta.get("rejected"):
                continue
            after_stl = mean_surface_distance(aligned[0], stl_ref) * 1000
            after_static = (
                mean_surface_distance(aligned[0], static_ref) * 1000 if static_ref is not None else after_stl
            )
            if glb_name in INNER_KNUCKLE_GLBS:
                score = after_stl
            else:
                score = after_static if static_ref is not None else after_stl
            if score < best_score:
                best_mesh = aligned[0]
                best_mm = after_stl
                best_score = score
                best_meta = meta
                best_seed = seed_name
                best_mode = "strict_icp"
                best_target = metric_name

    if glb_name in INNER_KNUCKLE_GLBS:
        for seed_name, candidate in seeds:
            for target_name, icp_target, metric_name in icp_targets:
                relaxed, surf_stl, rmeta = _icp_rigid_once(candidate, icp_target)
                surf_static = (
                    mean_surface_distance(relaxed, static_ref) * 1000 if static_ref is not None else surf_stl
                )
                score = surf_stl
                if surf_stl < best_mm and abs(rmeta["det_R"] - 1.0) < 0.02 and score < best_score:
                    best_mesh = relaxed
                    best_mm = surf_stl
                    best_score = score
                    best_meta = {**rmeta, "rejected": False, "rigid": True}
                    best_seed = seed_name
                    best_mode = "relaxed_icp"
                    best_target = metric_name

    report = {
        "surf_before_icp_mm": round(float(before_mm), 2),
        "surf_after_icp_mm": round(float(best_mm), 2),
        "icp_seed": best_seed,
        "icp_mode": best_mode,
        "icp_target": best_target,
        "extent_ratio_min": round(float(best_meta.get("extent_ratio_min", 0.0)), 3),
        "extent_ratio_max": round(float(best_meta.get("extent_ratio_max", 0.0)), 3),
        "rejected": bool(best_meta.get("rejected")),
        "rigid": bool(best_meta.get("rigid")),
    }
    if best_meta.get("rejected"):
        out = _apply_link_local_visual_rotation(mesh, glb_name)
        report["fallback"] = "link_local_visual_rpy"
        report["surf_after_fallback_mm"] = round(mean_surface_distance(out, stl_ref) * 1000, 2)
        return out, report
    return best_mesh, report


def _apply_link_local_visual_rotation(mesh: trimesh.Trimesh, glb_name: str) -> trimesh.Trimesh:
    """Apply visual-only link-local orientation fixes without moving the link centroid."""
    rpy = LINK_LOCAL_VISUAL_RPY_DEG.get(glb_name)
    if rpy is None:
        return mesh
    out = mesh.copy()
    center = out.centroid.copy()
    rot = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    out.vertices = (out.vertices - center) @ rot.T + center
    return out


def _semantic_movable_candidates(
    baked_parts: list[trimesh.Trimesh],
    link_poses_link6: dict[str, np.ndarray],
    flange_metrics: dict,
    static_refs: dict[str, trimesh.Trimesh] | None = None,
) -> tuple[dict[str, trimesh.Trimesh], dict[str, dict]]:
    """Build semantic Genesis-baked candidates in URDF link-local frames."""
    ee_glb = _ee_glb_path("link6")
    base_aligned, semantic_flange_metrics = align_parts_to_ee_flange(
        [baked_parts[GENESIS_PART_INDICES["base.glb"][0]].copy()],
        ee_glb,
        "link6",
    )
    base_translation = np.array(
        [
            semantic_flange_metrics["xy_shift_mm"][0] / 1000.0,
            semantic_flange_metrics["xy_shift_mm"][1] / 1000.0,
            semantic_flange_metrics["z_shift_mm"] / 1000.0,
        ],
        dtype=np.float64,
    )
    # Genesis' GLB bake leaves the semantic gripper model with left/right spread on X.
    # URDF gripper links use Y for left/right, so rotate shell + moving parts Rz90
    # into the same flange frame as gripper_g2_static.glb (linkages already had this;
    # base/case was missing it, causing shell yaw misalignment in movable preview).
    semantic_rot = SEMANTIC_FRAME_ROT

    out: dict[str, trimesh.Trimesh] = {}
    icp_reports: dict[str, dict] = {}
    base_mesh = base_aligned[0].copy()
    # Same link6-frame Rz90 as fingers/knuckles (about origin), not a separate centroid pivot.
    base_mesh.vertices = base_mesh.vertices @ semantic_rot.T
    out["base.glb"] = base_mesh
    for glb_name, indices in GENESIS_PART_INDICES.items():
        if glb_name == "base.glb":
            continue
        parts = _apply_rigid_to_parts(
            [baked_parts[i] for i in indices],
            rotation=semantic_rot,
            translation=base_translation,
        )
        mesh_link6 = trimesh.util.concatenate(parts)
        mesh_local = _split_assembly_to_link(mesh_link6, link_poses_link6[glb_name])
        stl_ref = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        mesh_local = _align_centroid_to_stl(mesh_local, stl_ref)
        if glb_name not in KNUCKLE_GLBS:
            mesh_local = _apply_link_local_visual_rotation(mesh_local, glb_name)
        out[glb_name] = mesh_local
    return out, icp_reports


def _candidate_surface_mm(mesh: trimesh.Trimesh, stl_ref: trimesh.Trimesh) -> float:
    """Surface metric used to choose between static-split and semantic candidates."""
    return mean_surface_distance(mesh, stl_ref) * 1000


def _mesh_quality_metrics(mesh: trimesh.Trimesh) -> dict:
    """Boundary/open-edge stats; high open_edge_ratio indicates vertex-split burrs."""
    from collections import Counter

    edges = mesh.edges_sorted
    keys = [tuple(sorted((int(a), int(b)))) for a, b in edges]
    counts = Counter(keys)
    open_edges = sum(1 for c in counts.values() if c == 1)
    unique_edges = len(counts)
    return {
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "open_edges": int(open_edges),
        "open_edge_ratio": round(open_edges / max(unique_edges, 1), 4),
        "watertight": bool(mesh.is_watertight),
    }


def _bounds_volume(mesh: trimesh.Trimesh) -> float:
    ext = mesh.bounds[1] - mesh.bounds[0]
    return float(np.prod(np.maximum(ext, 1e-9)))


def _volume_ratio_vs_stl(mesh: trimesh.Trimesh, stl_ref: trimesh.Trimesh) -> float:
    return _bounds_volume(mesh) / max(_bounds_volume(stl_ref), 1e-12)


def _knuckle_candidate_score(
    mesh: trimesh.Trimesh,
    stl_ref: trimesh.Trimesh,
    method: str,
) -> float:
    """Lower is better. Penalize low volume coverage and vertex-split burrs."""
    surf_mm = mean_surface_distance(mesh, stl_ref) * 1000
    vol_ratio = _volume_ratio_vs_stl(mesh, stl_ref)
    score = float(surf_mm)
    if vol_ratio < 0.85:
        score += 25.0 * (0.85 - vol_ratio)
    if method == "static_vertex_stl_cloud":
        oer = _mesh_quality_metrics(mesh)["open_edge_ratio"]
        if oer > 0.05:
            score += 40.0 * oer
    return score


def _knuckle_from_static_whole(
    static_parts_eef: list[trimesh.Trimesh],
    link_poses_link6: dict[str, np.ndarray],
) -> dict[str, trimesh.Trimesh]:
    """Whole high-res static baked part(s) per knuckle, URDF link-local frame."""
    out: dict[str, trimesh.Trimesh] = {}
    for glb_name in KNUCKLE_GLBS:
        indices = STATIC_KNUCKLE_PART_INDICES[glb_name]
        parts = [static_parts_eef[i].copy() for i in indices]
        mesh_link6 = trimesh.util.concatenate(parts)
        mesh_local = _split_assembly_to_link(mesh_link6, link_poses_link6[glb_name])
        stl_ref = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        mesh_local = _align_centroid_to_stl(mesh_local, stl_ref)
        aligned, meta = refine_rigid_to_stl([mesh_local], stl_ref)
        if not meta.get("rejected"):
            mesh_local = aligned[0]
        out[glb_name] = mesh_local
    return out


def _procrustes_rigid(source_pts: np.ndarray, target_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch rotation + translation mapping source_pts -> target_pts."""
    src_mu = source_pts.mean(axis=0)
    tgt_mu = target_pts.mean(axis=0)
    src_c = source_pts - src_mu
    tgt_c = target_pts - tgt_mu
    h = src_c.T @ tgt_c
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = tgt_mu - src_mu @ rot.T
    return rot, trans


def _rigid_align_mesh_to_ref(source: trimesh.Trimesh, target: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """Procrustes on NN surface pairs (pose from static ref, topology from scene)."""
    n = 2000
    src_pts = source.sample(n)
    tgt_cloud = target.sample(4000)
    _, idx = cKDTree(tgt_cloud).query(src_pts, k=1)
    tgt_pts = tgt_cloud[idx]
    rot, trans = _procrustes_rigid(src_pts, tgt_pts)
    out = source.copy()
    out.vertices = out.vertices @ rot.T + trans
    warped = src_pts @ rot.T + trans
    mean_mm = float(np.linalg.norm(warped - tgt_pts, axis=1).mean() * 1000)
    meta = {
        "det_R": round(float(np.linalg.det(rot)), 4),
        "translation_mm": [round(float(x) * 1000, 2) for x in trans],
        "procrustes_mean_mm": round(mean_mm, 3),
    }
    return out, meta


def _knuckle_candidates_from_scene(
    baked_parts: list[trimesh.Trimesh],
    link_poses_link6: dict[str, np.ndarray],
) -> dict[str, trimesh.Trimesh]:
    """Clean knuckle meshes from gripper_g2.glb scene graph (Rz90 + flange, link-local)."""
    groups = extract_movable_groups(MOVABLE_SRC)
    baked_base = baked_parts[GENESIS_PART_INDICES["base.glb"][0]].copy()
    _, flange_metrics = align_parts_to_ee_flange(
        [baked_base], _ee_glb_path("link6"), "link6"
    )
    base_translation = np.array(
        [
            flange_metrics["xy_shift_mm"][0] / 1000.0,
            flange_metrics["xy_shift_mm"][1] / 1000.0,
            flange_metrics["z_shift_mm"] / 1000.0,
        ],
        dtype=np.float64,
    )
    out: dict[str, trimesh.Trimesh] = {}
    for glb_name in KNUCKLE_GLBS:
        mesh = trimesh.util.concatenate(groups[glb_name][1])
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices @ SEMANTIC_FRAME_ROT.T + base_translation
        mesh_local = _split_assembly_to_link(mesh, link_poses_link6[glb_name])
        stl_ref = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        out[glb_name] = _align_centroid_to_stl(mesh_local, stl_ref)
    return out


def _submesh_for_vertex_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh | None:
    # Majority vote: thin finger walls often share vertices across link boundaries.
    face_ok = vertex_mask[mesh.faces].sum(axis=1) >= 2
    if not np.any(face_ok):
        return None
    sub = mesh.submesh([face_ok], append=True, only_watertight=False)
    if isinstance(sub, list):
        sub = trimesh.util.concatenate(sub)
    if len(sub.vertices) == 0:
        return None
    return sub


def _vertex_labels_via_stl_cloud(
    combined_link6: trimesh.Trimesh,
    link_poses_link6: dict[str, np.ndarray],
    samples_per_link: int = 3000,
) -> dict[str, np.ndarray]:
    """Label each static vertex to a URDF link using STL collision surfaces in link6 frame."""
    ref_pts: list[np.ndarray] = []
    ref_glb: list[str] = []
    for glb_name in MOVABLE_SOLID_GLBS:
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        vertices = np.asarray(stl.vertices)
        if len(vertices) > samples_per_link:
            sample_idx = np.linspace(0, len(vertices) - 1, samples_per_link, dtype=int)
            pts = vertices[sample_idx]
        else:
            pts = vertices
        if glb_name == "base.glb":
            pts_link6 = pts
        else:
            pts_link6 = _transform_points_row(pts, link_poses_link6[glb_name])
        ref_pts.append(pts_link6)
        ref_glb.extend([glb_name] * len(pts))
    cloud = np.vstack(ref_pts)
    glb_arr = np.array(ref_glb)
    _, nn = cKDTree(cloud).query(combined_link6.vertices, k=1)
    labels = glb_arr[nn]
    return {glb: labels == glb for glb in MOVABLE_SOLID_GLBS}


def _movable_from_static_parts(
    parts_link6: list[trimesh.Trimesh],
    link_poses_link6: dict[str, np.ndarray],
) -> tuple[dict[str, trimesh.Trimesh], dict[str, int]]:
    """Vertex-split static assembly per link, then express each group in that link's local frame."""
    combined = trimesh.util.concatenate(parts_link6)
    masks = _vertex_labels_via_stl_cloud(combined, link_poses_link6)
    out: dict[str, trimesh.Trimesh] = {}
    counts: dict[str, int] = {}
    for glb_name in MOVABLE_SOLID_GLBS:
        sub = _submesh_for_vertex_mask(combined, masks[glb_name])
        if sub is None:
            raise RuntimeError(f"No faces labeled for {glb_name} (genesis cloud split)")
        counts[glb_name] = int(masks[glb_name].sum())
        if glb_name == "base.glb":
            out[glb_name] = sub
        else:
            out[glb_name] = _split_assembly_to_link(sub, link_poses_link6[glb_name])
    return out, counts


def _extract_parts_genesis(
    baked_parts: list[trimesh.Trimesh],
    indices: list[int],
    label: str,
) -> list[trimesh.Trimesh]:
    """Pick solid meshes from a Genesis bake (trimesh scene graph flattens many nodes to 2D sheets)."""
    selected = [baked_parts[i].copy() for i in indices]
    for mesh in selected:
        extent = mesh.vertices.max(0) - mesh.vertices.min(0)
        if float(extent.max()) < 1e-6:
            raise RuntimeError(f"Genesis part degenerate for {label} (indices {indices})")
    return selected


def relocalize_shared_movable_parts(
    dry_run: bool,
    static_parts_eef: list[trimesh.Trimesh],
    flange_metrics: dict,
) -> list[dict]:
    """Generate 6 shared link-local GLBs (fingers/knuckles; base is per-EE)."""
    report: list[dict] = []
    groups = extract_movable_groups(MOVABLE_SRC)
    link_poses_link6 = _gripper_link_poses_in_link6()
    baked_parts = bake_glb_genesis_parts(MOVABLE_SRC)
    static_candidates, vertex_counts = _movable_from_static_parts(static_parts_eef, link_poses_link6)
    semantic_candidates, knuckle_icp_reports = _semantic_movable_candidates(
        baked_parts,
        link_poses_link6,
        flange_metrics,
        static_refs=static_candidates,
    )
    scene_knuckles = _knuckle_candidates_from_scene(baked_parts, link_poses_link6)
    static_whole_knuckles = _knuckle_from_static_whole(static_parts_eef, link_poses_link6)
    knuckle_pose_aligned: dict[str, trimesh.Trimesh] = {}
    knuckle_align_meta: dict[str, dict] = {}
    for glb_name in KNUCKLE_GLBS:
        aligned, meta = _rigid_align_mesh_to_ref(
            scene_knuckles[glb_name],
            static_candidates[glb_name],
        )
        knuckle_pose_aligned[glb_name] = aligned
        knuckle_align_meta[glb_name] = meta

    if not dry_run:
        VISUAL_GLB_OUT.mkdir(parents=True, exist_ok=True)

    for glb_name, (node_names, _) in groups.items():
        if glb_name == "base.glb":
            continue
        stl_ref = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        candidates = {
            "static_vertex_stl_cloud": static_candidates[glb_name],
            "semantic_genesis_centroid": semantic_candidates[glb_name],
        }
        if glb_name in KNUCKLE_GLBS:
            candidates["scene_graph_link_local"] = scene_knuckles[glb_name]
            candidates["static_whole_link_local"] = static_whole_knuckles[glb_name]
            candidates["scene_graph_pose_static_ref"] = knuckle_pose_aligned[glb_name]
        candidate_metrics = {
            name: _candidate_surface_mm(mesh, stl_ref) for name, mesh in candidates.items()
        }
        if glb_name in KNUCKLE_GLBS:
            knuckle_scores = {
                name: round(_knuckle_candidate_score(mesh, stl_ref, name), 3)
                for name, mesh in candidates.items()
            }
            method = min(knuckle_scores, key=knuckle_scores.get)
            combined = candidates[method]
        else:
            method = "semantic_genesis_centroid"
            combined = candidates[method]
            knuckle_scores = {}
        surf_mm = mean_surface_distance(combined, stl_ref) * 1000
        geom_centroid_mm = float(np.linalg.norm(combined.centroid - stl_ref.centroid) * 1000)
        entry = {
            "file": glb_name,
            "scope": "shared",
            "source": (
                "gripper_g2_movable.glb"
                if method == "static_whole_link_local"
                else (
                    MOVABLE_SRC.name
                    if method in ("scene_graph_pose_static_ref", "scene_graph_link_local")
                    else (
                        "gripper_g2_static_link6.glb"
                        if method == "static_vertex_stl_cloud"
                        else MOVABLE_SRC.name
                    )
                )
            ),
            "method": method,
            "static_knuckle_part_indices": list(STATIC_KNUCKLE_PART_INDICES.get(glb_name, [])),
            "labeled_vertices": vertex_counts[glb_name],
            "static_parts_in": len(static_parts_eef),
            "nodes": node_names,
            "parts": 1,
            "verts": int(len(combined.vertices)),
            "mean_surface_mm": round(surf_mm, 2),
            "geom_centroid_vs_stl_mm": round(geom_centroid_mm, 2),
            "candidate_mean_surface_mm": {
                name: round(value, 2) for name, value in candidate_metrics.items()
            },
            **({"candidate_scores": knuckle_scores} if knuckle_scores else {}),
            "link_local_visual_rpy_deg": list(LINK_LOCAL_VISUAL_RPY_DEG.get(glb_name, (0.0, 0.0, 0.0))),
            **({"knuckle_icp": knuckle_icp_reports[glb_name]} if glb_name in knuckle_icp_reports else {}),
        }
        report.append(entry)
        print(
            f"{glb_name}: method={method}, verts={entry['verts']}, "
            f"mean_surface={entry['mean_surface_mm']:.1f}mm, "
            f"geom_centroid_vs_stl={entry['geom_centroid_vs_stl_mm']:.1f}mm"
        )
        if not dry_run:
            material = G2_WHITE_MATERIAL if glb_name == "base.glb" else G2_BLACK_MATERIAL
            export_opaque_doublesided_glb([combined], VISUAL_GLB_OUT / glb_name, material)

    return report


def relocalize_base_for_ee(
    ee_link: str,
    dry_run: bool,
    flange_metrics: dict,
) -> dict:
    """Align base shell GLB to a specific EE flange."""
    ee_glb = _ee_glb_path(ee_link)
    if not ee_glb.is_file():
        raise FileNotFoundError(f"Missing EE GLB for {ee_link}: {ee_glb}")
    baked_parts = bake_glb_genesis_parts(MOVABLE_SRC)
    base_aligned, _ = align_parts_to_ee_flange(
        [baked_parts[GENESIS_PART_INDICES["base.glb"][0]].copy()],
        ee_glb,
        ee_link,
    )
    combined = base_aligned[0].copy()
    combined.vertices = combined.vertices @ SEMANTIC_FRAME_ROT.T
    stl_ref = trimesh.load(GRIPPER_STL_DIR / "base_link.STL", force="mesh")
    ee_ref = trimesh.load(ee_glb, force="mesh")
    z_g2_ring = _ring_plane_z_and_xy(combined, *G2_RING_R, "area_peak")[0]
    z_ee_ring = _ring_plane_z_and_xy(ee_ref, *EE_RING_R, "max")[0]
    flange_gap_mm = abs(z_g2_ring - z_ee_ring) * 1000
    out_name = f"visual_glb/{ee_link}/base.glb"
    entry = {
        "file": out_name,
        "scope": ee_link,
        "source": f"gripper_g2_static_{ee_link}.glb",
        "verts": int(len(combined.vertices)),
        "mean_surface_mm": round(mean_surface_distance(combined, stl_ref) * 1000, 2),
        "flange_gap_mm": round(flange_gap_mm, 3),
        **flange_metrics,
    }
    print(f"{out_name}: flange_gap={flange_gap_mm:.3f}mm")
    if not dry_run:
        out_dir = VISUAL_GLB_OUT / ee_link
        out_dir.mkdir(parents=True, exist_ok=True)
        export_opaque_doublesided_glb([combined], out_dir / "base.glb", G2_WHITE_MATERIAL)
    return entry


def relocalize_static_assembly(ee_link: str, dry_run: bool) -> tuple[dict, list[trimesh.Trimesh]]:
    """Bake gripper_g2_movable.glb and align mating top flush with EE flange."""
    ee_glb = _ee_glb_path(ee_link)
    if not ee_glb.is_file():
        raise FileNotFoundError(f"Missing EE GLB for {ee_link}: {ee_glb}")
    parts = bake_glb_genesis_parts(STATIC_SRC)
    aligned, flange_metrics = align_parts_to_ee_flange(parts, ee_glb, ee_link)
    combined = trimesh.util.concatenate(aligned)
    stl_ref = trimesh.load(GRIPPER_STL_DIR / "base_link.STL", force="mesh")
    pbr_materials = _raw_material_pbr(STATIC_SRC)
    surf_mm = mean_surface_distance(combined, stl_ref) * 1000
    geom_centroid_mm = float(np.linalg.norm(combined.centroid - stl_ref.centroid) * 1000)
    ee_ref = trimesh.load(ee_glb, force="mesh")
    z_g2_ring = _ring_plane_z_and_xy(combined, *G2_RING_R, "area_peak")[0]
    z_ee_ring = _ring_plane_z_and_xy(ee_ref, *EE_RING_R, "max")[0]
    flange_gap_mm = abs(z_g2_ring - z_ee_ring) * 1000
    out_name = f"gripper_g2_static_{ee_link}.glb"
    entry = {
        "file": out_name,
        "scope": ee_link,
        "source": "gripper_g2_movable.glb",
        "parts": len(aligned),
        "mean_surface_mm": round(surf_mm, 2),
        "geom_centroid_vs_stl_mm": round(geom_centroid_mm, 2),
        "flange_gap_mm": round(flange_gap_mm, 3),
        **flange_metrics,
    }
    print(
        f"{out_name}: parts={entry['parts']}, mean_surface={entry['mean_surface_mm']:.1f}mm, "
        f"flange_gap={flange_gap_mm:.3f}mm"
    )
    if not dry_run:
        export_opaque_doublesided_glb(aligned, GRIPPER_VISUAL_DIR / out_name, pbr_materials)
    return entry, aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Relocalize Gripper G2 GLB meshes to link frames")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report alignment metrics without writing files",
    )
    args = parser.parse_args()

    if not MOVABLE_SRC.exists():
        raise FileNotFoundError(MOVABLE_SRC)
    if not STATIC_SRC.exists():
        raise FileNotFoundError(STATIC_SRC)

    _init_genesis()
    ee_links = _supported_ee_links()
    report: list[dict] = []

    static_parts_by_ee: dict[str, list[trimesh.Trimesh]] = {}
    for ee_link in ee_links:
        static_entry, static_parts = relocalize_static_assembly(ee_link, args.dry_run)
        report.append(static_entry)
        static_parts_by_ee[ee_link] = static_parts

    link6_static = static_parts_by_ee.get("link6")
    if link6_static is None:
        raise RuntimeError("link6 required as reference for shared movable parts")
    flange_meta = {
        k: v
        for k, v in report[0].items()
        if k.startswith(("alignment", "z_", "xy_", "ring_", "flange", "ee_"))
    }
    report.extend(relocalize_shared_movable_parts(args.dry_run, link6_static, flange_meta))

    for ee_link in ee_links:
        static_entry = next(e for e in report if e.get("scope") == ee_link and e["file"].startswith("gripper_g2_static"))
        flange_meta = {
            k: static_entry[k]
            for k in static_entry
            if k.startswith(("alignment", "z_", "xy_", "ring_", "flange", "ee_"))
        }
        report.append(relocalize_base_for_ee(ee_link, args.dry_run, flange_meta))

    if not args.dry_run:
        metrics_path = GRIPPER_VISUAL_DIR / "relocalize_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote shared GLBs to {VISUAL_GLB_OUT}")
        print(f"Wrote per-EE static/base GLBs under {GRIPPER_VISUAL_DIR}")
        print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
