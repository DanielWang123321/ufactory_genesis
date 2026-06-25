#!/usr/bin/env python3
"""
Relocalize Bio Gripper G2 GLB for static and movable visual combo URDFs.

Per EE link (link5 / link6 / link7):
  1. Genesis bake source CAD GLB (×0.1 m)
  2. Coarse CAD→STL (z +270°, centroid match link_base.stl)
  3. Pin–hole Kabsch: gripper locating pins ↔ arm flange locating holes
  4. Z-only outer-ring coplanar refine (same annulus as G2 gripper)
  5. Export static merged mesh + per-EE bio_gripper_g2_base.glb

Per EE link (movable visual GLBs in visual_glb/{ee_link}/):
  bio_gripper_g2_base.glb, bio_gripper_g2_left_finger.glb, bio_gripper_g2_right_finger.glb

Use bio_* GLB names to avoid Genesis basename clashes with arm link_base.glb / G2 fingers.
CAD GLB's compressed vertices). Finger GLBs are EE-aligned CAD groups mapped into
URDF finger link frames so they match the static merged visual with no per-visual flip.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from relocalize_arm_glb import (
    _init_genesis,
    bake_glb_genesis_parts,
    export_opaque_doublesided_glb,
    mean_surface_distance,
)
from relocalize_gripper_glb import (
    EE_RING_R,
    G2_RING_R,
    MOUNT_AXIS,
    _mesh_quality_metrics,
    _procrustes_rigid,
    _ring_plane_z_and_xy,
)
from ufactory.paths import BIO_GRIPPER_G2_ASSETS
from ufactory.robot_registry import ROBOT_PROFILES

VISUAL_DIR = BIO_GRIPPER_G2_ASSETS / "meshes" / "visual"
GLB_OUT_DIR = VISUAL_DIR / "visual_glb"
STL_BASE = VISUAL_DIR / "link_base.stl"
STL_LEFT_FINGER = VISUAL_DIR / "left_finger.stl"
STL_RIGHT_FINGER = VISUAL_DIR / "right_finger.stl"
SRC_GLB = VISUAL_DIR / "visual_glb_src" / "bio_gripper_g2.glb"
FALLBACK_SRC = VISUAL_DIR / "bio_gripper_g2.glb"
METRICS_PATH = VISUAL_DIR / "relocalize_metrics.json"

CAD_TO_STL_ROT_Z_DEG = 270.0
PIN_ANNULUS_R = (0.020, 0.028)
PIN_SEARCH_RADIUS_M = 0.015
Y_SPLIT_M = 0.003
# bio_gripper_g2.urdf finger joint origin in base frame
FINGER_JOINT_ORIGIN = np.array([0.059, 0.0, 0.027], dtype=np.float64)
# bio_gripper_g2.urdf TCP direction in gripper base frame (+X with pitch)
FINGER_TARGET = np.array([0.135, 0.0, 0.055], dtype=np.float64)
FINGER_TARGET /= np.linalg.norm(FINGER_TARGET)

_CAD_TO_STL_RZ = Rotation.from_euler("z", CAD_TO_STL_ROT_Z_DEG, degrees=True).as_matrix()
CAD_TO_METRES = 0.1

# CAD source GLB: case=white plastic (met=0), splint/flange=silver metal (met≈0.8, same as arm EE)
BIO_WHITE_PLASTIC = [{"rgba": [1.0, 1.0, 1.0, 1.0], "metallic": 0.0, "roughness": 0.5}]
# xArm/UF850 link6 EE flange metal (read from relocalized arm GLB)
ARM_EE_METAL = [
    {
        "rgba": [0.9529411764705883, 0.9529411764705883, 0.9490196078431372, 1.0],
        "metallic": 0.8,
        "roughness": 0.640130877494812,
    }
]
METAL_METALLIC_MIN = 0.5
FLANGE_Z_SLACK_M = 0.008
FLANGE_MAX_X_M = 0.08
BIO_GRIPPER_G2_BASE_GLB = "bio_gripper_g2_base.glb"
BIO_LEFT_FINGER_GLB = "bio_gripper_g2_left_finger.glb"
BIO_RIGHT_FINGER_GLB = "bio_gripper_g2_right_finger.glb"
METAL_SUBDIVIDE_ITERS = 3
FINGER_SUBDIVIDE_ITERS = 2
# Moving-jaw classification: a source-CAD node body is a finger jaw only if it fits
# a finger collision STL (under the prismatic Y dof) within this residual. The two
# real jaws fit at ~10 mm; the static housing/rail bodies sit at >=33 mm, so the
# threshold sits safely inside that gap. This keeps body screws / the UFACTORY
# nameplate on the static base link instead of riding along with the fingers.
FINGER_NODE_RESIDUAL_MAX_M = 0.020
_FINGER_YSHIFT_GRID = np.linspace(-0.05, 0.05, 41)
# Real Bio Gripper G2 closed two-finger gap (jaw=0). Stroke opens to 150 mm.
CLOSED_GAP_M = 0.071
OPEN_GAP_M = 0.150


def _align_coarse_part(
    part: trimesh.Trimesh,
    center_offset: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dz: float,
) -> trimesh.Trimesh:
    coarse_part = _part_to_assembly_coarse(part, center_offset)
    return _apply_transform(coarse_part, R, t, dz)


def _align_genesis_group(
    groups: dict[str, list[trimesh.Trimesh]],
    key: str,
    center_offset: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dz: float,
) -> trimesh.Trimesh | None:
    merged_part = _merge_group(groups[key])
    if merged_part is None:
        return None
    return _align_coarse_part(merged_part, center_offset, R, t, dz)


def _bake_bio_material_parts(glb_path: Path) -> list[tuple[trimesh.Trimesh, float]]:
    """Genesis bake with per-CAD-material meshes (preserves plastic vs metal)."""
    from genesis.options import surfaces
    from genesis.utils import gltf as gltf_utils

    surface = surfaces.Default()
    meshes = gltf_utils.parse_mesh_glb(
        str(glb_path),
        group_by_material=True,
        scale=None,
        is_mesh_zup=True,
        surface=surface,
    )
    if not meshes:
        raise RuntimeError(f"No geometry in {glb_path}")
    out: list[tuple[trimesh.Trimesh, float]] = []
    for mesh in meshes:
        part = mesh.trimesh.copy()
        part.apply_scale(CAD_TO_METRES)
        met_tex = mesh.surface.metallic_texture
        metallic = (
            float(met_tex.color[0])
            if met_tex is not None and met_tex.color is not None
            else 0.0
        )
        out.append((part, metallic))
    return out


def _apply_rigid(mesh: trimesh.Trimesh, rot: np.ndarray, trans: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = out.vertices @ rot.T + trans
    return out


def _gripper_frame_rt(
    groups: dict[str, list[trimesh.Trimesh]],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid pose mapping genesis base assembly → URDF link_base.stl frame."""
    merged_base = _merge_group(groups["base"])
    assert merged_base is not None
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    base_local = _part_to_assembly_coarse(merged_base, center_offset)
    n = 2000
    src_pts = base_local.sample(n)
    tgt_cloud = stl_base.sample(4000)
    _, idx = cKDTree(tgt_cloud).query(src_pts, k=1)
    tgt_pts = tgt_cloud[idx]
    return _procrustes_rigid(src_pts, tgt_pts)


def _is_finger_region_part(coarse_part: trimesh.Trimesh) -> bool:
    return abs(float(coarse_part.centroid[1])) > Y_SPLIT_M


def _is_flange_metal_part(part: trimesh.Trimesh, base_zmin: float) -> bool:
    """Keep only mount-plate metal; exclude finger rails mis-tagged near Y=0."""
    if float(part.vertices[:, 2].min()) > base_zmin + FLANGE_Z_SLACK_M:
        return False
    if float(np.max(part.vertices[:, 0])) > FLANGE_MAX_X_M:
        return False
    return True


def _subdivide_visual_mesh(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    out = mesh.copy()
    for _ in range(iterations):
        out = out.subdivide()
    return out


def _movable_base_gripper_frame(
    material_parts: list[tuple[trimesh.Trimesh, float]],
    jaw_labels: list[str | None],
    rot: np.ndarray,
    trans: np.ndarray,
    center_offset: np.ndarray,
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Movable base in the canonical gripper base_link frame (EE-link independent).

    Everything except the two moving jaw nodes (white-plastic shell plus all non-jaw
    metal: mount flange, housing rail, body screws, UFACTORY nameplate), expressed in
    the link_base.stl frame.  The combo URDF mount joint (rpy Rx(pi)) orients this
    base onto each arm flange, so the same base mesh works for every EE link and for
    the standalone (rpy 0) gripper.
    """
    plastic_parts: list[trimesh.Trimesh] = []
    metal_parts: list[trimesh.Trimesh] = []
    for (part, metallic), label in zip(material_parts, jaw_labels):
        if label is not None:
            continue
        g = _apply_rigid(_part_to_assembly_coarse(part, center_offset), rot, trans)
        if metallic <= 0.01:
            plastic_parts.append(g)
        elif metallic >= METAL_METALLIC_MIN:
            metal_parts.append(g)
    plastic = trimesh.util.concatenate(plastic_parts) if plastic_parts else None
    metal = trimesh.util.concatenate(metal_parts) if metal_parts else None
    if metal is not None:
        metal = _subdivide_visual_mesh(metal, METAL_SUBDIVIDE_ITERS)
    return plastic, metal


def _finger_meshes_gripper_frame(
    material_parts: list[tuple[trimesh.Trimesh, float]],
    jaw_labels: list[str | None],
    finger_yshift: dict[str, float],
    rot: np.ndarray,
    trans: np.ndarray,
    center_offset: np.ndarray,
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Moving jaws in the URDF finger link frames, built in the canonical gripper frame.

    The jaws are mapped into the link_base.stl (base_link) frame where both the
    left/right jaw labels and ``finger_yshift`` were computed, so the closed-pose
    registration is valid and the result is EE-link independent.  Each side is shifted
    in Y onto its joint-zero collision STL (closed pose) and then expressed in the
    finger link frame, so the [0, limit] prismatic stroke reproduces the real
    71 -> 150 mm opening for the standalone and every arm combo alike.
    """
    left_parts: list[trimesh.Trimesh] = []
    right_parts: list[trimesh.Trimesh] = []
    for (part, _metallic), label in zip(material_parts, jaw_labels):
        if label is None:
            continue
        g = _apply_rigid(_part_to_assembly_coarse(part, center_offset), rot, trans)
        if label == "left":
            left_parts.append(g)
        else:
            right_parts.append(g)
    left = trimesh.util.concatenate(left_parts) if left_parts else None
    right = trimesh.util.concatenate(right_parts) if right_parts else None
    # finger_yshift gives a coarse closed registration; a final calibration measured on
    # the assembled finger mesh then snaps each distal gripping blade face exactly onto
    # +/- CLOSED_GAP_M/2, so the closed two-finger distance is exactly the real 71 mm
    # (self-consistent regardless of node-vs-assembly sampling differences).
    if left is not None:
        left.vertices[:, 1] += finger_yshift.get("left", 0.0)
        left = _subdivide_visual_mesh(_finger_mesh_in_link_frame(left), FINGER_SUBDIVIDE_ITERS)
        left.vertices[:, 1] += (-CLOSED_GAP_M / 2.0) - _blade_inner_face(left.vertices, "left")
    if right is not None:
        right.vertices[:, 1] += finger_yshift.get("right", 0.0)
        right = _subdivide_visual_mesh(_finger_mesh_in_link_frame(right), FINGER_SUBDIVIDE_ITERS)
        right.vertices[:, 1] += (CLOSED_GAP_M / 2.0) - _blade_inner_face(right.vertices, "right")
    return left, right


def _movable_base_meshes_material_split(
    material_parts: list[tuple[trimesh.Trimesh, float]],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
    groups: dict[str, list[trimesh.Trimesh]],
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Split base into white plastic shell + silver adapter flange (multi-material GLB)."""
    rot, trans = _gripper_frame_rt(groups, merged_coarse, stl_base)
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    base_zmin = float(stl_base.vertices[:, 2].min())
    plastic_parts: list[trimesh.Trimesh] = []
    metal_parts: list[trimesh.Trimesh] = []
    for part, metallic in material_parts:
        coarse = _apply_rigid(_part_to_assembly_coarse(part, center_offset), rot, trans)
        if _is_finger_region_part(coarse):
            continue
        if metallic <= 0.01:
            plastic_parts.append(coarse)
        elif metallic >= METAL_METALLIC_MIN and _is_flange_metal_part(coarse, base_zmin):
            metal_parts.append(coarse)
    plastic = trimesh.util.concatenate(plastic_parts) if plastic_parts else None
    metal = trimesh.util.concatenate(metal_parts) if metal_parts else None
    if metal is not None:
        metal = _subdivide_visual_mesh(metal, METAL_SUBDIVIDE_ITERS)
    return plastic, metal


def _static_ee_material_meshes(
    material_parts: list[tuple[trimesh.Trimesh, float]],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
    R: np.ndarray,
    t: np.ndarray,
    dz: float,
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Full gripper in EE link frame: plastic shell + all metal CAD parts."""
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    plastic_parts: list[trimesh.Trimesh] = []
    metal_parts: list[trimesh.Trimesh] = []
    for part, metallic in material_parts:
        coarse = _part_to_assembly_coarse(part, center_offset)
        aligned = _apply_transform(coarse, R, t, dz)
        if metallic <= 0.01:
            plastic_parts.append(aligned)
        elif metallic >= METAL_METALLIC_MIN:
            metal_parts.append(aligned)
    plastic = trimesh.util.concatenate(plastic_parts) if plastic_parts else None
    metal = trimesh.util.concatenate(metal_parts) if metal_parts else None
    if metal is not None:
        metal = _subdivide_visual_mesh(metal, METAL_SUBDIVIDE_ITERS)
    return plastic, metal


def _export_link_base_glb(
    plastic: trimesh.Trimesh | None,
    metal: trimesh.Trimesh | None,
    out_path: Path,
) -> dict:
    meshes: list[trimesh.Trimesh] = []
    materials: list[dict] = []
    if plastic is not None and len(plastic.vertices):
        meshes.append(plastic)
        materials.append(BIO_WHITE_PLASTIC[0])
    if metal is not None and len(metal.vertices):
        meshes.append(metal)
        materials.append(ARM_EE_METAL[0])
    if not meshes:
        raise RuntimeError("No base meshes to export")
    export_opaque_doublesided_glb(meshes, out_path, materials)
    return {
        "submesh_count": len(meshes),
        "plastic_verts": int(plastic.vertices.shape[0]) if plastic is not None else 0,
        "metal_verts": int(metal.vertices.shape[0]) if metal is not None else 0,
    }


def _finger_mesh_in_link_frame(mesh_ee: trimesh.Trimesh) -> trimesh.Trimesh:
    """Finger link mesh with per-link visual rpy=0: t + p = p_ee in base frame."""
    out = mesh_ee.copy()
    out.vertices = out.vertices - FINGER_JOINT_ORIGIN
    return out


def _best_finger_yshift_residual(verts: np.ndarray, tree: cKDTree) -> tuple[float, float]:
    """Min mean point->STL distance over the prismatic Y dof, and the Y-shift achieving it.

    The finger slides only in Y, so registering the source-CAD jaw (at its native
    modelled opening) onto the joint-zero collision STL is a pure Y translation. A
    coarse grid locates the basin; a fine local sweep refines the shift to <1 mm so
    the visual jaw sits flush against the body at joint=0.
    """

    def mean_dist(dy: float) -> float:
        shifted = verts.copy()
        shifted[:, 1] += dy
        return float(np.mean(tree.query(shifted)[0]))

    coarse = min(_FINGER_YSHIFT_GRID, key=mean_dist)
    fine_grid = np.linspace(coarse - 0.003, coarse + 0.003, 49)
    best_dy = min(fine_grid, key=mean_dist)
    return mean_dist(best_dy), float(best_dy)


def _blade_inner_face(verts: np.ndarray, side: str) -> float:
    """Center-facing Y of the distal gripping blade flat face (robust to chamfers).

    The jaw slides only in Y, so the closed two-finger distance is set by where the
    distal blade's inner surface sits.  We sample the distal blade (high X) and take a
    near-boundary percentile of the inner-facing Y so a small chamfer/edge vertex does
    not bias the closed gap (a pure max/min would over-reach to the chamfer tip).
    """
    distal = verts[verts[:, 0] >= np.percentile(verts[:, 0], 70)]
    # left jaw sits at -Y → inner face toward +Y (high pct); right jaw at +Y → low pct
    return float(np.percentile(distal[:, 1], 85 if side == "left" else 15))


def _jaw_labels_for_material_parts(
    parts: list[trimesh.Trimesh],
    material_parts: list[tuple[trimesh.Trimesh, float]],
    groups: dict[str, list[trimesh.Trimesh]],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
) -> tuple[list[str | None], dict[str, float]]:
    """Classify each material part as a moving finger jaw ('left'/'right') or static (None).

    Works at the source-CAD *node* level (group_by_material=False bodies): only the
    two nodes that fit a finger collision STL within FINGER_NODE_RESIDUAL_MAX_M are
    jaws. Each per-material part inherits the label of the node it belongs to, so the
    housing rail, mount flange, body screws, and the UFACTORY nameplate stay static.
    Classification runs in the link_base.stl (base link) frame where the finger STLs
    live; left/right is frame-consistent with the aligned EE frame (Y is preserved).

    Also returns the per-side Y-shift (base frame, = GLB frame since Y is preserved)
    that registers each jaw node onto its joint-zero collision STL, so the visual
    fingers sit closed/flush at joint=0 instead of at the CAD's native open pose.
    """
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    rot, trans = _gripper_frame_rt(groups, merged_coarse, stl_base)
    left_pts = trimesh.load(STL_LEFT_FINGER, force="mesh").vertices + FINGER_JOINT_ORIGIN
    right_pts = trimesh.load(STL_RIGHT_FINGER, force="mesh").vertices + FINGER_JOINT_ORIGIN
    left_tree = cKDTree(left_pts)
    right_tree = cKDTree(right_pts)

    def to_base_frame(part: trimesh.Trimesh) -> np.ndarray:
        return _apply_rigid(_part_to_assembly_coarse(part, center_offset), rot, trans).vertices

    # Joint-0 (closed) target: place each jaw's distal gripping blade face at
    # +/- CLOSED_GAP_M/2 so the closed two-finger distance equals the real 71 mm, and
    # the [0, limit] prismatic stroke opens symmetrically to the real 150 mm.  We target
    # the spec gap directly (rather than full-cloud ICP onto the simplified collision
    # STL, whose mount riser pulls the blade past center) so the gripping faces land on
    # the real mechanical range regardless of CAD-vs-STL shape differences.
    target_inner = {"left": -CLOSED_GAP_M / 2.0, "right": CLOSED_GAP_M / 2.0}

    node_trees: list[cKDTree] = []
    node_labels: list[str | None] = []
    finger_yshift: dict[str, float] = {"left": 0.0, "right": 0.0}
    for part in parts:
        verts = to_base_frame(part)
        res_left, _ = _best_finger_yshift_residual(verts, left_tree)
        res_right, _ = _best_finger_yshift_residual(verts, right_tree)
        if min(res_left, res_right) <= FINGER_NODE_RESIDUAL_MAX_M:
            side = "left" if res_left < res_right else "right"
            node_labels.append(side)
            finger_yshift[side] = target_inner[side] - _blade_inner_face(verts, side)
        else:
            node_labels.append(None)
        node_trees.append(cKDTree(verts))

    labels: list[str | None] = []
    for part, _metallic in material_parts:
        verts = to_base_frame(part)
        node_idx = int(np.argmin([float(np.mean(t.query(verts)[0])) for t in node_trees]))
        labels.append(node_labels[node_idx])
    return labels, finger_yshift


def _finger_direction(mesh: trimesh.Trimesh) -> np.ndarray:
    finger_pts = mesh.vertices[mesh.vertices[:, 0] > np.percentile(mesh.vertices[:, 0], 75)]
    vec = finger_pts.mean(axis=0)
    return vec / (np.linalg.norm(vec) + 1e-12)


def _correct_finger_opening(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Opening along +X in EE frame; keep finger in XZ plane (no roll tilt)."""
    out = mesh.copy()
    fd = _finger_direction(out)
    if fd[0] < 0.0:
        rz = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
        out.vertices = out.vertices @ rz.T
        fd = _finger_direction(out)
    if abs(fd[1]) > 0.03:
        elev = np.arctan2(fd[1], np.hypot(fd[0], fd[2]))
        rx = Rotation.from_euler("x", -elev).as_matrix()
        out.vertices = out.vertices @ rx.T
    return out


def _kabsch(P: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cp = P.mean(axis=0)
    ch = H.mean(axis=0)
    A = (P - cp).T @ (H - ch)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = ch - R @ cp
    return R, t


def _gripper_pin_points_stl(stl_base: trimesh.Trimesh) -> np.ndarray:
    verts = stl_base.vertices
    x_plane = float(verts[:, 0].min())
    yz_r = np.linalg.norm(verts[:, 1:3], axis=1)
    cand = verts[
        (yz_r > PIN_ANNULUS_R[0])
        & (yz_r < PIN_ANNULUS_R[1])
        & (verts[:, 0] < x_plane + 0.012)
    ]
    if len(cand) < 2:
        raise RuntimeError("Could not find locating pins on link_base.stl mount face")
    pos = cand[cand[:, 1] > 0].mean(axis=0)
    neg = cand[cand[:, 1] < 0].mean(axis=0)
    return np.array([pos, neg], dtype=np.float64)


def _arm_locating_holes(ee_mesh: trimesh.Trimesh) -> np.ndarray:
    fn = ee_mesh.face_normals
    fc = ee_mesh.triangles_center
    af = fn[:, MOUNT_AXIS] < -0.85
    r = np.linalg.norm(fc[:, :2], axis=1)
    flange = af & (r >= EE_RING_R[0]) & (r <= EE_RING_R[1])
    if not flange.any():
        raise RuntimeError("No EE flange annulus found for hole detection")
    z = float(fc[flange, MOUNT_AXIS].max())
    hf = (fn[:, MOUNT_AXIS] > 0.2) & (np.abs(fc[:, MOUNT_AXIS] - z) < 0.008)
    hf &= (np.linalg.norm(fc[:, :2], axis=1) >= 0.015) & (np.linalg.norm(fc[:, :2], axis=1) <= 0.026)
    pts_xy = fc[hf][:, :2]
    if len(pts_xy) < 20:
        raise RuntimeError("No locating hole floor points on EE flange")
    r = np.linalg.norm(pts_xy, axis=1)
    pts_xy = pts_xy[(r > PIN_ANNULUS_R[0]) & (r < PIN_ANNULUS_R[1])]
    if len(pts_xy) < 10:
        raise RuntimeError("No locating holes in pin annulus on EE flange")
    pos = pts_xy[pts_xy[:, 1] > 0].mean(axis=0)
    neg = pts_xy[pts_xy[:, 1] < 0].mean(axis=0)
    return np.array([[pos[0], pos[1], z], [neg[0], neg[1], z]], dtype=np.float64)


def _pin_vertices_on_mesh(mesh: trimesh.Trimesh, stl_pins: np.ndarray) -> np.ndarray:
    verts = mesh.vertices
    tree = cKDTree(verts)
    pin_verts: list[np.ndarray] = []
    for pin in stl_pins:
        mask = np.linalg.norm(verts - pin, axis=1) < PIN_SEARCH_RADIUS_M
        if pin[1] >= 0:
            mask &= verts[:, 1] > -0.005
        else:
            mask &= verts[:, 1] < 0.005
        sub = verts[mask]
        if len(sub) < 5:
            pin_verts.append(verts[tree.query(pin)[1]])
        else:
            pin_verts.append(sub[sub[:, 0].argmin()])
    return np.array(pin_verts, dtype=np.float64)


def _cad_to_stl_coarse(mesh: trimesh.Trimesh, stl_base: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = out.vertices @ _CAD_TO_STL_RZ.T
    out.vertices += stl_base.centroid - out.centroid
    return out


def _assembly_center_offset(merged: trimesh.Trimesh, stl_base: trimesh.Trimesh) -> np.ndarray:
    """Shared CAD→STL centroid shift from the full assembly (not per-part)."""
    rotated = merged.vertices @ _CAD_TO_STL_RZ.T
    return stl_base.centroid - rotated.mean(axis=0)


def _part_to_assembly_coarse(part: trimesh.Trimesh, center_offset: np.ndarray) -> trimesh.Trimesh:
    out = part.copy()
    out.vertices = out.vertices @ _CAD_TO_STL_RZ.T + center_offset
    return out


def _z_shift_to_flange(mesh: trimesh.Trimesh, ee_mesh: trimesh.Trimesh) -> float:
    z_ee, _ = _ring_plane_z_and_xy(ee_mesh, *EE_RING_R, z_pick="max")
    z_grip, _ = _ring_plane_z_and_xy(mesh, *G2_RING_R, z_pick="area_peak")
    return float(z_ee - z_grip)


def _hole_fit_error_mm(mesh: trimesh.Trimesh, holes: np.ndarray) -> float:
    tree = cKDTree(mesh.vertices)
    return float(sum(tree.query(h)[0] for h in holes) * 1000)


def _bio_gripper_profiles() -> list:
    return [
        p
        for p in ROBOT_PROFILES.values()
        if p.supports_bio_gripper_g2 and p.bio_gripper_g2_visual_urdf
    ]


def _robot_ee_glb_path(profile) -> Path:
    return (
        profile.assets_dir
        / "meshes"
        / profile.mesh_variant
        / "visual_glb"
        / f"{profile.ee_link}.glb"
    )


def _canonical_ee_glb_path(ee_link: str) -> Path:
    """Canonical EE GLB for per-link static export (prefer xarm*_1305)."""
    matches = [p for p in _bio_gripper_profiles() if p.ee_link == ee_link]
    if not matches:
        raise KeyError(f"No Bio Gripper G2 profile with ee_link={ee_link}")
    matches.sort(key=lambda p: (0 if p.key.startswith("xarm") else 1, p.key))
    return _robot_ee_glb_path(matches[0])


def _ee_glb_path(ee_link: str) -> Path:
    return _canonical_ee_glb_path(ee_link)


def _urdf_origin_rt(elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if elem is None:
        return np.eye(3), np.zeros(3)
    xyz = np.fromstring(elem.get("xyz", "0 0 0"), sep=" ", dtype=float)
    roll, pitch, yaw = np.fromstring(elem.get("rpy", "0 0 0"), sep=" ", dtype=float)
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix(), xyz


def _urdf_compose(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    ra, ta = a
    rb, tb = b
    return ra @ rb, ta + ra @ tb


def _urdf_link_world_rt(root: ET.Element, link_name: str) -> tuple[np.ndarray, np.ndarray]:
    parent_by_child: dict[str, tuple[str, tuple[np.ndarray, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_by_child[child.get("link", "")] = (
            parent.get("link", ""),
            _urdf_origin_rt(joint.find("origin")),
        )
    chain: list[tuple[np.ndarray, np.ndarray]] = []
    current = link_name
    while current in parent_by_child:
        parent, origin_rt = parent_by_child[current]
        chain.append(origin_rt)
        current = parent
    out = (np.eye(3), np.zeros(3))
    for origin_rt in reversed(chain):
        out = _urdf_compose(out, origin_rt)
    return out


def _arm_ee_world_rt_at_zero(profile) -> np.ndarray:
    """EE link world rotation at URDF zero config (revolute joints at 0)."""
    urdf_path = profile.assets_dir / profile.visual_glb_urdf
    root = ET.parse(urdf_path).getroot()
    R, _ = _urdf_link_world_rt(root, profile.ee_link)
    return R


def _apply_transform(mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray, dz: float = 0.0) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = out.vertices @ R.T + t
    if abs(dz) <= 0.002:
        out.vertices[:, MOUNT_AXIS] += dz
    return out


def _partition_parts(parts: list[trimesh.Trimesh]) -> dict[str, list[trimesh.Trimesh]]:
    groups: dict[str, list[trimesh.Trimesh]] = {
        "base": [],
        "right_finger": [],
        "left_finger": [],
    }
    for part in parts:
        cy = float(part.centroid[1])
        if cy > Y_SPLIT_M:
            groups["right_finger"].append(part)
        elif cy < -Y_SPLIT_M:
            groups["left_finger"].append(part)
        else:
            groups["base"].append(part)
    if not groups["base"]:
        zmins = [float(p.vertices[:, 2].min()) for p in parts]
        shell_idx = int(np.argmin(zmins))
        groups["base"] = [parts[shell_idx]]
        finger_parts = [p for i, p in enumerate(parts) if i != shell_idx]
        ordered = sorted(finger_parts, key=lambda p: float(p.centroid[1]))
        mid = max(1, len(ordered) // 2)
        groups["left_finger"] = ordered[:mid]
        groups["right_finger"] = ordered[mid:]
    return groups


def _merge_group(group_parts: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    if not group_parts:
        return None
    return trimesh.util.concatenate(group_parts)


def _pick_pin_hole_transform(
    coarse: trimesh.Trimesh,
    pin_verts: np.ndarray,
    holes: np.ndarray,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, float]:
    """Try both hole orders; prefer low hole error, +X opening, low |Y| tilt.

    Holes are sorted by Y coordinate before enumeration so pin-to-hole
    correspondence is canonical across all EE links.  pin_verts are
    always [pos_y, neg_y] from _gripper_pin_points_stl; canonical holes
    are [neg_y, pos_y] — the two permutations then cover both matchings.
    """
    # Canonical hole ordering: sort by Y so permutations are deterministic
    # across EE link geometries (pin_verts from _gripper_pin_points_stl
    # already returns [pos_y, neg_y]).
    holes_canon = holes[np.argsort(holes[:, 1])]  # [neg_y, pos_y]
    best: tuple[float, trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, float, bool] | None = None
    for hole_order in (holes_canon, holes_canon[::-1]):
        R, t = _kabsch(pin_verts, hole_order)
        aligned = coarse.copy()
        aligned.vertices = aligned.vertices @ R.T + t
        # Check finger direction BEFORE correction so the 180° Z-flip
        # penalty can discriminate between naturally-+X and
        # corrected-to-+X pin-hole permutations (see worklog 2026-06-18).
        fd_before = _finger_direction(aligned)
        needs_rz_flip = fd_before[0] < 0.0
        aligned = _correct_finger_opening(aligned)
        hole_fit = _hole_fit_error_mm(aligned, hole_order)
        fd = _finger_direction(aligned)
        score = hole_fit
        if needs_rz_flip:
            score += 80.0
        score += abs(fd[1]) * 40.0
        score += (1.0 - float(fd @ FINGER_TARGET)) * 10.0
        # link5/link6 canonical: finger_dir Z < 0 in EE frame before URDF Rx(pi).
        # link7 previously picked the mirrored pin-hole solution (Z > 0), sinking
        # the gripper into the flange in Genesis static preview.
        if fd[2] > 0.0:
            score += 100.0
        if best is None or score < best[0]:
            best = (score, aligned, R, t, hole_order, hole_fit, needs_rz_flip)
    assert best is not None
    _, aligned, R, t, holes, hole_fit, needs_rz_flip = best
    # Propagate any 180° Z-flip correction to R, t so the per-part
    # movable GLBs built later use the corrected transform.
    if needs_rz_flip:
        rz = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
        R = rz @ R
        t = rz @ t
    return aligned, R, t, holes, hole_fit


def _attach_hole_fit_mm(
    R: np.ndarray, t: np.ndarray, pins_base: np.ndarray, holes: np.ndarray
) -> float:
    """Mean pin→hole distance (mm) after applying attach transform base→EE."""
    mapped = pins_base @ R.T + t
    return float(np.mean(np.linalg.norm(mapped - holes, axis=1)) * 1000)


def _score_attach_candidate(
    R: np.ndarray,
    t: np.ndarray,
    stl_pins: np.ndarray,
    holes: np.ndarray,
    stl_base: trimesh.Trimesh,
    ee_mesh: trimesh.Trimesh,
    R_world_ee: np.ndarray,
) -> tuple[float, float, float, bool, np.ndarray, np.ndarray, np.ndarray]:
    """Score an attach rigid map (base_link -> EE) for hole fit, ring coplanarity, finger +X."""
    holes_canon = holes[np.argsort(holes[:, 1])]
    base_in_ee = stl_base.copy()
    base_in_ee.vertices = base_in_ee.vertices @ R.T + t
    fd_before = _finger_direction(base_in_ee)
    needs_rz_flip = fd_before[0] < 0.0
    hole_fit = min(
        _attach_hole_fit_mm(R, t, stl_pins, holes_canon),
        _attach_hole_fit_mm(R, t, stl_pins, holes_canon[::-1]),
    )
    dz = _z_shift_to_flange(base_in_ee, ee_mesh)
    if abs(dz) <= 0.002:
        t = t.copy()
        t[MOUNT_AXIS] += dz
        base_in_ee.vertices[:, MOUNT_AXIS] += dz
        hole_fit = min(
            _attach_hole_fit_mm(R, t, stl_pins, holes_canon),
            _attach_hole_fit_mm(R, t, stl_pins, holes_canon[::-1]),
        )
    z_ee, _ = _ring_plane_z_and_xy(ee_mesh, *EE_RING_R, z_pick="max")
    z_g, _ = _ring_plane_z_and_xy(base_in_ee, *G2_RING_R, z_pick="area_peak")
    ring_gap = abs(z_ee - z_g) * 1000
    fd = _finger_direction(base_in_ee)
    finger_world = R_world_ee @ (R @ FINGER_TARGET)
    finger_world = finger_world / (np.linalg.norm(finger_world) + 1e-12)
    score = hole_fit
    if needs_rz_flip:
        score += 80.0
    score += abs(fd[1]) * 40.0
    score += (1.0 - float(fd @ FINGER_TARGET)) * 10.0
    score += (1.0 - float(finger_world[0])) * 60.0
    score += abs(float(finger_world[1])) * 30.0
    ring_penalty = max(0.0, ring_gap - 5.0) * 3.0
    if hole_fit >= 5.0:
        ring_penalty *= 0.2
    score += ring_penalty
    return score, hole_fit, ring_gap, needs_rz_flip, R, t, finger_world


def _static_visual_finger_world_ref(profile) -> np.ndarray | None:
    """Finger opening direction from the static combo visual mesh (user-verified reference)."""
    urdf_path = profile.assets_dir / profile.bio_gripper_g2_visual_urdf
    if not urdf_path.is_file():
        return None
    root = ET.parse(urdf_path).getroot()
    visual_link = "bio_gripper_g2_visual"
    mesh_ref = None
    visual_rt = (np.eye(3), np.zeros(3))
    for link in root.findall("link"):
        if link.get("name") != visual_link:
            continue
        visual = link.find("visual")
        if visual is None:
            return None
        mesh = visual.find("./geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            return None
        mesh_ref = mesh.get("filename")
        visual_rt = _urdf_origin_rt(visual.find("origin"))
        break
    if mesh_ref is None:
        return None
    mesh_path = (urdf_path.parent / mesh_ref).resolve()
    if not mesh_path.is_file():
        return None
    mesh = trimesh.load(mesh_path, force="mesh")
    R, t = _urdf_compose(_urdf_link_world_rt(root, visual_link), visual_rt)
    verts = mesh.vertices @ R.T + t
    fd = _finger_direction(trimesh.Trimesh(vertices=verts, faces=mesh.faces))
    return fd / (np.linalg.norm(fd) + 1e-12)


def _attach_candidate_sort_key(
    score: float,
    hole_fit: float,
    finger_world: np.ndarray,
    static_ref: np.ndarray | None,
    ring_gap: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Lower is better: world +X, low |world Y|, ring coplanarity, hole fit, static alignment, score."""
    fx = float(finger_world[0])
    fy = abs(float(finger_world[1]))
    lateral = fy if fy > 0.05 else 0.0
    angle = 0.0
    if static_ref is not None:
        angle = float(
            np.degrees(
                np.arccos(float(np.clip(np.dot(finger_world, static_ref), -1.0, 1.0)))
            )
        )
    ring_penalty = max(0.0, float(ring_gap) - 5.0)
    return (-round(fx, 2), lateral, ring_penalty, hole_fit, angle, score)


def _attach_origin_for_ee(
    stl_pins: np.ndarray,
    holes: np.ndarray,
    stl_base: trimesh.Trimesh,
    ee_mesh: trimesh.Trimesh,
    R_pin: np.ndarray,
    t_pin: np.ndarray,
    rot_g: np.ndarray,
    trans_g: np.ndarray,
    profile,
) -> dict:
    """Compute URDF ``bio_gripper_g2_attach`` origin for movable arm combos.

    Movable GLBs live in the canonical gripper ``base_link`` frame.  Pick the best of:

    1. Direct pin-hole Kabsch (``stl_pins`` -> EE holes), scored like static alignment.
    2. Static-visual equivalence: ``T_attach = Rx(pi) @ T_pin @ inv(T_gripper_frame)`` so
       the movable base matches the static overlay pose on the same EE flange.
    """
    holes_canon = holes[np.argsort(holes[:, 1])]
    rx_pi = Rotation.from_euler("x", np.pi, degrees=False).as_matrix()
    R_world_ee = _arm_ee_world_rt_at_zero(profile)
    static_ref = _static_visual_finger_world_ref(profile)
    best: tuple[
        tuple[float, float, float, float, float, float],
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
    ] | None = None

    def _consider(
        score: float,
        R: np.ndarray,
        t: np.ndarray,
        hole_fit: float,
        ring_gap: float,
        finger_world: np.ndarray,
    ) -> None:
        nonlocal best
        key = _attach_candidate_sort_key(score, hole_fit, finger_world, static_ref, ring_gap)
        if best is None or key < best[0]:
            best = (key, R, t, hole_fit, ring_gap, finger_world)

    for hole_order in (holes_canon, holes_canon[::-1]):
        R, t = _kabsch(stl_pins, hole_order)
        score, hole_fit, ring_gap, needs_rz, R, t, finger_world = _score_attach_candidate(
            R, t, stl_pins, holes, stl_base, ee_mesh, R_world_ee
        )
        if finger_world[0] < 0.0:
            rz = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
            score2, hole_fit2, ring_gap2, _, R2, t2, fw2 = _score_attach_candidate(
                rz @ R, rz @ t, stl_pins, holes, stl_base, ee_mesh, R_world_ee
            )
            if fw2[0] > finger_world[0]:
                score, hole_fit, ring_gap, R, t, finger_world = (
                    score2,
                    hole_fit2,
                    ring_gap2,
                    R2,
                    t2,
                    fw2,
                )
        elif needs_rz:
            rz = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
            score2, hole_fit2, ring_gap2, _, R2, t2, fw2 = _score_attach_candidate(
                rz @ R, rz @ t, stl_pins, holes, stl_base, ee_mesh, R_world_ee
            )
            if _attach_candidate_sort_key(
                score2, hole_fit2, fw2, static_ref, ring_gap2
            ) < _attach_candidate_sort_key(score, hole_fit, finger_world, static_ref, ring_gap):
                score, hole_fit, ring_gap, R, t, finger_world = (
                    score2,
                    hole_fit2,
                    ring_gap2,
                    R2,
                    t2,
                    fw2,
                )
        _consider(score, R, t, hole_fit, ring_gap, finger_world)

    R_static = rx_pi @ R_pin @ rot_g.T
    t_static = t_pin @ rx_pi.T - trans_g @ rot_g @ R_pin.T @ rx_pi.T
    score, hole_fit, ring_gap, needs_rz, R, t, finger_world = _score_attach_candidate(
        R_static, t_static, stl_pins, holes, stl_base, ee_mesh, R_world_ee
    )
    if needs_rz:
        rz = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
        score2, hole_fit2, ring_gap2, _, R2, t2, fw2 = _score_attach_candidate(
            rz @ R_static, rz @ t_static, stl_pins, holes, stl_base, ee_mesh, R_world_ee
        )
        if _attach_candidate_sort_key(
            score2, hole_fit2, fw2, static_ref, ring_gap2
        ) < _attach_candidate_sort_key(score, hole_fit, finger_world, static_ref, ring_gap):
            score, hole_fit, ring_gap, R, t, finger_world = (
                score2,
                hole_fit2,
                ring_gap2,
                R2,
                t2,
                fw2,
            )
    else:
        R, t = R_static, t_static
    _consider(score, R, t, hole_fit, ring_gap, finger_world)

    assert best is not None
    _, R, t, hole_fit, ring_gap, finger_world = best

    base_in_ee = stl_base.copy()
    base_in_ee.vertices = base_in_ee.vertices @ R.T + t
    rpy = Rotation.from_matrix(R).as_euler("xyz")
    fd = R @ FINGER_TARGET
    fd = fd / (np.linalg.norm(fd) + 1e-12)
    static_angle_deg = None
    if static_ref is not None:
        static_angle_deg = round(
            float(
                np.degrees(
                    np.arccos(float(np.clip(np.dot(finger_world, static_ref), -1.0, 1.0)))
                )
            ),
            2,
        )
    return {
        "attach_xyz": [round(float(x), 6) for x in t],
        "attach_rpy": [round(float(x), 8) for x in rpy],
        "attach_xyz_str": " ".join(f"{x:.6f}" for x in t),
        "attach_rpy_str": " ".join(f"{x:.8f}" for x in rpy),
        "attach_hole_fit_mm": round(hole_fit, 2),
        "attach_ring_gap_mm": round(ring_gap, 3),
        "attach_finger_dir": [round(float(x), 4) for x in fd],
        "attach_finger_world_dir": [round(float(x), 4) for x in finger_world],
        "attach_finger_dot_base_x": round(float(finger_world[0]), 4),
        "attach_static_angle_deg": static_angle_deg,
    }


def relocalize_for_ee_link(
    groups: dict[str, list[trimesh.Trimesh]],
    material_parts: list[tuple[trimesh.Trimesh, float]],
    jaw_labels: list[str | None],
    finger_yshift: dict[str, float],
    stl_base: trimesh.Trimesh,
    stl_pins: np.ndarray,
    ee_link: str,
    ee_glb: Path,
) -> tuple[trimesh.Trimesh, dict[str, trimesh.Trimesh | None], dict]:
    ee_mesh = trimesh.load(ee_glb, force="mesh")
    holes = _arm_locating_holes(ee_mesh)

    merged_coarse = _merge_group(
        groups["base"] + groups["left_finger"] + groups["right_finger"]
    )
    assert merged_coarse is not None

    coarse = _cad_to_stl_coarse(merged_coarse, stl_base)
    pin_verts = _pin_vertices_on_mesh(coarse, stl_pins)

    if np.linalg.norm(pin_verts[0] - pin_verts[1]) < 0.005:
        raise RuntimeError(f"Degenerate pin pair on {ee_link}; pin verts collapsed")

    aligned_flange, R, t, holes, hole_fit = _pick_pin_hole_transform(coarse, pin_verts, holes)

    # link7 EE flange hole layout admits a mirrored pin-hole solution (finger_dir
    # Z > 0) that scores well on TCP alignment but sinks the static GLB into the
    # arm flange after URDF Rx(pi).  Force the link5/6 hemisphere (finger_dir Z < 0).
    fd = _finger_direction(aligned_flange)
    if fd[2] > 0.0:
      rx_pi = Rotation.from_euler("x", np.pi).as_matrix()
      R = R @ rx_pi.T
      aligned_flange = coarse.copy()
      aligned_flange.vertices = coarse.vertices @ R.T + t
      aligned_flange = _correct_finger_opening(aligned_flange)
      hole_fit = _hole_fit_error_mm(aligned_flange, holes)

    dz = _z_shift_to_flange(aligned_flange, ee_mesh)
    if abs(dz) <= 0.002:
        aligned_flange.vertices[:, MOUNT_AXIS] += dz
        aligned_flange = _correct_finger_opening(aligned_flange)

    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    rot_g, trans_g = _gripper_frame_rt(groups, merged_coarse, stl_base)

    # Movable base + fingers are built in the canonical gripper base_link frame
    # (link_base.stl frame), NOT the per-EE-link pin-hole frame.  Finger stroke stays
    # EE-independent; per-EE attach origin (below) maps those meshes onto the arm flange.
    aligned_base_plastic, aligned_base_metal = _movable_base_gripper_frame(
        material_parts, jaw_labels, rot_g, trans_g, center_offset
    )
    left_finger_out, right_finger_out = _finger_meshes_gripper_frame(
        material_parts, jaw_labels, finger_yshift, rot_g, trans_g, center_offset
    )
    finger_method = "gripper_base_link_frame"
    static_plastic, static_metal = _static_ee_material_meshes(
        material_parts, merged_coarse, stl_base, R, t, dz
    )

    aligned = aligned_flange
    # frame_rx_pi is intentionally always False: the combo URDF mount
    # rpy="3.14159265 0 0" (Rx(pi)) already handles the EE-flange-to-gripper
    # orientation at the kinematics level.  Applying an additional mesh-level
    # Rx(pi) would double-rotate the base and invert finger Y, swapping the
    # left/right finger visuals (see plan for link7 travel bug analysis).
    frame_rx_pi = False

    aligned_base = trimesh.util.concatenate(
        [m for m in (aligned_base_plastic, aligned_base_metal) if m is not None]
    )

    movable: dict[str, trimesh.Trimesh | None] = {
        "link_base_plastic": aligned_base_plastic,
        "link_base_metal": aligned_base_metal,
        "left_finger": left_finger_out,
        "right_finger": right_finger_out,
    }

    base_q = _mesh_quality_metrics(
        trimesh.util.concatenate(
            [m for m in (aligned_base_plastic, aligned_base_metal) if m is not None]
        )
    ) if (aligned_base_plastic is not None or aligned_base_metal is not None) else {}
    static_q = _mesh_quality_metrics(aligned)
    left_q = _mesh_quality_metrics(left_finger_out) if left_finger_out is not None else {}
    right_q = _mesh_quality_metrics(right_finger_out) if right_finger_out is not None else {}

    fd = _finger_direction(aligned)
    z_ee, xy_ee = _ring_plane_z_and_xy(ee_mesh, *EE_RING_R, z_pick="max")
    z_g, xy_g = _ring_plane_z_and_xy(aligned_flange, *G2_RING_R, z_pick="area_peak")
    entry: dict = {
        "ee_link": ee_link,
        "output": f"bio_gripper_g2_visual_{ee_link}.glb",
        "verts": int(aligned.vertices.shape[0]),
        "hole_fit_mm": round(hole_fit, 2),
        "finger_dir": [round(float(x), 4) for x in fd],
        "finger_dot_tcp": round(float(fd @ FINGER_TARGET), 4),
        "finger_y_abs": round(float(abs(fd[1])), 4),
        "ring_gap_mm": round(abs(z_ee - z_g) * 1000, 3),
        "xy_ring_gap_mm": [round(float((xy_ee - xy_g)[0]) * 1000, 2), round(float((xy_ee - xy_g)[1]) * 1000, 2)],
        "stl_surface_mm": round(mean_surface_distance(aligned, stl_base) * 1000, 2),
        "pin_verts_mm": (pin_verts @ R.T + t * 1000).round(2).tolist(),
        "holes_mm": (holes * 1000).round(2).tolist(),
        "pin_separation_mm": round(float(np.linalg.norm(pin_verts[0] - pin_verts[1]) * 1000), 2),
        "z_shift_skipped_mm": round(float(dz * 1000), 3),
        "partition_counts": {k: len(v) for k, v in groups.items()},
        "finger_method": finger_method,
        "frame_rx_pi": frame_rx_pi,
        "static_open_edge_ratio": static_q.get("open_edge_ratio"),
        "base_open_edge_ratio": base_q.get("open_edge_ratio"),
        "base_extents_mm": [round(float(x) * 1000, 2) for x in aligned_base.extents]
        if aligned_base is not None
        else None,
        "left_open_edge_ratio": left_q.get("open_edge_ratio"),
        "right_open_edge_ratio": right_q.get("open_edge_ratio"),
    }
    if aligned_base_plastic is not None or aligned_base_metal is not None:
        base_mesh = trimesh.util.concatenate(
            [m for m in (aligned_base_plastic, aligned_base_metal) if m is not None]
        )
        entry["base_verts"] = int(base_mesh.vertices.shape[0])
    if movable["left_finger"] is not None:
        entry["left_finger_verts"] = int(movable["left_finger"].vertices.shape[0])
    if movable["right_finger"] is not None:
        entry["right_finger_verts"] = int(movable["right_finger"].vertices.shape[0])
    return aligned, movable, entry, static_plastic, static_metal


def relocalize_bio_gripper_g2(dry_run: bool = False) -> list[dict]:
    src = SRC_GLB if SRC_GLB.is_file() else FALLBACK_SRC
    if not src.is_file():
        raise FileNotFoundError(f"Missing Bio Gripper G2 source GLB: {SRC_GLB} or {FALLBACK_SRC}")
    if not STL_BASE.is_file():
        raise FileNotFoundError(STL_BASE)

    stl_base = trimesh.load(STL_BASE, force="mesh")
    stl_pins = _gripper_pin_points_stl(stl_base)
    parts = bake_glb_genesis_parts(src)
    material_parts = _bake_bio_material_parts(src)
    groups = _partition_parts(parts)

    ee_links = sorted({p.ee_link for p in _bio_gripper_profiles()})

    report: list[dict] = []

    merged_coarse = _merge_group(
        groups["base"] + groups["left_finger"] + groups["right_finger"]
    )
    assert merged_coarse is not None

    jaw_labels, finger_yshift = _jaw_labels_for_material_parts(
        parts, material_parts, groups, merged_coarse, stl_base
    )
    n_left = sum(1 for l in jaw_labels if l == "left")
    n_right = sum(1 for l in jaw_labels if l == "right")
    print(
        f"jaw classification: {n_left} left + {n_right} right jaw parts, "
        f"{len(jaw_labels) - n_left - n_right} static base parts (of {len(jaw_labels)}); "
        f"finger_yshift mm: left={finger_yshift['left']*1000:.1f} right={finger_yshift['right']*1000:.1f}"
    )

    coarse = _cad_to_stl_coarse(merged_coarse, stl_base)
    pin_verts = _pin_vertices_on_mesh(coarse, stl_pins)
    rot_g, trans_g = _gripper_frame_rt(groups, merged_coarse, stl_base)

    for ee_link in ee_links:
        ee_glb = _canonical_ee_glb_path(ee_link)
        if not ee_glb.is_file():
            raise FileNotFoundError(f"Missing EE GLB for {ee_link}: {ee_glb}")
        aligned, movable, entry, static_plastic, static_metal = relocalize_for_ee_link(
            groups, material_parts, jaw_labels, finger_yshift, stl_base, stl_pins, ee_link, ee_glb,
        )
        report.append(entry)
        print(
            f"{entry['output']}: hole_fit={entry['hole_fit_mm']:.1f}mm, "
            f"ring_gap={entry['ring_gap_mm']:.3f}mm, "
            f"pin_sep={entry['pin_separation_mm']:.1f}mm, "
            f"static_oer={entry.get('static_open_edge_ratio')}, "
            f"base_oer={entry.get('base_open_edge_ratio')}"
        )
        if not dry_run:
            VISUAL_DIR.mkdir(parents=True, exist_ok=True)
            out = VISUAL_DIR / entry["output"]
            _export_link_base_glb(static_plastic, static_metal, out)

            ee_base_dir = GLB_OUT_DIR / ee_link
            ee_base_dir.mkdir(parents=True, exist_ok=True)
            _export_link_base_glb(
                movable["link_base_plastic"],
                movable["link_base_metal"],
                ee_base_dir / BIO_GRIPPER_G2_BASE_GLB,
            )
            for name, mesh in (
                ("left_finger", movable["left_finger"]),
                ("right_finger", movable["right_finger"]),
            ):
                if mesh is None:
                    continue
                glb_name = BIO_LEFT_FINGER_GLB if name == "left_finger" else BIO_RIGHT_FINGER_GLB
                export_opaque_doublesided_glb([mesh], ee_base_dir / glb_name, ARM_EE_METAL)

    for profile in _bio_gripper_profiles():
        ee_glb = _robot_ee_glb_path(profile)
        ee_mesh = trimesh.load(ee_glb, force="mesh")
        holes = _arm_locating_holes(ee_mesh)
        canon_glb = _canonical_ee_glb_path(profile.ee_link)
        if canon_glb == ee_glb:
            canon_holes = holes
        else:
            canon_holes = _arm_locating_holes(trimesh.load(canon_glb, force="mesh"))
        # Static-visual equivalence must use the same canonical EE pin-hole frame as the
        # exported bio_gripper_g2_visual_{ee_link}.glb (xarm*_1305), not the robot-specific
        # EE mesh (uf850 link6 differs from xarm6 link6).
        _, R_pin, t_pin, _, _ = _pick_pin_hole_transform(coarse, pin_verts, canon_holes)
        attach_entry = {
            "robot_key": profile.key,
            "ee_link": profile.ee_link,
            **_attach_origin_for_ee(
                stl_pins, holes, stl_base, ee_mesh, R_pin, t_pin, rot_g, trans_g, profile
            ),
        }
        report.append(attach_entry)
        print(
            f"{profile.key} attach: hole_fit={attach_entry['attach_hole_fit_mm']:.1f}mm, "
            f"ring_gap={attach_entry['attach_ring_gap_mm']:.3f}mm, "
            f"finger_dot_base_x={attach_entry.get('attach_finger_dot_base_x')}"
        )

    if not dry_run:
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {METRICS_PATH}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Relocalize Bio Gripper G2 GLB via pin–hole flange alignment")
    parser.add_argument("--dry-run", action="store_true", help="Report metrics without writing files")
    args = parser.parse_args()

    _init_genesis()
    relocalize_bio_gripper_g2(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
