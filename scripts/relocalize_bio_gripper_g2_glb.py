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
    _rigid_align_mesh_to_ref,
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
_RX_PI = Rotation.from_euler("x", np.pi, degrees=False).as_matrix()


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


def _movable_base_ee_material_meshes(
    material_parts: list[tuple[trimesh.Trimesh, float]],
    jaw_labels: list[str | None],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
    R: np.ndarray,
    t: np.ndarray,
    dz: float,
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Static base in EE link frame: everything except the two moving jaw nodes.

    Keeps the white-plastic shell plus all non-jaw metal (mount flange, housing rail,
    body screws, UFACTORY nameplate) so only the jaws move with the finger joints.
    """
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    plastic_parts: list[trimesh.Trimesh] = []
    metal_parts: list[trimesh.Trimesh] = []
    for (part, metallic), label in zip(material_parts, jaw_labels):
        if label is not None:
            continue
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

    def gripping_inner_edge(verts: np.ndarray, side: str) -> float:
        """Center-facing Y of the gripping blade (front half = high X)."""
        front = verts[verts[:, 0] >= np.percentile(verts[:, 0], 60)]
        # left jaw sits at -Y, its inner face is the max Y; right jaw the min Y
        return float(np.percentile(front[:, 1], 97 if side == "left" else 3))

    # collision-STL gripping faces define the joint-0 (closed) target plane
    stl_inner = {
        "left": gripping_inner_edge(left_pts, "left"),
        "right": gripping_inner_edge(right_pts, "right"),
    }

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
            # register the jaw's gripping face onto the collision STL gripping face,
            # so the visual is closed/flush at joint=0 and the [0,limit] stroke
            # reproduces the real opening width
            finger_yshift[side] = stl_inner[side] - gripping_inner_edge(verts, side)
        else:
            node_labels.append(None)
        node_trees.append(cKDTree(verts))

    labels: list[str | None] = []
    for part, _metallic in material_parts:
        verts = to_base_frame(part)
        node_idx = int(np.argmin([float(np.mean(t.query(verts)[0])) for t in node_trees]))
        labels.append(node_labels[node_idx])
    return labels, finger_yshift


def _finger_meshes_from_static_material(
    material_parts: list[tuple[trimesh.Trimesh, float]],
    jaw_labels: list[str | None],
    finger_yshift: dict[str, float],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
    R: np.ndarray,
    t: np.ndarray,
    dz: float,
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """Moving jaw geometry only, mapped into the URDF finger link frames.

    Only parts belonging to a jaw node (per _jaw_labels_for_material_parts) are kept;
    left/right is resolved by the aligned-frame Y sign (consistent with base frame).
    Each side is then shifted in Y by finger_yshift[side] so the jaw registers onto
    its joint-zero collision STL (closed pose), keeping the fingers flush with the
    body instead of floating at the CAD's native open opening.
    """
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    left_parts: list[trimesh.Trimesh] = []
    right_parts: list[trimesh.Trimesh] = []
    for (part, _metallic), label in zip(material_parts, jaw_labels):
        if label is None:
            continue
        coarse = _part_to_assembly_coarse(part, center_offset)
        aligned = _apply_transform(coarse, R, t, dz)
        ay = float(aligned.centroid[1])
        if ay < 0.0:
            left_parts.append(aligned)
        else:
            right_parts.append(aligned)
    left = trimesh.util.concatenate(left_parts) if left_parts else None
    right = trimesh.util.concatenate(right_parts) if right_parts else None
    if left is not None:
        left.vertices[:, 1] += finger_yshift.get("left", 0.0)
        left = _subdivide_visual_mesh(_finger_mesh_in_link_frame(left), FINGER_SUBDIVIDE_ITERS)
    if right is not None:
        right.vertices[:, 1] += finger_yshift.get("right", 0.0)
        right = _subdivide_visual_mesh(_finger_mesh_in_link_frame(right), FINGER_SUBDIVIDE_ITERS)
    return left, right


def _finger_display_centroid(mesh_link: trimesh.Trimesh) -> np.ndarray:
    return mesh_link.vertices + FINGER_JOINT_ORIGIN


def _rx_pi_base_frame(mesh: trimesh.Trimesh | None) -> trimesh.Trimesh | None:
    if mesh is None:
        return None
    out = mesh.copy()
    out.vertices = out.vertices @ _RX_PI.T
    return out


def _rx_pi_finger_link_frame(mesh: trimesh.Trimesh | None) -> trimesh.Trimesh | None:
    if mesh is None:
        return None
    out = mesh.copy()
    out.vertices = (out.vertices + FINGER_JOINT_ORIGIN) @ _RX_PI.T - FINGER_JOINT_ORIGIN
    return out


def _flange_above_plastic(
    plastic: trimesh.Trimesh | None,
    metal: trimesh.Trimesh | None,
) -> bool:
    if plastic is None or metal is None:
        return True
    return float(metal.vertices[:, 2].max()) > float(plastic.vertices[:, 2].max()) + 1e-4


def _movable_base_in_gripper_frame(
    groups: dict[str, list[trimesh.Trimesh]],
    merged_coarse: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Movable base GLB in bio_gripper_g2_base_link frame (matches finger STL kinematics)."""
    merged_base = _merge_group(groups["base"])
    assert merged_base is not None
    center_offset = _assembly_center_offset(merged_coarse, stl_base)
    base_local = _part_to_assembly_coarse(merged_base, center_offset)
    aligned, _ = _rigid_align_mesh_to_ref(base_local, stl_base)
    return aligned


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


def _ee_glb_path(ee_link: str) -> Path:
    for profile in ROBOT_PROFILES.values():
        if profile.ee_link == ee_link:
            return profile.assets_dir / "meshes" / profile.mesh_variant / "visual_glb" / f"{ee_link}.glb"
    raise KeyError(f"No robot profile with ee_link={ee_link}")


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
    """Try both hole orders; prefer low hole error, +X opening, low |Y| tilt."""
    best: tuple[float, trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, float, bool] | None = None
    for hole_order in (holes, holes[::-1]):
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

    dz = _z_shift_to_flange(aligned_flange, ee_mesh)
    if abs(dz) <= 0.002:
        aligned_flange.vertices[:, MOUNT_AXIS] += dz
        aligned_flange = _correct_finger_opening(aligned_flange)

    center_offset = _assembly_center_offset(merged_coarse, stl_base)

    aligned_base_plastic, aligned_base_metal = _movable_base_ee_material_meshes(
        material_parts, jaw_labels, merged_coarse, stl_base, R, t, dz
    )
    left_finger_out, right_finger_out = _finger_meshes_from_static_material(
        material_parts, jaw_labels, finger_yshift, merged_coarse, stl_base, R, t, dz
    )
    finger_method = "ee_align_link_frame"
    static_plastic, static_metal = _static_ee_material_meshes(
        material_parts, merged_coarse, stl_base, R, t, dz
    )

    aligned = aligned_flange
    frame_rx_pi = False
    if not _flange_above_plastic(aligned_base_plastic, aligned_base_metal):
        frame_rx_pi = True
        aligned_base_plastic = _rx_pi_base_frame(aligned_base_plastic)
        aligned_base_metal = _rx_pi_base_frame(aligned_base_metal)
        left_finger_out = _rx_pi_finger_link_frame(left_finger_out)
        right_finger_out = _rx_pi_finger_link_frame(right_finger_out)
        static_plastic = _rx_pi_base_frame(static_plastic)
        static_metal = _rx_pi_base_frame(static_metal)
        aligned = _rx_pi_base_frame(aligned_flange)

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

    ee_links = sorted(
        {
            p.ee_link
            for p in ROBOT_PROFILES.values()
            if p.supports_bio_gripper_g2 and p.bio_gripper_g2_visual_urdf
        }
    )

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

    for ee_link in ee_links:
        ee_glb = _ee_glb_path(ee_link)
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
