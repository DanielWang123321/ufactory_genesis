#!/usr/bin/env python3
"""
Relocalize Bio Gripper G2 GLB for static visual combo URDFs.

Per EE link (link5 / link6 / link7):
  1. Genesis bake source CAD GLB (×0.1 m)
  2. Coarse CAD→STL (z +270°, centroid match link_base.stl)
  3. Pin–hole Kabsch: gripper locating pins ↔ arm flange locating holes
  4. Z-only outer-ring coplanar refine (same annulus as G2 gripper)
  5. Export single merged mesh (one GLB root) in that EE link frame
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
    _raw_material_pbr,
    bake_glb_genesis_parts,
    export_opaque_doublesided_glb,
    mean_surface_distance,
)
from relocalize_gripper_glb import (
    G2_RING_R,
    LINK6_RING_R,
    MOUNT_AXIS,
    _ring_plane_z_and_xy,
)
from ufactory.paths import BIO_GRIPPER_ASSETS
from ufactory.robot_registry import ROBOT_PROFILES

VISUAL_DIR = BIO_GRIPPER_ASSETS / "meshes" / "visual"
STL_BASE = VISUAL_DIR / "link_base.stl"
SRC_GLB = VISUAL_DIR / "visual_glb_src" / "bio_gripper_g2.glb"
FALLBACK_SRC = VISUAL_DIR / "bio_gripper_g2.glb"
METRICS_PATH = VISUAL_DIR / "relocalize_metrics.json"

CAD_TO_STL_ROT_Z_DEG = 270.0
PIN_ANNULUS_R = (0.020, 0.028)
PIN_SEARCH_RADIUS_M = 0.015
# bio_gripper.urdf TCP direction in gripper base frame (+X with pitch)
FINGER_TARGET = np.array([0.135, 0.0, 0.055], dtype=np.float64)
FINGER_TARGET /= np.linalg.norm(FINGER_TARGET)


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
    flange = af & (r >= LINK6_RING_R[0]) & (r <= LINK6_RING_R[1])
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
    Rz = Rotation.from_euler("z", CAD_TO_STL_ROT_Z_DEG, degrees=True).as_matrix()
    out.vertices = out.vertices @ Rz.T
    out.vertices += stl_base.centroid - out.centroid
    return out


def _z_shift_to_flange(mesh: trimesh.Trimesh, ee_mesh: trimesh.Trimesh) -> float:
    z_ee, _ = _ring_plane_z_and_xy(ee_mesh, *LINK6_RING_R, z_pick="max")
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


def _pick_pin_hole_transform(
    coarse: trimesh.Trimesh,
    pin_verts: np.ndarray,
    holes: np.ndarray,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, float]:
    """Try both hole orders; prefer low hole error, +X opening, low |Y| tilt."""
    best: tuple[float, trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for hole_order in (holes, holes[::-1]):
        R, t = _kabsch(pin_verts, hole_order)
        aligned = coarse.copy()
        aligned.vertices = aligned.vertices @ R.T + t
        aligned = _correct_finger_opening(aligned)
        hole_fit = _hole_fit_error_mm(aligned, hole_order)
        fd = _finger_direction(aligned)
        score = hole_fit
        if fd[0] < 0.3:
            score += 80.0
        score += abs(fd[1]) * 40.0
        score += (1.0 - float(fd @ FINGER_TARGET)) * 10.0
        if best is None or score < best[0]:
            best = (score, aligned, R, t, hole_order, hole_fit)
    assert best is not None
    _, aligned, R, t, holes, hole_fit = best
    return aligned, R, t, holes, hole_fit


def relocalize_for_ee_link(
    merged: trimesh.Trimesh,
    stl_base: trimesh.Trimesh,
    stl_pins: np.ndarray,
    ee_link: str,
    ee_glb: Path,
) -> tuple[trimesh.Trimesh, dict]:
    ee_mesh = trimesh.load(ee_glb, force="mesh")
    holes = _arm_locating_holes(ee_mesh)

    coarse = _cad_to_stl_coarse(merged, stl_base)
    pin_verts = _pin_vertices_on_mesh(coarse, stl_pins)

    if np.linalg.norm(pin_verts[0] - pin_verts[1]) < 0.005:
        raise RuntimeError(f"Degenerate pin pair on {ee_link}; pin verts collapsed")

    aligned, R, t, holes, hole_fit = _pick_pin_hole_transform(coarse, pin_verts, holes)

    # Pin–hole Kabsch already fixes flange Z; ring-based z-shift can mis-fire on link7.
    dz = _z_shift_to_flange(aligned, ee_mesh)
    if abs(dz) <= 0.002:
        aligned.vertices[:, MOUNT_AXIS] += dz
        aligned = _correct_finger_opening(aligned)

    fd = _finger_direction(aligned)
    z_ee, xy_ee = _ring_plane_z_and_xy(ee_mesh, *LINK6_RING_R, z_pick="max")
    z_g, xy_g = _ring_plane_z_and_xy(aligned, *G2_RING_R, z_pick="area_peak")
    entry = {
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
    }
    return aligned, entry


def relocalize_bio_gripper(dry_run: bool = False) -> list[dict]:
    src = SRC_GLB if SRC_GLB.is_file() else FALLBACK_SRC
    if not src.is_file():
        raise FileNotFoundError(f"Missing bio gripper source GLB: {SRC_GLB} or {FALLBACK_SRC}")
    if not STL_BASE.is_file():
        raise FileNotFoundError(STL_BASE)

    stl_base = trimesh.load(STL_BASE, force="mesh")
    stl_pins = _gripper_pin_points_stl(stl_base)
    parts = bake_glb_genesis_parts(src)
    merged = trimesh.util.concatenate(parts)

    ee_links = sorted(
        {
            p.ee_link
            for p in ROBOT_PROFILES.values()
            if p.supports_bio_gripper_g2 and p.bio_gripper_g2_visual_urdf
        }
    )

    report: list[dict] = []
    for ee_link in ee_links:
        ee_glb = _ee_glb_path(ee_link)
        if not ee_glb.is_file():
            raise FileNotFoundError(f"Missing EE GLB for {ee_link}: {ee_glb}")
        aligned, entry = relocalize_for_ee_link(merged, stl_base, stl_pins, ee_link, ee_glb)
        report.append(entry)
        print(
            f"{entry['output']}: hole_fit={entry['hole_fit_mm']:.1f}mm, "
            f"ring_gap={entry['ring_gap_mm']:.3f}mm, "
            f"pin_sep={entry['pin_separation_mm']:.1f}mm"
        )
        if not dry_run:
            VISUAL_DIR.mkdir(parents=True, exist_ok=True)
            out = VISUAL_DIR / entry["output"]
            export_opaque_doublesided_glb([aligned], out, _raw_material_pbr(src))

    if not dry_run:
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {METRICS_PATH}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Relocalize Bio Gripper G2 GLB via pin–hole flange alignment")
    parser.add_argument("--dry-run", action="store_true", help="Report metrics without writing files")
    args = parser.parse_args()

    _init_genesis()
    relocalize_bio_gripper(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
