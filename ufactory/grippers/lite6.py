"""Lite6 reversed parallel gripper command-space conversions.

Unlike Gripper G2 (angled linkage, ``drive_joint`` in radians), the Lite6
gripper is a simple parallel-jaw mechanism: ``finger_joint1`` is a prismatic
joint (metres) with ``finger_joint2`` mirroring it via a ``mimic`` constraint.
The installed real gripper is reversed and has a 20-38 mm physical two-finger
gap. The trajectory sim models that as a fixed 20 mm closed offset plus the
vendor 0-8.9 mm per-finger URDF travel.

Real hardware note: the xArm Python SDK exposes this gripper only as two
digital-IO commands (``open_lite6_gripper`` / ``close_lite6_gripper``), not a
continuous position API. The gap-based conversions below are for the Genesis
sim/mirror path; the real executor maps any gripper segment to the nearer of
open/close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LITE6_GRIPPER_FINGER_TRAVEL_M = 0.0089
LITE6_GRIPPER_CLOSED_GAP_M = 0.020
LITE6_GRIPPER_OPEN_GAP_M = 0.038

LITE6_GRIPPER_SIM_CLOSED_DRIVE = 0.0
LITE6_GRIPPER_SIM_OPEN_DRIVE = LITE6_GRIPPER_FINGER_TRAVEL_M

# Legacy binary-close offset. The default sim grasp now targets the object width
# and uses a geometric weld to model the real firmware stopping on contact.
LITE6_GRIPPER_GAP_CALIBRATION_OFFSET_M = 0.010

_FINGER_MESH_DIR = Path(__file__).resolve().parents[2] / "assets" / "urdf" / "lite6_gripper" / "meshes" / "collision"
_FINGER1_VISUAL_ORIGIN_M = np.asarray((0.0, 0.010, 0.0), dtype=np.float64)
_FINGER2_VISUAL_ORIGIN_M = np.asarray((0.0, -0.010, 0.0), dtype=np.float64)
_FINGER1_VERTS: np.ndarray | None = None
_FINGER2_VERTS: np.ndarray | None = None


def clamp_lite6_gripper_gap_m(gap_m: float) -> float:
    """Clamp a two-finger gap in metres to the Lite6 gripper's physical range."""
    return max(LITE6_GRIPPER_CLOSED_GAP_M, min(LITE6_GRIPPER_OPEN_GAP_M, float(gap_m)))


def lite6_gripper_gap_m_to_sim_drive(gap_m: float) -> float:
    """Map a desired physical gap in metres to the Genesis ``finger_joint1`` position."""
    span = LITE6_GRIPPER_OPEN_GAP_M - LITE6_GRIPPER_CLOSED_GAP_M
    if span <= 0.0:
        return LITE6_GRIPPER_SIM_CLOSED_DRIVE
    open_fraction = (clamp_lite6_gripper_gap_m(gap_m) - LITE6_GRIPPER_CLOSED_GAP_M) / span
    return LITE6_GRIPPER_SIM_CLOSED_DRIVE + open_fraction * (
        LITE6_GRIPPER_SIM_OPEN_DRIVE - LITE6_GRIPPER_SIM_CLOSED_DRIVE
    )


def lite6_gripper_sim_drive_to_gap_m(drive: float) -> float:
    """Map Genesis ``finger_joint1`` position to physical two-finger gap in metres."""
    clipped = max(LITE6_GRIPPER_SIM_CLOSED_DRIVE, min(LITE6_GRIPPER_SIM_OPEN_DRIVE, float(drive)))
    span = LITE6_GRIPPER_SIM_OPEN_DRIVE - LITE6_GRIPPER_SIM_CLOSED_DRIVE
    if span <= 0.0:
        return LITE6_GRIPPER_CLOSED_GAP_M
    open_fraction = (clipped - LITE6_GRIPPER_SIM_CLOSED_DRIVE) / span
    return LITE6_GRIPPER_CLOSED_GAP_M + open_fraction * (
        LITE6_GRIPPER_OPEN_GAP_M - LITE6_GRIPPER_CLOSED_GAP_M
    )


def _finger_mesh_vertices() -> tuple[np.ndarray, np.ndarray]:
    global _FINGER1_VERTS, _FINGER2_VERTS
    if _FINGER1_VERTS is None or _FINGER2_VERTS is None:
        import trimesh

        _FINGER1_VERTS = np.asarray(
            trimesh.load(_FINGER_MESH_DIR / "finger1.stl", force="mesh").vertices,
            dtype=np.float64,
        ) + _FINGER1_VISUAL_ORIGIN_M
        _FINGER2_VERTS = np.asarray(
            trimesh.load(_FINGER_MESH_DIR / "finger2.stl", force="mesh").vertices,
            dtype=np.float64,
        ) + _FINGER2_VISUAL_ORIGIN_M
    return _FINGER1_VERTS, _FINGER2_VERTS


def _link_world_vertices(link, mesh_vertices: np.ndarray) -> np.ndarray:
    pos = link.get_pos()[0].detach().cpu().numpy()
    quat = link.get_quat()[0].detach().cpu().numpy()
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return mesh_vertices @ rot.T + pos


def _lite6_glb_inner_contact_geometry(ctx, obj_pos=None) -> tuple[float, float, float, float]:
    if obj_pos is None:
        obj_pos = ctx.obj.get_pos()[0].detach().cpu().numpy()
    else:
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
    half_y = float(ctx.obj_size[1]) / 2.0
    half_z = float(ctx.obj_size[2]) / 2.0
    z0, z1 = obj_pos[2] - half_z, obj_pos[2] + half_z
    oy0, oy1 = obj_pos[1] - half_y, obj_pos[1] + half_y
    finger1, finger2 = _finger_mesh_vertices()
    w1 = _link_world_vertices(ctx.left_finger, finger1)
    w2 = _link_world_vertices(ctx.right_finger, finger2)
    band1 = w1[(w1[:, 2] >= z0) & (w1[:, 2] <= z1)]
    band2 = w2[(w2[:, 2] >= z0) & (w2[:, 2] <= z1)]
    if len(band1) == 0 or len(band2) == 0:
        return 0.0, 0.0, 0.0, 0.0
    if float(band1[:, 1].mean()) <= float(band2[:, 1].mean()):
        neg_band, pos_band = band1, band2
    else:
        neg_band, pos_band = band2, band1
    inner_neg_y = float(neg_band[:, 1].max())
    inner_pos_y = float(pos_band[:, 1].min())
    gap_neg = float(oy0 - inner_neg_y)
    gap_pos = float(inner_pos_y - oy1)
    return inner_neg_y, inner_pos_y, gap_neg, gap_pos


def measure_lite6_glb_side_clearance_m(ctx, obj_pos=None) -> tuple[float, float]:
    """Return per-face GLB pad-to-block side clearance (m) at the object mid-height."""
    _, _, gap_neg, gap_pos = _lite6_glb_inner_contact_geometry(ctx, obj_pos=obj_pos)
    return gap_neg, gap_pos


def snap_object_to_lite6_glb_grasp(ctx) -> float:
    """Center the block between measured GLB inner pad Y coordinates.

  A single Y translation cannot close both air gaps when the pad span exceeds
    the cube width; centering removes left/right asymmetry so the residual
    clearance is ``(gap_neg + gap_pos) / 2`` per face.

    Returns the applied Y shift in metres.
    """
    _, _, gap_neg, gap_pos = _lite6_glb_inner_contact_geometry(ctx)
    shift_y = 0.5 * (gap_pos - gap_neg)
    if abs(shift_y) <= 1e-6:
        return 0.0
    pos = ctx.obj.get_pos()[0].clone()
    old_y = float(pos[1].item())
    pos[1] = pos[1] + shift_y
    ctx.obj.set_pos(pos.unsqueeze(0), zero_velocity=True)
    return float(pos[1].item() - old_y)
