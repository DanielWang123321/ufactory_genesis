"""xArm Gripper G2 command-space conversions.

The Genesis URDF exposes a simulated ``drive_joint`` where 0.0 is fully open
and 0.85 is fully closed. The xArm Python SDK exposes Gripper G2 position in
millimetres where 0 mm is closed and 84 mm is open.
"""

from __future__ import annotations

GRIPPER_G2_CLOSED_GAP_MM = 0.0
GRIPPER_G2_OPEN_GAP_MM = 84.0
GRIPPER_G2_OPEN_GAP_M = GRIPPER_G2_OPEN_GAP_MM / 1000.0

GRIPPER_G2_SIM_OPEN_DRIVE = 0.0
GRIPPER_G2_SIM_CLOSE_DRIVE = 0.85


def clamp_gripper_g2_gap_m(gap_m: float) -> float:
    """Clamp a two-finger gap in metres to the Gripper G2 physical range."""
    return max(GRIPPER_G2_CLOSED_GAP_MM / 1000.0, min(GRIPPER_G2_OPEN_GAP_M, float(gap_m)))


def gripper_g2_gap_m_to_sim_drive(gap_m: float) -> float:
    """Map a desired physical gap in metres to Genesis ``drive_joint`` position."""
    gap = clamp_gripper_g2_gap_m(gap_m)
    open_fraction = gap / GRIPPER_G2_OPEN_GAP_M
    return GRIPPER_G2_SIM_CLOSE_DRIVE * (1.0 - open_fraction)


def gripper_g2_sim_drive_to_gap_m(drive: float) -> float:
    """Map Genesis ``drive_joint`` position to physical two-finger gap in metres."""
    clipped = max(GRIPPER_G2_SIM_OPEN_DRIVE, min(GRIPPER_G2_SIM_CLOSE_DRIVE, float(drive)))
    closed_fraction = clipped / GRIPPER_G2_SIM_CLOSE_DRIVE
    return GRIPPER_G2_OPEN_GAP_M * (1.0 - closed_fraction)


def gripper_g2_gap_m_to_sdk_pos_mm(gap_m: float) -> float:
    """Map a physical gap in metres to ``set_gripper_g2_position`` millimetres."""
    return clamp_gripper_g2_gap_m(gap_m) * 1000.0


def gripper_g2_sdk_pos_mm_to_gap_m(pos_mm: float) -> float:
    """Map ``get_gripper_g2_position`` millimetres to physical gap in metres."""
    return clamp_gripper_g2_gap_m(float(pos_mm) / 1000.0)
