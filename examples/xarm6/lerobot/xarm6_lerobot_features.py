"""EE pose and G2 gripper encoding for LeRobot datasets."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import constants

DRIVE_CLOSED = constants.DRIVE_CLOSED
DRIVE_OPEN = constants.DRIVE_OPEN
GRIPPER_IDX = constants.GRIPPER_IDX
GRIPPER_MAX_WIDTH_MM = constants.GRIPPER_MAX_WIDTH_MM
POS_SLICE = constants.POS_SLICE
ROTVEC_SLICE = constants.ROTVEC_SLICE
STATE_DIM = constants.STATE_DIM

__all__ = [
    "STATE_DIM",
    "gripper_openness_from_drive",
    "drive_from_gripper_openness",
    "gripper_openness_from_width_mm",
    "width_mm_from_gripper_openness",
    "pack_ee_state",
    "unpack_ee_state",
    "quat_wxyz_to_rotvec",
    "rotvec_to_quat_wxyz",
    "assert_allowed_features",
]


def gripper_openness_from_drive(drive_joint: float) -> float:
    """Map G2 drive_joint to openness in [0, 1] (0=closed, 1=open)."""
    drive = float(np.clip(drive_joint, DRIVE_OPEN, DRIVE_CLOSED))
    return float(1.0 - drive / DRIVE_CLOSED)


def drive_from_gripper_openness(openness: float) -> float:
    """Inverse of gripper_openness_from_drive."""
    g = float(np.clip(openness, 0.0, 1.0))
    return float(DRIVE_CLOSED * (1.0 - g))


def gripper_openness_from_width_mm(width_mm: float) -> float:
    """FastUMI clamp width (mm) → openness."""
    return float(np.clip(width_mm / GRIPPER_MAX_WIDTH_MM, 0.0, 1.0))


def width_mm_from_gripper_openness(openness: float) -> float:
    return float(np.clip(openness, 0.0, 1.0) * GRIPPER_MAX_WIDTH_MM)


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → rotation vector."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector → quaternion (w, x, y, z)."""
    q_xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def pack_ee_state(
    pos_base: np.ndarray,
    rotvec: np.ndarray,
    gripper_openness: float,
) -> np.ndarray:
    """Build 7D EE vector in robot-base frame."""
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[POS_SLICE] = np.asarray(pos_base, dtype=np.float32).reshape(3)
    out[ROTVEC_SLICE] = np.asarray(rotvec, dtype=np.float32).reshape(3)
    out[GRIPPER_IDX] = np.float32(gripper_openness)
    return out


def unpack_ee_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    s = np.asarray(state, dtype=np.float32).reshape(STATE_DIM)
    return s[POS_SLICE].copy(), s[ROTVEC_SLICE].copy(), float(s[GRIPPER_IDX])


def assert_allowed_features(features: dict) -> None:
    """Ensure dataset schema excludes ToF/depth (training whitelist)."""
    from constants import ALLOWED_FEATURE_KEYS

    keys = set(features.keys())
    extra = keys - ALLOWED_FEATURE_KEYS
    if extra:
        raise ValueError(f"Disallowed dataset features (ToF/depth not permitted): {sorted(extra)}")
    missing = ALLOWED_FEATURE_KEYS - keys
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
