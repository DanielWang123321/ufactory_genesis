"""SDK TCP offset helpers for flange ↔ tool-frame pose conversion.

xArm ``tcp_offset`` is ``[x, y, z, roll, pitch, yaw]`` with xyz in mm and rpy in
radians when the API is constructed with ``is_radian=True``. The offset is
expressed in the flange (tool) frame:

    T_tcp = T_flange @ T_offset
    T_flange = T_tcp @ inv(T_offset)
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

_TCP_OFFSET_LEN = 6


def _as_pose6(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != _TCP_OFFSET_LEN:
        raise ValueError(f"{name} must have length {_TCP_OFFSET_LEN}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr.copy()


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (same convention as URDF RPY)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extract roll/pitch/yaw from R = Rz @ Ry @ Rx."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-8:
        # Gimbal lock: yaw := 0, roll from remaining DOF.
        roll = math.atan2(-r[0, 1], r[1, 1])
        yaw = 0.0
    else:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def pose_to_matrix(pose_mm_rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from ``[x_mm, y_mm, z_mm, roll, pitch, yaw]``."""
    pose = _as_pose6(pose_mm_rpy, name="pose")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rpy_matrix(float(pose[3]), float(pose[4]), float(pose[5]))
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform to ``[x_mm, y_mm, z_mm, roll, pitch, yaw]``."""
    t = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    roll, pitch, yaw = matrix_to_rpy(t[:3, :3])
    return np.array([t[0, 3], t[1, 3], t[2, 3], roll, pitch, yaw], dtype=np.float64)


def pose_flange_to_tcp(
    pose_flange_mm_rpy: Sequence[float] | np.ndarray,
    tcp_offset: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply firmware TCP offset: ``T_tcp = T_flange @ T_offset``."""
    offset = _as_pose6(tcp_offset, name="tcp_offset")
    return matrix_to_pose(pose_to_matrix(pose_flange_mm_rpy) @ pose_to_matrix(offset))


def pose_tcp_to_flange(
    pose_tcp_mm_rpy: Sequence[float] | np.ndarray,
    tcp_offset: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Remove firmware TCP offset: ``T_flange = T_tcp @ inv(T_offset)``."""
    offset = _as_pose6(tcp_offset, name="tcp_offset")
    return matrix_to_pose(pose_to_matrix(pose_tcp_mm_rpy) @ np.linalg.inv(pose_to_matrix(offset)))


def read_tcp_offset(
    arm: object,
    *,
    timeout_s: float = 2.0,
    poll_s: float = 0.05,
) -> np.ndarray:
    """Read ``arm.tcp_offset`` after waiting for the controller report to sync.

    Immediately after connect the SDK may report an all-zero offset before the
    real firmware value arrives. Non-zero values are returned as soon as they
    appear (with one confirmation poll). A persistent all-zero result after
    ``timeout_s`` is treated as a legitimate zero offset.
    """
    if timeout_s < 0.0:
        raise ValueError("timeout_s must be >= 0")
    if poll_s <= 0.0:
        raise ValueError("poll_s must be > 0")

    deadline = time.monotonic() + timeout_s
    last = np.zeros(_TCP_OFFSET_LEN, dtype=np.float64)
    while time.monotonic() < deadline:
        current = _as_pose6(getattr(arm, "tcp_offset"), name="arm.tcp_offset")
        last = current
        if float(np.linalg.norm(current[:3])) > 1e-6 or float(np.linalg.norm(current[3:])) > 1e-9:
            time.sleep(poll_s)
            return _as_pose6(getattr(arm, "tcp_offset"), name="arm.tcp_offset")
        time.sleep(poll_s)
    return last
