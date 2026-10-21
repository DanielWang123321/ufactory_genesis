"""Shared end-effector orientation contracts without simulator dependencies."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


GRIPPER_DOWN_RPY_RAD: tuple[float, float, float] = (math.pi, 0.0, 0.0)
GRIPPER_DOWN_QUAT_XYZW: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def wrap_rpy_diff_rad(diff: Sequence[float] | np.ndarray) -> np.ndarray:
    """Wrap element-wise Euler angle differences (rad) into [-pi, pi)."""
    arr = np.asarray(diff, dtype=np.float64)
    return (arr + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["GRIPPER_DOWN_QUAT_XYZW", "GRIPPER_DOWN_RPY_RAD", "wrap_rpy_diff_rad"]
