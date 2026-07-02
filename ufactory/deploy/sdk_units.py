"""xArm SDK pose unit conversions for sim-to-real deploy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# SDK ``get_position`` / ``get_forward_kinematics`` return xyz in millimetres.
MM_PER_M = 1000.0


def sdk_position_m(pose: Sequence[float]) -> np.ndarray:
    """Convert SDK pose xyz (mm) to metres."""
    return np.asarray(pose[:3], dtype=np.float64) / MM_PER_M


def sdk_position_mm(pose: Sequence[float]) -> np.ndarray:
    """SDK pose xyz already in millimetres."""
    return np.asarray(pose[:3], dtype=np.float64)
