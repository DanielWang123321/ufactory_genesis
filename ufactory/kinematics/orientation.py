"""Shared end-effector orientation contracts without simulator dependencies."""

from __future__ import annotations

import math


GRIPPER_DOWN_RPY_RAD: tuple[float, float, float] = (math.pi, 0.0, 0.0)
GRIPPER_DOWN_QUAT_XYZW: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


__all__ = ["GRIPPER_DOWN_QUAT_XYZW", "GRIPPER_DOWN_RPY_RAD"]
