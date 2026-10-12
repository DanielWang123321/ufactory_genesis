"""Dependency-free reach control limits and environment version."""

from __future__ import annotations

import math

REACH_ENV_VERSION = 2


def effective_max_joint_delta_rad(
    *,
    action_scale: float,
    action_clip: float,
    ctrl_dt: float | None = None,
    servo_speed_rad_s: float | None = None,
    max_joint_delta_rad: float | None = None,
) -> float:
    """Per-step joint increment limit after scaling and the servo speed cap."""

    values = [action_scale, action_clip, ctrl_dt, servo_speed_rad_s, max_joint_delta_rad]
    if any(value is not None and (not math.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError("reach control limits must be finite and positive")
    action_bound = float(action_scale) * float(action_clip)
    candidates = [action_bound]
    if ctrl_dt is not None and servo_speed_rad_s is not None:
        candidates.append(abs(float(servo_speed_rad_s)) * abs(float(ctrl_dt)))
    if max_joint_delta_rad is not None:
        candidates.append(abs(float(max_joint_delta_rad)))
    limit = min(candidates)
    if limit <= 0.0:
        raise ValueError(f"max joint delta must be positive, got {limit}")
    return float(limit)

