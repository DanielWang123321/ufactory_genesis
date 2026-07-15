"""Reach-policy action scaling shared by training and simulation evaluation."""

from __future__ import annotations

import torch


def effective_max_joint_delta_rad(
    *,
    action_scale: float,
    action_clip: float,
    ctrl_dt: float | None = None,
    servo_speed_rad_s: float | None = None,
    max_joint_delta_rad: float | None = None,
) -> float:
    """Return the per-step joint delta limit used by reach action adapters."""
    action_bound = abs(float(action_scale)) * abs(float(action_clip))
    candidates = [action_bound]
    if ctrl_dt is not None and servo_speed_rad_s is not None:
        candidates.append(abs(float(servo_speed_rad_s)) * abs(float(ctrl_dt)))
    if max_joint_delta_rad is not None:
        candidates.append(abs(float(max_joint_delta_rad)))
    limit = min(candidates)
    if limit <= 0.0:
        raise ValueError(f"max joint delta must be positive, got {limit}")
    return float(limit)


def reach_action_delta_torch(
    actions: torch.Tensor,
    *,
    action_scale: float,
    action_clip: float,
    max_joint_delta_rad: float,
) -> torch.Tensor:
    """Torch equivalent of the normalized action to joint-delta conversion."""
    if actions.shape[-1] <= 0:
        raise ValueError(f"Expected actions with non-empty last dimension, got {tuple(actions.shape)}")
    clipped = torch.clamp(actions, -abs(float(action_clip)), abs(float(action_clip)))
    scaled_delta = clipped * float(action_scale)
    max_delta = abs(float(max_joint_delta_rad))
    if max_delta <= 0.0:
        raise ValueError(f"max_joint_delta_rad must be positive, got {max_joint_delta_rad}")
    return torch.clamp(scaled_delta, -max_delta, max_delta)
