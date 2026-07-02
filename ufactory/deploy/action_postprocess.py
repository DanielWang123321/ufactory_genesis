"""Reach policy action post-processing shared by simulation and deploy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReachActionCommand:
    """Processed reach action ready to become a joint-position command."""

    raw_action: np.ndarray
    clipped_action: np.ndarray
    joint_delta: np.ndarray
    target_q: np.ndarray
    max_joint_delta_rad: float

    @property
    def action_saturation_fraction(self) -> float:
        clipped = np.isclose(self.raw_action, self.clipped_action, rtol=0.0, atol=1e-12)
        return float(1.0 - np.count_nonzero(clipped) / clipped.size)

    @property
    def delta_saturation_fraction(self) -> float:
        limited = np.isclose(np.abs(self.joint_delta), self.max_joint_delta_rad, rtol=0.0, atol=1e-12)
        return float(np.count_nonzero(limited) / limited.size)


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


def process_reach_action_np(
    current_q: np.ndarray,
    action: np.ndarray,
    *,
    action_scale: float,
    action_clip: float,
    max_joint_delta_rad: float,
) -> ReachActionCommand:
    """Clip a normalized reach action and convert it to a safe joint target."""
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    if raw.shape != current.shape:
        raise ValueError(f"Action shape {raw.shape} does not match current_q shape {current.shape}")

    clipped = np.clip(raw, -abs(float(action_clip)), abs(float(action_clip)))
    scaled_delta = clipped * float(action_scale)
    max_delta = abs(float(max_joint_delta_rad))
    if max_delta <= 0.0:
        raise ValueError(f"max_joint_delta_rad must be positive, got {max_joint_delta_rad}")
    joint_delta = np.clip(scaled_delta, -max_delta, max_delta)
    target_q = current + joint_delta
    return ReachActionCommand(
        raw_action=raw,
        clipped_action=clipped,
        joint_delta=joint_delta,
        target_q=target_q,
        max_joint_delta_rad=max_delta,
    )


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
