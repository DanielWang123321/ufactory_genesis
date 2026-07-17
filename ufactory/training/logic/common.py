"""Shared Genesis-free reward helpers for RL tasks (CPU-testable)."""

from __future__ import annotations

import torch


def action_penalty(joint_vel: torch.Tensor) -> torch.Tensor:
    """-sum(qd^2); penalizes fast joints. -> (N,)."""
    return -torch.sum(joint_vel**2, dim=-1)
