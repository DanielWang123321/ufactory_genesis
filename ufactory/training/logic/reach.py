"""Genesis-free numeric core for the xArm6 reach task."""

from __future__ import annotations

import torch

from ufactory.training.reach_config import effective_max_joint_delta_rad


def reach_action_delta_torch(
    actions: torch.Tensor,
    *,
    action_scale: float,
    action_clip: float,
    max_joint_delta_rad: float,
) -> torch.Tensor:
    """Map the complete action range linearly to the effective increment limit."""

    if actions.shape[-1] <= 0:
        raise ValueError(f"Expected actions with a non-empty last dimension, got {tuple(actions.shape)}")
    limit = effective_max_joint_delta_rad(
        action_scale=action_scale, action_clip=action_clip, max_joint_delta_rad=max_joint_delta_rad
    )
    return torch.clamp(actions / float(action_clip), -1.0, 1.0) * limit


def reach_observation(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    target_pos: torch.Tensor,
    position_command: torch.Tensor,
) -> torch.Tensor:
    """Joint state, end-effector position, and target relative to the end effector."""

    return torch.cat([joint_pos, joint_vel, ee_pos, target_pos - ee_pos, position_command - joint_pos], dim=-1)


def reward_reach_position(ee_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    """Dense proximity reward, 1 at the target and about 0.37 at 10 cm."""

    distance_m = torch.linalg.norm(target_pos - ee_pos, dim=-1)
    return torch.exp(-10.0 * distance_m)
