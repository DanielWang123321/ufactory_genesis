"""Build reach-task observations matching ArmReachEnv layout."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ufactory.deploy.reach_config import ReachDeployConfig


def build_reach_obs(
    joint_pos: Sequence[float],
    joint_vel: Sequence[float],
    ee_pos_m: Sequence[float],
    target_pos_m: Sequence[float],
) -> np.ndarray:
    """Return 18-dim observation: q(6) + qdot(6) + ee(3) + target_rel(3)."""
    q = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
    qd = np.asarray(joint_vel, dtype=np.float64).reshape(-1)
    ee = np.asarray(ee_pos_m, dtype=np.float64).reshape(3)
    target = np.asarray(target_pos_m, dtype=np.float64).reshape(3)
    target_rel = target - ee
    return np.concatenate([q, qd, ee, target_rel])


def reach_obs_tensor(
    joint_pos: Sequence[float],
    joint_vel: Sequence[float],
    ee_pos_m: Sequence[float],
    target_pos_m: Sequence[float],
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    obs = build_reach_obs(joint_pos, joint_vel, ee_pos_m, target_pos_m)
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def parse_target_xyz(target: str) -> np.ndarray:
    """Parse ``x,y,z`` target position in metres."""
    parts = [float(v.strip()) for v in target.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected target 'x,y,z' in metres, got {target!r}")
    return np.asarray(parts, dtype=np.float64)


def validate_obs_shape(obs: np.ndarray, config: ReachDeployConfig) -> None:
    flat = np.asarray(obs, dtype=np.float64).reshape(-1)
    if flat.size != config.num_obs:
        raise ValueError(f"Expected obs dim {config.num_obs}, got {flat.size}")
