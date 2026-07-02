"""Link6 inverse-kinematics wrapper for the trajectory pipeline.

Wraps :meth:`gs.RigidEntity.inverse_kinematics` so sim/replay code streams
Cartesian (MoveL) targets one tick at a time and gets back per-tick arm joint
angles. The orientation is fixed gripper-down (roll=pi) for v1 (``tcp_offset=0``),
matching the grasp-place demo and reach env.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import xyz_to_quat

# Gripper-down orientation quaternion (roll=180deg), reused across MoveL ticks.
def down_quat(device=None, dtype=None) -> torch.Tensor:
    dev = device or gs.device
    dt = dtype or gs.tc_float
    return xyz_to_quat(
        torch.tensor([[math.pi, 0.0, 0.0]], device=dev, dtype=dt),
        rpy=True,
        degrees=False,
    )


def solve_link6_ik(
    robot,
    ik_link,
    pos_base,
    *,
    arm_dof_idx,
    quat=None,
    init_qpos=None,
) -> np.ndarray:
    """Solve IK for a single link6 base-frame xyz target.

    ``pos_base`` may be a length-3 sequence (m). Returns the 6 arm joint angles
    (rad) as a numpy array. ``quat`` defaults to gripper-down.
    """
    if quat is None:
        quat = down_quat()
    pos_t = torch.as_tensor([list(pos_base)], device=gs.device, dtype=gs.tc_float)
    sol = robot.inverse_kinematics(
        link=ik_link,
        pos=pos_t,
        quat=quat,
        dofs_idx_local=arm_dof_idx,
        init_qpos=init_qpos,
    )
    return sol[0, arm_dof_idx].detach().cpu().numpy().astype(np.float64)