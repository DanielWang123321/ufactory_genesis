"""Link6 inverse-kinematics wrapper for the trajectory pipeline.

Wraps :meth:`gs.RigidEntity.inverse_kinematics` so sim/replay code streams
Cartesian (MoveL) targets one tick at a time and gets back per-tick arm joint
angles. The orientation is fixed gripper-down (roll=pi) for v1 (``tcp_offset=0``),
matching the pick-place demo and reach env.
"""

from __future__ import annotations

import numpy as np
import torch

import genesis as gs
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW


# Gripper-down orientation quaternion (roll=180deg), reused across MoveL ticks.
def down_quat(device=None, dtype=None) -> torch.Tensor:
    dev = device or gs.device
    dt = dtype or gs.tc_float
    x, y, z, w = GRIPPER_DOWN_QUAT_XYZW
    # Genesis uses WXYZ at its IK boundary; the public kinematics contract is
    # XYZW, so convert explicitly instead of defining another orientation.
    return torch.tensor([[w, x, y, z]], device=dev, dtype=dt)


def solve_link6_ik(
    robot,
    ik_link,
    pos_base,
    *,
    arm_dof_idx,
    quat=None,
    init_qpos=None,
    damping: float | None = None,
) -> np.ndarray:
    """Solve IK for a single link6 base-frame xyz target.

    ``pos_base`` may be a length-3 sequence (m). Returns the 6 arm joint angles
    (rad) as a numpy array. ``quat`` defaults to gripper-down.
    """
    if quat is None:
        quat = down_quat()
    pos_t = torch.as_tensor([list(pos_base)], device=gs.device, dtype=gs.tc_float)
    if init_qpos is not None:
        init_qpos = _normalize_init_qpos(robot, init_qpos, arm_dof_idx)
    ik_kwargs = {}
    if damping is not None:
        ik_kwargs["damping"] = float(damping)
    sol = robot.inverse_kinematics(
        link=ik_link,
        pos=pos_t,
        quat=quat,
        dofs_idx_local=arm_dof_idx,
        init_qpos=init_qpos,
        **ik_kwargs,
    )
    return sol[0, arm_dof_idx].detach().cpu().numpy().astype(np.float64)


def _normalize_init_qpos(robot, init_qpos, arm_dof_idx) -> torch.Tensor:
    """Accept either full-robot qpos or arm-only qpos as an IK seed."""
    q = torch.as_tensor(init_qpos, device=gs.device, dtype=gs.tc_float)
    if q.ndim == 2 and q.shape[0] == 1:
        q = q.reshape(-1)
    arm_n = len(arm_dof_idx)
    if q.ndim == 1 and q.numel() == arm_n and getattr(robot, "n_qs", arm_n) != arm_n:
        try:
            full = robot.get_qpos()[0].detach().clone().to(device=gs.device, dtype=gs.tc_float)
        except Exception:
            full = torch.zeros(int(robot.n_qs), device=gs.device, dtype=gs.tc_float)
        full[arm_dof_idx] = q
        return full
    return q
