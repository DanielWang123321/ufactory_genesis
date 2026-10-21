"""Link6 inverse-kinematics wrapper for the trajectory pipeline.

Wraps :meth:`gs.RigidEntity.inverse_kinematics` so sim/replay code streams
Cartesian (MoveL) targets one tick at a time and gets back per-tick arm joint
angles. The orientation is fixed gripper-down (roll=pi) for v1 (``tcp_offset=0``),
matching the pick-place demo.
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


DEFAULT_IK_MAX_POS_ERR_M = 2e-3


def solve_link6_ik(
    robot,
    ik_link,
    pos_world=None,
    *,
    arm_dof_idx,
    quat=None,
    init_qpos=None,
    damping: float | None = None,
    max_pos_err_m: float | None = DEFAULT_IK_MAX_POS_ERR_M,
    pos_base=None,
) -> np.ndarray:
    """Solve IK for a single link6 world-frame xyz target.

    ``pos_world`` may be a length-3 sequence (m) in the Genesis world frame.
    Returns the arm joint angles (rad) as a numpy array. ``quat`` defaults to
    gripper-down. When ``max_pos_err_m`` is set and ``robot`` supports forward
    kinematics, raises :class:`ValueError` if ``|FK(q) - pos_world|`` exceeds
    ``max_pos_err_m``.
    """
    if pos_world is None:
        if pos_base is None:
            raise TypeError("solve_link6_ik() missing required argument: 'pos_world'")
        pos_world = pos_base
    elif pos_base is not None:
        raise TypeError("solve_link6_ik() got both 'pos_world' and legacy 'pos_base'")
    if quat is None:
        quat = down_quat()
    target_xyz = np.asarray(list(pos_world), dtype=np.float64).reshape(3)
    pos_t = torch.as_tensor([target_xyz.tolist()], device=gs.device, dtype=gs.tc_float)
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
    if max_pos_err_m is not None:
        fk_pos = _eval_ik_link_pos(robot, sol, ik_link)
        if fk_pos is not None:
            pos_err = float(np.linalg.norm(fk_pos - target_xyz))
            if not np.isfinite(pos_err) or pos_err > float(max_pos_err_m):
                raise ValueError(
                    f"Genesis IK position residual {pos_err * 1000.0:.3f} mm exceeds "
                    f"limit {float(max_pos_err_m) * 1000.0:.3f} mm for target {target_xyz.tolist()}"
                )
    sol_vec = sol[0] if getattr(sol, "ndim", 1) == 2 else sol
    return sol_vec[arm_dof_idx].detach().cpu().numpy().astype(np.float64)


def _eval_ik_link_pos(robot, qpos, ik_link) -> np.ndarray | None:
    idx_local = getattr(ik_link, "idx_local", None)
    if idx_local is None:
        return None
    has_fk = (
        getattr(robot, "solver", None) is not None
        or getattr(robot, "_solver", None) is not None
        or callable(getattr(robot, "forward_kinematics", None))
    )
    if not has_fk:
        return None
    from ufactory.simulation.compat import forward_kinematics

    links_pos, _ = forward_kinematics(robot, qpos)
    idx = int(idx_local)
    ee_pos = links_pos[idx] if links_pos.ndim == 2 else links_pos[0, idx]
    if hasattr(ee_pos, "detach"):
        ee_pos = ee_pos.detach().cpu().numpy()
    return np.asarray(ee_pos, dtype=np.float64).reshape(3)


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
