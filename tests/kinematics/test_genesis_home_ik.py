"""Held start poses are refined to micrometres on the joint branch Genesis's default IK picks."""

from __future__ import annotations

import numpy as np
import pytest

from ufactory.kinematics.genesis import (
    HOME_IK_MAX_POS_ERR_M,
    HOME_IK_POS_TOL_M,
    HOME_IK_REFINE_ITERS,
    HOME_IK_ROT_TOL_RAD,
    solve_home_ik,
)

ARM = [0, 1, 2, 3, 4, 5]
TARGET = np.array([[0.3, 0.0, 0.36]])


class _Robot:
    def __init__(self, residual) -> None:
        self.residual = np.asarray(residual, dtype=np.float64)
        self.coarse = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]])
        self.refined = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 1e-3, 0.5]])
        self.calls: list[dict] = []

    def inverse_kinematics(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("return_error"):
            return self.refined, self.residual
        return self.coarse


def test_refines_from_the_default_solution_without_restarting() -> None:
    robot = _Robot([[2e-7, 0.0, 0.0, 0.0, 3e-6, 0.0]])
    seed = np.zeros((1, 7))
    q = solve_home_ik(robot, "link6", TARGET, "down", arm_dof_idx=ARM, init_qpos=seed)

    coarse_call, refine_call = robot.calls
    # The default solve (Genesis tolerances and random restarts) still picks the joint branch.
    assert coarse_call == {"link": "link6", "pos": TARGET, "quat": "down", "dofs_idx_local": ARM, "init_qpos": seed}
    assert refine_call["init_qpos"] is robot.coarse
    assert refine_call["max_samples"] == 1
    assert refine_call["max_solver_iters"] == HOME_IK_REFINE_ITERS
    assert (refine_call["pos_tol"], refine_call["rot_tol"]) == (HOME_IK_POS_TOL_M, HOME_IK_ROT_TOL_RAD)
    assert q is robot.refined


def test_without_a_seed_the_default_solve_starts_from_the_current_state() -> None:
    robot = _Robot(np.zeros((1, 6)))
    solve_home_ik(robot, "link6", TARGET, "down", arm_dof_idx=ARM)
    assert "init_qpos" not in robot.calls[0]


@pytest.mark.parametrize(
    "residual",
    [
        [0.9 * HOME_IK_MAX_POS_ERR_M, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Calibrated 5-axis xArm5 (XF130505D43A02): exact gripper-down is out of reach, the position is not.
        [1.8e-6, 0.0, 0.0, 0.0, 1.4747e-3, 0.0],
    ],
)
def test_accepts_a_residual_inside_the_bounds(residual) -> None:
    robot = _Robot([residual])
    assert solve_home_ik(robot, "link6", TARGET, "down", arm_dof_idx=ARM) is robot.refined


@pytest.mark.parametrize(
    "residual",
    [
        [4.45e-4, 0.0, 0.0, 0.0, 0.0, 0.0],  # where the default solve stopped at the xArm6 + 2F-140 home
        [0.0, 0.0, 0.0, 0.0, 6e-3, 0.0],
        [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_refuses_a_residual_above_the_bound(residual) -> None:
    with pytest.raises(ValueError, match=r"Genesis home IK residual .* for target \[0\.3, 0\.0, 0\.36\]"):
        solve_home_ik(_Robot([residual]), "link6", TARGET, "down", arm_dof_idx=ARM)
