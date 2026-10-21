"""Gripper G2 open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

from ufactory.grippers.g2 import (
    G2_JOINT_NAMES,
    GRIPPER_G2_SIM_OPEN_DRIVE as GRIPPER_OPEN,
    GRIPPER_G2_SIM_CLOSE_DRIVE as GRIPPER_CLOSE,
    gripper_g2_sim_drive_to_gap_m,
)

GRIPPER_HOLD_STEPS = 200


def gripper_dof_indices(robot) -> tuple[list[int], list[int]]:
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    if "drive_joint" not in joint_map:
        return [], []
    drive_idx = [joint_map["drive_joint"].dofs_idx_local[0]]
    all_idx = [joint_map[n].dofs_idx_local[0] for n in G2_JOINT_NAMES if n in joint_map]
    return drive_idx, all_idx


def setup_gripper_pd(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int]) -> None:
    from ufactory.simulation import configure_g2_mimic_constraints
    from ufactory.simulation.g2_adapter import G2GenesisController, configure_g2_drive

    configure_g2_mimic_constraints(robot)
    configure_g2_drive(robot)
    robot._ufactory_g2_controller = G2GenesisController(
        robot, robot.get_link("left_finger"), robot.get_link("right_finger")
    )
    set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)


def set_gripper_pose(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int], value: float) -> None:
    # All mimic joints must be set explicitly in Genesis visual preview.
    active = all_gripper_dof_idx or gripper_dof_idx
    adapter = getattr(robot, "_ufactory_g2_controller", None)
    if adapter is not None:
        gap_mm = 1000 * gripper_g2_sim_drive_to_gap_m(value)
        robot.set_dofs_position(np.full(len(active), value), active)
        adapter.reset(gap_mm)
        adapter.command(gap_mm)
        adapter.apply()
        return
    target = np.full(len(active), value)
    robot.set_dofs_position(target, active)
    robot.control_dofs_position(target, active)


def control_gripper_pose(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int], value: float) -> None:
    adapter = robot._ufactory_g2_controller
    position, speed, force = getattr(robot, "_ufactory_g2_demo_settings", (0, 50, 50))
    adapter.command(position if value > GRIPPER_OPEN else 84, speed, force)
    adapter.apply(0.02)


def gripper_demo_target(step: int) -> float:
    phase = (step // GRIPPER_HOLD_STEPS) % 2
    return GRIPPER_CLOSE if phase else GRIPPER_OPEN
