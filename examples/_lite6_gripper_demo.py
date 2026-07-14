"""Lite6 parallel gripper open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

from ufactory.grippers.lite6 import (
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    lite6_gripper_demo_drive,
)

# Lite6 ships with the reversed parallel gripper in this project: drive 0.0 is
# the 20 mm physical closed endpoint, while 0.0089 m is the 38 mm open endpoint.
LITE6_GRIPPER_CLOSE = LITE6_GRIPPER_SIM_CLOSED_DRIVE
LITE6_GRIPPER_OPEN = LITE6_GRIPPER_SIM_OPEN_DRIVE
LITE6_GRIPPER_HOLD_STEPS = 200

LITE6_GRIPPER_JOINTS = (
    "finger_joint1",
    "finger_joint2",
)


def lite6_gripper_dof_indices(robot) -> tuple[list[int], list[int]]:
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    if "finger_joint1" not in joint_map:
        return [], []
    drive_idx = [joint_map["finger_joint1"].dofs_idx_local[0]]
    all_idx = [joint_map[n].dofs_idx_local[0] for n in LITE6_GRIPPER_JOINTS if n in joint_map]
    return drive_idx, all_idx


def setup_lite6_gripper_pd(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int]) -> None:
    pd_idx = gripper_dof_idx or all_gripper_dof_idx
    damp_idx = all_gripper_dof_idx or gripper_dof_idx
    robot.set_dofs_kp(np.full(len(pd_idx), 500.0), pd_idx)
    robot.set_dofs_kv(np.full(len(pd_idx), 50.0), pd_idx)
    robot.set_dofs_force_range(np.full(len(pd_idx), -20.0), np.full(len(pd_idx), 20.0), pd_idx)
    robot.set_dofs_damping(np.full(len(damp_idx), 0.05), damp_idx)
    robot.set_dofs_frictionloss(np.zeros(len(damp_idx)), damp_idx)


def set_lite6_gripper_pose(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    value: float,
) -> None:
    active = all_gripper_dof_idx or gripper_dof_idx
    target = np.full(len(active), value)
    robot.set_dofs_position(target, active)
    robot.control_dofs_position(target, active)


def control_lite6_gripper_pose(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    value: float,
) -> None:
    active = all_gripper_dof_idx or gripper_dof_idx
    target = np.full(len(active), value)
    robot.set_dofs_position(target, active, zero_velocity=False)
    robot.control_dofs_position(target, active)


def lite6_gripper_demo_target(step: int) -> float:
    return lite6_gripper_demo_drive(step, hold_steps=LITE6_GRIPPER_HOLD_STEPS)
