"""Gripper G2 open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.85
GRIPPER_HOLD_STEPS = 200

ALL_GRIPPER_JOINTS = (
  "drive_joint",
  "left_finger_joint",
  "left_inner_knuckle_joint",
  "right_outer_knuckle_joint",
  "right_finger_joint",
  "right_inner_knuckle_joint",
)


def gripper_dof_indices(robot) -> tuple[list[int], list[int]]:
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  if "drive_joint" not in joint_map:
    return [], []
  drive_idx = [joint_map["drive_joint"].dofs_idx_local[0]]
  all_idx = [joint_map[n].dofs_idx_local[0] for n in ALL_GRIPPER_JOINTS if n in joint_map]
  return drive_idx, all_idx


def setup_gripper_pd(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int]) -> None:
  active = all_gripper_dof_idx or gripper_dof_idx
  n = len(active)
  robot.set_dofs_kp(np.full(n, 30.0), active)
  robot.set_dofs_kv(np.full(n, 6.0), active)
  robot.set_dofs_force_range(np.full(n, -50.0), np.full(n, 50.0), active)
  robot.set_dofs_damping(np.full(n, 0.05), active)
  robot.set_dofs_frictionloss(np.zeros(n), active)


def set_gripper_pose(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int], value: float) -> None:
  active = all_gripper_dof_idx or gripper_dof_idx
  target = np.full(len(active), value)
  robot.set_dofs_position(target, active)
  robot.control_dofs_position(target, active)


def control_gripper_pose(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int], value: float) -> None:
  active = all_gripper_dof_idx or gripper_dof_idx
  target = np.full(len(active), value)
  robot.set_dofs_position(target, active, zero_velocity=False)
  robot.control_dofs_position(target, active)


def gripper_demo_target(step: int) -> float:
  phase = (step // GRIPPER_HOLD_STEPS) % 2
  return GRIPPER_CLOSE if phase else GRIPPER_OPEN
