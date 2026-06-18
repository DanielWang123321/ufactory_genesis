"""Bio Gripper G2 parallel open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

BIO_GRIPPER_G2_OPEN = 0.0
BIO_GRIPPER_G2_CLOSE = 0.04
BIO_GRIPPER_G2_HOLD_STEPS = 200

BIO_GRIPPER_G2_JOINTS = (
  "bio_gripper_g2_right_finger_joint",
  "bio_gripper_g2_left_finger_joint",
)


def bio_gripper_g2_dof_indices(robot) -> tuple[list[int], list[int]]:
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  if "bio_gripper_g2_right_finger_joint" not in joint_map:
    return [], []
  drive_idx = [joint_map["bio_gripper_g2_right_finger_joint"].dofs_idx_local[0]]
  all_idx = [joint_map[n].dofs_idx_local[0] for n in BIO_GRIPPER_G2_JOINTS if n in joint_map]
  return drive_idx, all_idx


def _targets_for_active(all_dof_idx: list[int], right_value: float, robot) -> np.ndarray:
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  values: list[float] = []
  for dof_i in all_dof_idx:
    for name in BIO_GRIPPER_G2_JOINTS:
      if name in joint_map and joint_map[name].dofs_idx_local[0] == dof_i:
        values.append(right_value if name == "bio_gripper_g2_right_finger_joint" else -right_value)
        break
  return np.array(values)


def setup_bio_gripper_g2_pd(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int]) -> None:
  pd_idx = gripper_dof_idx or all_gripper_dof_idx
  damp_idx = all_gripper_dof_idx or gripper_dof_idx
  robot.set_dofs_kp(np.full(len(pd_idx), 500.0), pd_idx)
  robot.set_dofs_kv(np.full(len(pd_idx), 50.0), pd_idx)
  robot.set_dofs_force_range(np.full(len(pd_idx), -20.0), np.full(len(pd_idx), 20.0), pd_idx)
  robot.set_dofs_damping(np.full(len(damp_idx), 0.05), damp_idx)
  robot.set_dofs_frictionloss(np.zeros(len(damp_idx)), damp_idx)


def set_bio_gripper_g2_pose(
  robot,
  gripper_dof_idx: list[int],
  all_gripper_dof_idx: list[int],
  right_value: float,
) -> None:
  active = all_gripper_dof_idx or gripper_dof_idx
  target = _targets_for_active(active, right_value, robot)
  robot.set_dofs_position(target, active)
  robot.control_dofs_position(target, active)


def control_bio_gripper_g2_pose(
  robot,
  gripper_dof_idx: list[int],
  all_gripper_dof_idx: list[int],
  right_value: float,
) -> None:
  active = all_gripper_dof_idx or gripper_dof_idx
  target = _targets_for_active(active, right_value, robot)
  robot.set_dofs_position(target, active, zero_velocity=False)
  robot.control_dofs_position(target, active)


def bio_gripper_g2_demo_target(step: int) -> float:
  phase = (step // BIO_GRIPPER_G2_HOLD_STEPS) % 2
  return BIO_GRIPPER_G2_CLOSE if phase else BIO_GRIPPER_G2_OPEN
