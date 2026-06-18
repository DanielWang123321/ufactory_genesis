"""Scripted pick-place trajectory for IL dataset recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from constants import (
    DRIVE_CLOSED,
    DRIVE_OPEN,
    HOME_Z,
    LIFT_Z,
    OBJ_XY_DEFAULT,
    PLACE_XY_DEFAULT,
    TABLE_HEIGHT,
)


@dataclass
class PickPlaceConfig:
    table_height: float = TABLE_HEIGHT
    obj_xy: tuple[float, float] = OBJ_XY_DEFAULT
    place_xy: tuple[float, float] = PLACE_XY_DEFAULT
    lift_z: float = LIFT_Z
    home_z: float = HOME_Z
    finger_z_offset: float = 0.0
    finger_pad_below_fc: float = 0.061
    finger_close_descent: float = 0.015
    grasp_table_clearance: float = 0.010


@dataclass
class TrajectoryStep:
    """One control step target."""

    link6_pos: tuple[float, float, float]
    drive_joint: float
    label: str = ""


def compute_grasp_link6_z(cfg: PickPlaceConfig) -> float:
    return (
        cfg.table_height
        + cfg.grasp_table_clearance
        + cfg.finger_close_descent
        + cfg.finger_pad_below_fc
        + cfg.finger_z_offset
    )


def build_pick_place_trajectory(cfg: PickPlaceConfig) -> list[TrajectoryStep]:
    """Expand scripted demo into per-step targets (~60Hz segments)."""
    th = cfg.table_height
    grasp_z = compute_grasp_link6_z(cfg)
    pre_grasp_z = grasp_z + 0.10
    lift_z = th + cfg.lift_z
    home = (0.3, 0.0, th + cfg.home_z)
    ox, oy = cfg.obj_xy
    px, py = cfg.place_xy

    segments: list[tuple[tuple[float, float, float], tuple[float, float, float], float, float, int, str]] = [
        # (start, end, grip_start, grip_end, steps, label)
        (home, home, DRIVE_OPEN, DRIVE_OPEN, 30, "home"),
        (home, (ox, oy, pre_grasp_z), DRIVE_OPEN, DRIVE_OPEN, 100, "pre_grasp"),
        ((ox, oy, pre_grasp_z), (ox, oy, grasp_z), DRIVE_OPEN, DRIVE_OPEN, 120, "descend"),
        ((ox, oy, grasp_z), (ox, oy, grasp_z), DRIVE_OPEN, DRIVE_OPEN, 30, "settle"),
        ((ox, oy, grasp_z), (ox, oy, grasp_z), DRIVE_OPEN, DRIVE_CLOSED, 200, "grasp"),
        ((ox, oy, grasp_z), (ox, oy, lift_z), DRIVE_CLOSED, DRIVE_CLOSED, 120, "lift"),
        ((ox, oy, lift_z), (px, py, lift_z), DRIVE_CLOSED, DRIVE_CLOSED, 150, "move_place"),
        ((px, py, lift_z), (px, py, grasp_z), DRIVE_CLOSED, DRIVE_CLOSED, 100, "lower"),
        ((px, py, grasp_z), (px, py, grasp_z), DRIVE_CLOSED, DRIVE_OPEN, 150, "release"),
        ((px, py, grasp_z), (px, py, lift_z), DRIVE_OPEN, DRIVE_OPEN, 80, "retreat"),
        ((px, py, lift_z), home, DRIVE_OPEN, DRIVE_OPEN, 150, "restore_home"),
    ]

    steps: list[TrajectoryStep] = []
    for start, end, g0, g1, n, label in segments:
        for i in range(n):
            alpha = (i + 1) / n
            pos = (
                start[0] + alpha * (end[0] - start[0]),
                start[1] + alpha * (end[1] - start[1]),
                start[2] + alpha * (end[2] - start[2]),
            )
            drive = g0 + alpha * (g1 - g0)
            steps.append(TrajectoryStep(link6_pos=pos, drive_joint=drive, label=label))
    return steps


def measure_finger_z_offset(ik_link, left_finger, right_finger) -> float:
    link6_pos = ik_link.get_pos()[0]
    fc = ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]
    return (link6_pos[2] - fc[2]).item()


StepCallback = Callable[[int, TrajectoryStep], None]


def execute_trajectory_step(
    robot,
    scene,
    ik_link,
    arm_dof_idx: list[int],
    gripper_dof_idx: list[int],
    down_quat: torch.Tensor,
    step: TrajectoryStep,
) -> None:
    import genesis as gs

    target_t = torch.tensor([step.link6_pos], device=gs.device, dtype=gs.tc_float)
    grip_t = torch.tensor([[step.drive_joint]], device=gs.device, dtype=gs.tc_float)
    qpos = robot.inverse_kinematics(
        link=ik_link,
        pos=target_t,
        quat=down_quat,
        dofs_idx_local=arm_dof_idx,
    )
    robot.control_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx)
    robot.control_dofs_position(grip_t, gripper_dof_idx)
    scene.step()
