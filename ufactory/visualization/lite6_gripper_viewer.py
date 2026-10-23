"""Lite6 parallel gripper open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

from ufactory.grippers.lite6 import (
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    configure_lite6_gripper_drive,
    lite6_gripper_demo_drive,
)

# Lite6 ships with the reversed parallel gripper in this project: drive 0.0 is
# the 20 mm physical closed endpoint, while 0.0089 m is the 38 mm open endpoint.
LITE6_GRIPPER_CLOSE = LITE6_GRIPPER_SIM_CLOSED_DRIVE
LITE6_GRIPPER_OPEN = LITE6_GRIPPER_SIM_OPEN_DRIVE
LITE6_GRIPPER_HOLD_STEPS = 200
LITE6_GRIPPER_DEMO_HOLD_S = 0.5
LITE6_GRIPPER_ARRIVAL_TOL_M = 0.0005

LITE6_GRIPPER_JOINTS = (
    "finger_joint1",
    "finger_joint2",
)


class Lite6GripperDemoCycle:
    """State machine that switches between closed and open after holding at target for ``hold_s``."""

    def __init__(
        self,
        drive_dof_idx: list[int] | None = None,
        *,
        dt: float = 0.01,
        hold_s: float = LITE6_GRIPPER_DEMO_HOLD_S,
        arrival_tol_m: float = LITE6_GRIPPER_ARRIVAL_TOL_M,
    ) -> None:
        self.drive_dof_idx = list(drive_dof_idx or [])
        self.dt = float(dt)
        self.hold_s = float(hold_s)
        self.arrival_tol_m = float(arrival_tol_m)
        self.hold_steps_required = max(1, int(round(self.hold_s / self.dt)))
        self.phase: str = "closed"
        self.hold_steps: int = 0
        self.completed_cycles: int = 0
        self._pending_observation: bool = False

    @property
    def target_drive(self) -> float:
        return LITE6_GRIPPER_CLOSE if self.phase == "closed" else LITE6_GRIPPER_OPEN

    def step_command(self, robot) -> tuple[float, str]:
        if self._pending_observation:
            self.on_step_completed(robot)
        self._pending_observation = True
        return self.target_drive, self.phase

    def on_step_completed(self, robot) -> None:
        if not self._pending_observation:
            return
        self._pending_observation = False
        if self._is_at_target(robot):
            self.hold_steps += 1
            if self.hold_steps >= self.hold_steps_required:
                self.hold_steps = 0
                if self.phase == "open":
                    self.completed_cycles += 1
                    self.phase = "closed"
                else:
                    self.phase = "open"
        else:
            self.hold_steps = 0

    def _is_at_target(self, robot) -> bool:
        if not self.drive_dof_idx or not callable(getattr(robot, "get_dofs_position", None)):
            return True
        pos = robot.get_dofs_position(self.drive_dof_idx)
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        elif hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
        current_m = float(np.asarray(pos).reshape(-1)[0])
        return abs(current_m - self.target_drive) <= self.arrival_tol_m


def lite6_gripper_dof_indices(robot) -> tuple[list[int], list[int]]:
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    if "finger_joint1" not in joint_map:
        return [], []
    drive_idx = [joint_map["finger_joint1"].dofs_idx_local[0]]
    all_idx = [joint_map[n].dofs_idx_local[0] for n in LITE6_GRIPPER_JOINTS if n in joint_map]
    return drive_idx, all_idx


def setup_lite6_gripper_pd(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int]) -> None:
    configure_lite6_gripper_drive(robot, gripper_dof_idx, all_gripper_dof_idx)
    robot._ufactory_lite6_demo_cycle = Lite6GripperDemoCycle(drive_dof_idx=gripper_dof_idx)


def get_lite6_gripper_demo_cycle(
    robot,
    drive_dof_idx: list[int] | None = None,
    *,
    dt: float = 0.01,
) -> Lite6GripperDemoCycle:
    cycle = getattr(robot, "_ufactory_lite6_demo_cycle", None)
    if cycle is None:
        cycle = Lite6GripperDemoCycle(drive_dof_idx=drive_dof_idx, dt=dt)
        robot._ufactory_lite6_demo_cycle = cycle
    elif drive_dof_idx and not cycle.drive_dof_idx:
        cycle.drive_dof_idx = list(drive_dof_idx)
    return cycle


def step_lite6_gripper_demo_target(
    robot,
    drive_dof_idx: list[int] | None = None,
    *,
    dt: float = 0.01,
) -> tuple[float, str]:
    """Return ``(target_drive, phase_label)`` for the arrival-confirmed 0.5 s hold demo cycle."""
    return get_lite6_gripper_demo_cycle(robot, drive_dof_idx, dt=dt).step_command(robot)


def set_lite6_gripper_pose(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    value: float,
) -> None:
    active = all_gripper_dof_idx or gripper_dof_idx
    drive = gripper_dof_idx or all_gripper_dof_idx
    robot.set_dofs_position(np.full(len(active), value), active)
    robot.control_dofs_position(np.full(len(drive), value), drive)
    passive = [idx for idx in active if idx not in drive]
    if passive and callable(getattr(robot, "control_dofs_force", None)):
        robot.control_dofs_force(np.zeros(len(passive)), passive)


def control_lite6_gripper_pose(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    value: float,
) -> None:
    drive = gripper_dof_idx or all_gripper_dof_idx
    robot.control_dofs_position(np.full(len(drive), value), drive)


def lite6_gripper_demo_target(step: int) -> float:
    return lite6_gripper_demo_drive(step, hold_steps=LITE6_GRIPPER_HOLD_STEPS)
