"""Gripper G2 open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

from ufactory.grippers.g2 import (
    G2_JOINT_NAMES,
    GRIPPER_G2_OPEN_GAP_MM,
    GRIPPER_G2_SIM_OPEN_DRIVE as GRIPPER_OPEN,
    GRIPPER_G2_SIM_CLOSE_DRIVE as GRIPPER_CLOSE,
    gripper_g2_sim_drive_to_gap_m,
)

GRIPPER_HOLD_STEPS = 200
GRIPPER_DEMO_HOLD_S = 0.5
GRIPPER_G2_ARRIVAL_TOL_MM = 1.0


class G2DemoCycle:
    """State machine that switches between closed and open after holding at target for ``hold_s``."""

    def __init__(
        self,
        *,
        dt: float = 0.02,
        hold_s: float = GRIPPER_DEMO_HOLD_S,
        arrival_tol_mm: float = GRIPPER_G2_ARRIVAL_TOL_MM,
    ) -> None:
        self.dt = float(dt)
        self.hold_s = float(hold_s)
        self.arrival_tol_mm = float(arrival_tol_mm)
        self.hold_steps_required = max(1, int(round(self.hold_s / self.dt)))
        self.phase: str = "closed"
        self.hold_steps: int = 0
        self.completed_cycles: int = 0
        self._pending_observation: bool = False

    @property
    def target_drive(self) -> float:
        return GRIPPER_CLOSE if self.phase == "closed" else GRIPPER_OPEN

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
        settings = getattr(robot, "_ufactory_g2_demo_settings", (0.0, 50.0, 50.0))
        closed_gap_mm = float(settings[0])
        speed_percent = float(settings[1])
        if speed_percent <= 0.0:
            return False
        target_gap_mm = closed_gap_mm if self.phase == "closed" else GRIPPER_G2_OPEN_GAP_MM
        adapter = getattr(robot, "_ufactory_g2_controller", None)
        if adapter is None or not callable(getattr(adapter, "gap_mm", None)):
            return True
        measured_gap_mm = float(adapter.gap_mm())
        inner = getattr(adapter, "controller", None)
        if inner is not None and hasattr(inner, "reference_gap_mm"):
            ref_gap_mm = float(np.asarray(inner.reference_gap_mm).reshape(-1)[0])
            if abs(ref_gap_mm - target_gap_mm) > 1e-4:
                return False
        return abs(measured_gap_mm - target_gap_mm) <= self.arrival_tol_mm


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
    robot._ufactory_g2_demo_cycle = G2DemoCycle()
    set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)


def get_g2_demo_cycle(robot, *, dt: float = 0.02) -> G2DemoCycle:
    cycle = getattr(robot, "_ufactory_g2_demo_cycle", None)
    if cycle is None:
        cycle = G2DemoCycle(dt=dt)
        robot._ufactory_g2_demo_cycle = cycle
    return cycle


def step_gripper_demo_target(robot, *, dt: float = 0.02) -> tuple[float, str]:
    """Return ``(target_drive, phase_label)`` for the arrival-confirmed 0.5 s hold demo cycle."""
    return get_g2_demo_cycle(robot, dt=dt).step_command(robot)


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
