"""Robotiq 2F-85 and 2F-140 open/close demo helpers for Genesis viewers."""

from __future__ import annotations

import numpy as np

from ufactory.grippers.robotiq import RobotiqSpec, get_robotiq_spec

ROBOTIQ_HOLD_STEPS = 200
ROBOTIQ_DEMO_HOLD_S = 0.5
ROBOTIQ_ARRIVAL_TOL_MM = 1.0


class RobotiqDemoCycle:
    """State machine that switches between closed and open after holding at target for ``hold_s``."""

    def __init__(
        self,
        variant: str = "robotiq_2f_85",
        *,
        dt: float = 0.02,
        hold_s: float = ROBOTIQ_DEMO_HOLD_S,
        arrival_tol_mm: float = ROBOTIQ_ARRIVAL_TOL_MM,
    ) -> None:
        self.spec: RobotiqSpec = get_robotiq_spec(variant)
        self.variant = self.spec.adapter
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
        return self.spec.linkage.drive_upper_rad if self.phase == "closed" else self.spec.linkage.drive_lower_rad

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
        settings = getattr(robot, "_ufactory_robotiq_demo_settings", (0.0, 50.0, 50.0))
        closed_gap_mm = float(settings[0])
        speed_percent = float(settings[1])
        if speed_percent <= 0.0:
            return False
        target_gap_mm = closed_gap_mm if self.phase == "closed" else self.spec.open_gap_mm
        adapter = getattr(robot, "_ufactory_robotiq_controller", None)
        if adapter is None or not callable(getattr(adapter, "gap_mm", None)):
            return True
        measured_gap_mm = float(adapter.gap_mm())
        inner = getattr(adapter, "controller", None)
        if inner is not None and hasattr(inner, "reference_gap_mm"):
            ref_gap_mm = float(np.asarray(inner.reference_gap_mm).reshape(-1)[0])
            if abs(ref_gap_mm - target_gap_mm) > 1e-4:
                return False
        return abs(measured_gap_mm - target_gap_mm) <= self.arrival_tol_mm


def detect_robotiq_variant(robot) -> str | None:
    joint_names = {j.name.split("/")[-1] for j in robot.joints}
    for variant in ("robotiq_2f_85", "robotiq_2f_140"):
        if f"{variant}_drive_joint" in joint_names:
            return variant
    return None


def robotiq_dof_indices(robot, variant: str | None = None) -> tuple[str | None, list[int], list[int]]:
    resolved = variant or detect_robotiq_variant(robot)
    if resolved is None:
        return None, [], []
    spec = get_robotiq_spec(resolved)
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    if spec.drive_joint_name not in joint_map:
        return None, [], []
    drive_idx = [joint_map[spec.drive_joint_name].dofs_idx_local[0]]
    all_idx = [joint_map[name].dofs_idx_local[0] for name in spec.joint_names if name in joint_map]
    return spec.adapter, drive_idx, all_idx


def setup_robotiq_pd(
    robot,
    variant: str,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
) -> None:
    from ufactory.simulation.robotiq_adapter import (
        RobotiqGenesisController,
        configure_robotiq_drive,
        configure_robotiq_mimic_constraints,
    )

    spec = get_robotiq_spec(variant)
    configure_robotiq_mimic_constraints(robot, spec.adapter)
    configure_robotiq_drive(robot, spec.adapter)
    robot._ufactory_robotiq_controller = RobotiqGenesisController(
        robot,
        robot.get_link(f"{spec.prefix}_left_finger"),
        robot.get_link(f"{spec.prefix}_right_finger"),
        variant=spec.adapter,
    )
    robot._ufactory_robotiq_demo_cycle = RobotiqDemoCycle(spec.adapter)
    set_robotiq_pose(robot, spec.adapter, gripper_dof_idx, all_gripper_dof_idx, spec.linkage.drive_lower_rad)


def get_robotiq_demo_cycle(robot, variant: str, *, dt: float = 0.02) -> RobotiqDemoCycle:
    cycle = getattr(robot, "_ufactory_robotiq_demo_cycle", None)
    if cycle is None:
        cycle = RobotiqDemoCycle(variant, dt=dt)
        robot._ufactory_robotiq_demo_cycle = cycle
    return cycle


def step_robotiq_demo_target(robot, variant: str, *, dt: float = 0.02) -> tuple[float, str]:
    """Return ``(target_drive, phase_label)`` for the arrival-confirmed 0.5 s hold demo cycle."""
    return get_robotiq_demo_cycle(robot, variant, dt=dt).step_command(robot)


def set_robotiq_pose(
    robot,
    variant: str,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    value: float,
) -> None:
    spec = get_robotiq_spec(variant)
    active = all_gripper_dof_idx or gripper_dof_idx
    adapter = getattr(robot, "_ufactory_robotiq_controller", None)
    if adapter is not None:
        gap_mm = 1000.0 * spec.sim_drive_to_gap_m(value)
        robot.set_dofs_position(np.full(len(active), value), active)
        adapter.reset(gap_mm)
        adapter.command(gap_mm)
        adapter.apply()
        return
    target = np.full(len(active), value)
    robot.set_dofs_position(target, active)
    robot.control_dofs_position(target, active)


def control_robotiq_pose(
    robot,
    variant: str,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    value: float,
) -> None:
    del gripper_dof_idx, all_gripper_dof_idx
    spec = get_robotiq_spec(variant)
    adapter = robot._ufactory_robotiq_controller
    position, speed, force = getattr(robot, "_ufactory_robotiq_demo_settings", (0.0, 50.0, 50.0))
    adapter.command(position if value > spec.linkage.drive_lower_rad else spec.open_gap_mm, speed, force)
    adapter.apply(0.02)
