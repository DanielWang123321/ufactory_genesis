"""Genesis integration shared by Robotiq 2F-85 and 2F-140 simulations."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np
import torch

from ufactory.config.models import ROBOTIQ_2F_85_PHYSICS_PROFILE, ROBOTIQ_2F_140_PHYSICS_PROFILE
from ufactory.grippers.robotiq import get_robotiq_spec
from ufactory.simulation.g2_adapter import pad_normal  # noqa: F401
from ufactory.simulation.g2_controller import ControlOutput
from ufactory.simulation.robotiq_controller import RobotiqController
from ufactory.simulation.robotiq_model import (
    ROBOTIQ_ARMATURE,
    ROBOTIQ_KP,
    ROBOTIQ_KV,
    RobotiqModel,
)

ROBOTIQ_MAX_RIGID_SUBSTEP_DT_S = 0.000625
ROBOTIQ_MIMIC_CONSTRAINT_SOL_PARAMS = (0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)

_MIMIC_SUFFIXES = (
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


def _profile_name(variant: str) -> str:
    spec = get_robotiq_spec(variant)
    return ROBOTIQ_2F_85_PHYSICS_PROFILE if spec.adapter == "robotiq_2f_85" else ROBOTIQ_2F_140_PHYSICS_PROFILE


def robotiq_mimic_equality_names(variant: str) -> frozenset[str]:
    """Return the five Genesis equality constraint names for a Robotiq variant."""
    prefix = get_robotiq_spec(variant).prefix
    drive = f"{prefix}_drive_joint"
    return frozenset(f"mimic_{prefix}_{suffix}_to_{drive}" for suffix in _MIMIC_SUFFIXES)


def validate_robotiq_contact_substeps(*, dt: float, substeps: int, variant: str = "robotiq_2f_85") -> float:
    """Enforce the conservative rigid substep ceiling for Robotiq mimic constraints."""
    profile_name = _profile_name(variant)
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Robotiq contact scene dt must be finite and positive")
    if isinstance(substeps, bool) or not isinstance(substeps, Integral):
        raise ValueError("Robotiq contact scene substeps must be a positive integer")
    normalized_substeps = int(substeps)
    if normalized_substeps < 1:
        raise ValueError("Robotiq contact scene substeps must be a positive integer")
    substep_dt = dt / normalized_substeps
    if substep_dt > ROBOTIQ_MAX_RIGID_SUBSTEP_DT_S + 1e-15:
        required_substeps = math.ceil(dt / ROBOTIQ_MAX_RIGID_SUBSTEP_DT_S)
        raise ValueError(
            f"The {profile_name} project policy requires a rigid substep no longer than "
            f"0.625 ms; dt={dt:g}s with substeps={normalized_substeps} "
            f"gives {substep_dt * 1000.0:g} ms. Use at least {required_substeps} substeps."
        )
    return substep_dt


def configure_robotiq_mimic_constraints(robot: Any, variant: str = "robotiq_2f_85") -> int:
    """Apply the compliant solver profile to all five Robotiq mimic equalities."""
    profile_name = _profile_name(variant)
    expected_names = robotiq_mimic_equality_names(variant)
    sol_params = np.asarray(ROBOTIQ_MIMIC_CONSTRAINT_SOL_PARAMS, dtype=np.float64)
    equalities = {str(equality.name): equality for equality in robot.equalities}
    missing = sorted(expected_names - equalities.keys())
    if missing:
        raise RuntimeError(f"{profile_name} is missing Robotiq mimic equalities: {', '.join(missing)}")
    for name in sorted(expected_names):
        equalities[name].set_sol_params(sol_params)
    return len(expected_names)


def robotiq_pad_centers(model: RobotiqModel, left: Any, right: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from genesis.utils.geom import transform_by_quat

    g = model.linkage
    centers = []
    for link, y in ((left, g.pad_inner_y_m), (right, -g.pad_inner_y_m)):
        position = link.get_pos()
        point = position.new_tensor((0.0, y, g.pad_center_z_m))
        centers.append(position + transform_by_quat(point, link.get_quat()))
    return centers[0], centers[1]


def robotiq_measured_gap_mm(model: RobotiqModel, left: Any, right: Any, axis_link: Any = None) -> torch.Tensor:
    """Actual pad separation along the tool's closing axis in millimetres."""
    from genesis.utils.geom import transform_by_quat

    a, b = robotiq_pad_centers(model, left, right)
    normal = transform_by_quat(a.new_tensor((0.0, 1.0, 0.0)), (axis_link or left).get_quat())
    return ((b - a) * normal).sum(dim=-1).abs() * 1000.0


def configure_robotiq_drive(
    robot: Any,
    variant: str = "robotiq_2f_85",
    *,
    damping: float = 0.05,
    frictionloss: float = 0.0,
) -> tuple[list[int], list[int]]:
    """Only the master drive joint is actuated; all five mimic joints remain passive."""
    spec = get_robotiq_spec(variant)
    drive = [robot.get_joint(spec.drive_joint_name).dofs_idx_local[0]]
    all_idx = [robot.get_joint(name).dofs_idx_local[0] for name in spec.joint_names]
    state = robot.get_dofs_position(drive)
    passive = [idx for idx in all_idx if idx not in drive]
    robot.control_dofs_force(state.new_zeros((*state.shape[:-1], len(passive))), passive)
    robot.set_dofs_kp(state.new_tensor([ROBOTIQ_KP]), drive)
    robot.set_dofs_kv(state.new_tensor([ROBOTIQ_KV]), drive)
    robot.set_dofs_armature(state.new_full((len(all_idx),), ROBOTIQ_ARMATURE), all_idx)
    robot.set_dofs_damping(state.new_full((len(all_idx),), damping), all_idx)
    robot.set_dofs_frictionloss(state.new_full((len(all_idx),), frictionloss), all_idx)
    return drive, all_idx


class RobotiqGenesisController:
    """Single-environment Robotiq adapter; call ``apply`` before every ``scene.step()``."""

    def __init__(
        self,
        robot: Any,
        left: Any,
        right: Any,
        *,
        variant: str = "robotiq_2f_85",
        initial_position_mm: float | None = None,
        axis_link: Any = None,
    ) -> None:
        self.robot, self.left, self.right = robot, left, right
        self.axis_link = axis_link
        self.model = RobotiqModel.load(variant)
        self.variant = self.model.spec.adapter
        self.drive_idx = [robot.get_joint(self.model.spec.drive_joint_name).dofs_idx_local[0]]
        init_mm = self.model.spec.open_gap_mm if initial_position_mm is None else initial_position_mm
        self.controller = RobotiqController(self.model, initial_position_mm=init_mm)
        self.last_limit: float | None = None
        self.output: ControlOutput | None = None

    def gap_mm(self) -> float:
        return float(robotiq_measured_gap_mm(self.model, self.left, self.right, self.axis_link).reshape(-1)[0].item())

    def reset(self, position_mm: float | None = None, **kwargs: Any) -> None:
        pos = self.model.spec.open_gap_mm if position_mm is None else position_mm
        self.controller.reset(position_mm=pos, **kwargs)
        self.last_limit = None

    def command(self, position_mm: float, speed_percent: float = 50.0, force_percent: float = 50.0) -> None:
        self.controller.command(position_mm, speed_percent, force_percent)

    def apply(self, dt: float = 0.02) -> ControlOutput:
        q = self.robot.get_dofs_position(self.drive_idx)
        velocity = self.robot.get_dofs_velocity(self.drive_idx)
        self.output = output = self.controller.update(
            float(q.reshape(-1)[0]), float(velocity.reshape(-1)[0]), dt, measured_gap_mm=self.gap_mm()
        )
        limit = float(output.torque_limit_nm[0])
        if limit != self.last_limit:
            self.robot.set_dofs_force_range(q.new_tensor([-limit]), q.new_tensor([limit]), self.drive_idx)
            self.last_limit = limit
        self.robot.control_dofs_position_velocity(
            torch.full_like(q, float(output.position_rad[0])),
            torch.full_like(q, float(output.velocity_rad_s[0])),
            self.drive_idx,
        )
        return output


def apply_robotiq_batch_control(
    robot: Any,
    controller: RobotiqController,
    drive_idx: list[int],
    gap_mm: torch.Tensor,
    dt: float,
) -> ControlOutput:
    """Requires ``RigidOptions(batch_dofs_info=True)`` for independent per-env force caps."""
    q = robot.get_dofs_position(drive_idx).squeeze(-1)
    velocity = robot.get_dofs_velocity(drive_idx).squeeze(-1)
    output = controller.update(q, velocity, dt, measured_gap_mm=gap_mm)
    limit = output.torque_limit_nm.unsqueeze(-1)
    robot.set_dofs_force_range(-limit, limit, drive_idx)
    robot.control_dofs_position_velocity(
        output.position_rad.unsqueeze(-1), output.velocity_rad_s.unsqueeze(-1), drive_idx
    )
    return output


class RobotiqContactCollisionBackend:
    """Clamp preflight grasp envelope to ``object_width_m`` so object contact is allowed."""

    def __init__(self, backend: Any, *, variant: str, object_width_m: float) -> None:
        self.backend = backend
        self.model = RobotiqModel.load(variant)
        open_gap_m = self.model.spec.open_gap_m
        if not 0.0 < object_width_m < open_gap_m:
            raise ValueError(
                f"{self.model.spec.display_name} contact preflight requires object width within (0, {open_gap_m * 1000.0:g}) mm"
            )
        self.object_width_m = object_width_m

    def _drive(self, drive: float | None) -> float:
        gap = self.model.spec.open_gap_m if drive is None else self.model.spec.sim_drive_to_gap_m(drive)
        return self.model.linkage.drive_for_gap_m(max(gap, self.object_width_m))

    def check(self, q_rad: Any, *, stage: str, gripper_drive: float | None = None) -> Any:
        return self.backend.check(q_rad, stage=stage, gripper_drive=self._drive(gripper_drive))

    def check_all(self, q_rad: Any, *, stage: str, gripper_drive: float | None = None) -> Any:
        return self.backend.check_all(q_rad, stage=stage, gripper_drive=self._drive(gripper_drive))

    def check_all_within_margin(
        self, q_rad: Any, *, stage: str, margin_m: float, gripper_drive: float | None = None
    ) -> Any:
        return self.backend.check_all_within_margin(
            q_rad, stage=stage, margin_m=margin_m, gripper_drive=self._drive(gripper_drive)
        )
