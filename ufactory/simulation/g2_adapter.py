"""Genesis integration shared by G2 simulations.

Rigid substep guard, mimic equalities, joint control, pad measurements, pad
contact forces and contact preflight.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import torch

from ufactory.config.models import G2_PHYSICS_PROFILE
from ufactory.grippers.g2 import G2_JOINT_NAMES, gripper_g2_sim_drive_to_gap_m

from ufactory.simulation.g2_controller import ControlOutput, G2Controller
from ufactory.simulation.g2_model import G2Model, G2_KP, G2_KV, G2_ARMATURE

G2_MAX_RIGID_SUBSTEP_DT_S = 0.000625
# Genesis solver tuple:
# (timeconst, dampratio, dmin, dmax, width, midpoint, power).
#
# The historical project value used dmax=0.001 and dampratio=0.1, which creates
# reference-acceleration coefficients several orders of magnitude above the
# defaults and can overflow when all five mimic equalities meet bilateral finger
# contact. This deliberately compliant profile passed the 128-environment
# perturbed-contact case at both 8 and 32 substeps.
G2_MIMIC_CONSTRAINT_SOL_PARAMS = (0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)
G2_MIMIC_EQUALITY_NAMES = frozenset(
    {
        "mimic_left_finger_joint_to_drive_joint",
        "mimic_left_inner_knuckle_joint_to_drive_joint",
        "mimic_right_outer_knuckle_joint_to_drive_joint",
        "mimic_right_finger_joint_to_drive_joint",
        "mimic_right_inner_knuckle_joint_to_drive_joint",
    }
)


def object_finger_contact_forces_n(
    contacts: Mapping[str, Any],
    *,
    left_link_idx: int,
    right_link_idx: int,
    left_normal: torch.Tensor | None = None,
    right_normal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the object-only net contact load carried by each finger link.

    ``RigidEntity.get_contacts`` returns dense ``(B, C, ...)`` tensors for a
    parallel scene and ``(C, ...)`` tensors for one environment.  This helper
    normalizes both forms to one force value per environment, selects the force
    applied to the requested finger regardless of whether it is contact side A
    or B, and vector-sums all of that finger's object contacts before taking the
    magnitude.  Vector summation avoids overstating the pad load when Genesis
    emits several nearby contact points for one physical patch.

    Without normals the result is the net force magnitude (used by RL
    observations).  With world-frame unit pad normals of shape ``(3,)`` or
    ``(B, 3)`` the result is the absolute normal component, i.e. the clamping
    force compared against the calibrated static force.
    """

    link_a = torch.as_tensor(contacts.get("link_a", torch.empty(0, dtype=torch.int64)))
    link_b = torch.as_tensor(contacts.get("link_b", torch.empty(0, dtype=torch.int64)), device=link_a.device)
    unbatched = link_a.ndim == 1
    if unbatched:
        link_a = link_a.unsqueeze(0)
        link_b = link_b.unsqueeze(0)
    if link_a.ndim != 2 or link_b.shape != link_a.shape:
        raise ValueError("contact link arrays must have shape (contacts,) or (envs, contacts)")

    force_a = torch.as_tensor(contacts.get("force_a", torch.zeros((*link_a.shape, 3))), device=link_a.device)
    if unbatched and force_a.ndim == 2:
        force_a = force_a.unsqueeze(0)
    if force_a.shape != (*link_a.shape, 3):
        raise ValueError("contact force array does not match contact link arrays")
    raw_force_b = contacts.get("force_b")
    force_b = -force_a if raw_force_b is None else torch.as_tensor(raw_force_b, device=link_a.device)
    if unbatched and force_b.ndim == 2:
        force_b = force_b.unsqueeze(0)
    if force_b.shape != force_a.shape:
        raise ValueError("contact force_b array does not match contact force_a")

    raw_valid = contacts.get("valid_mask")
    if raw_valid is None:
        valid = torch.ones_like(link_a, dtype=torch.bool)
    else:
        valid = torch.as_tensor(raw_valid, device=link_a.device, dtype=torch.bool)
        if unbatched and valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != link_a.shape:
            raise ValueError("contact valid_mask does not match contact link arrays")
    if (left_normal is None) != (right_normal is None):
        raise ValueError("left_normal and right_normal must be given together")

    def force_for(link_idx: int, normal: torch.Tensor | None) -> torch.Tensor:
        mask_a = valid & (link_a == int(link_idx))
        mask_b = valid & (link_b == int(link_idx))
        applied = torch.where(mask_a.unsqueeze(-1), force_a, torch.zeros_like(force_a))
        applied = applied + torch.where(mask_b.unsqueeze(-1), force_b, torch.zeros_like(force_b))
        net = applied.sum(dim=1)
        if normal is None:
            return torch.linalg.vector_norm(net, dim=-1)
        normal = torch.as_tensor(normal, device=net.device, dtype=net.dtype).reshape(-1, 3)
        if normal.shape[0] not in (1, net.shape[0]):
            raise ValueError("pad normal must have shape (3,) or (envs, 3)")
        return (net * normal).sum(dim=-1).abs()

    return force_for(left_link_idx, left_normal), force_for(right_link_idx, right_normal)


def validate_g2_contact_substeps(*, dt: float, substeps: int) -> float:
    """Enforce the project's conservative G2 production-step policy.

    A 0.625 ms rigid step is retained as an empirical safety margin for the
    validated production profile; the mimic solver profile is the primary fix.
    """

    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("G2 contact scene dt must be finite and positive")
    if isinstance(substeps, bool) or not isinstance(substeps, Integral):
        raise ValueError("G2 contact scene substeps must be a positive integer")
    normalized_substeps = int(substeps)
    if normalized_substeps < 1:
        raise ValueError("G2 contact scene substeps must be a positive integer")
    substep_dt = dt / normalized_substeps
    if substep_dt > G2_MAX_RIGID_SUBSTEP_DT_S + 1e-15:
        required_substeps = math.ceil(dt / G2_MAX_RIGID_SUBSTEP_DT_S)
        raise ValueError(
            f"The {G2_PHYSICS_PROFILE} project policy requires a rigid substep no longer than "
            f"0.625 ms; dt={dt:g}s with substeps={normalized_substeps} "
            f"gives {substep_dt * 1000.0:g} ms. Use at least {required_substeps} substeps."
        )
    return substep_dt


def configure_g2_mimic_constraints(robot: Any) -> int:
    """Apply and verify the one project-wide policy on all five G2 equalities."""

    sol_params = np.asarray(G2_MIMIC_CONSTRAINT_SOL_PARAMS, dtype=np.float64)
    equalities = {str(equality.name): equality for equality in robot.equalities}
    missing = sorted(G2_MIMIC_EQUALITY_NAMES - equalities.keys())
    if missing:
        raise RuntimeError(f"{G2_PHYSICS_PROFILE} is missing G2 mimic equalities: {', '.join(missing)}")
    for name in sorted(G2_MIMIC_EQUALITY_NAMES):
        equalities[name].set_sol_params(sol_params)
    return len(G2_MIMIC_EQUALITY_NAMES)


def pad_normal(link: Any) -> torch.Tensor:
    """World-frame closing normal (local +Y) of one finger link."""
    from genesis.utils.geom import transform_by_quat

    quat = link.get_quat()
    return transform_by_quat(quat.new_tensor((0, 1, 0)), quat)


def pad_centers(model: G2Model, left: Any, right: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from genesis.utils.geom import transform_by_quat

    g = model.linkage
    centers = []
    for link, y in ((left, g.pad_inner_y_m), (right, -g.pad_inner_y_m)):
        position = link.get_pos()
        point = position.new_tensor((0, y, g.pad_center_z_m))
        centers.append(position + transform_by_quat(point, link.get_quat()))
    return centers[0], centers[1]


def measured_gap_mm(model: G2Model, left: Any, right: Any, axis_link: Any = None) -> torch.Tensor:
    """Actual pad separation along the tool's closing axis, in any world pose."""
    from genesis.utils.geom import transform_by_quat

    a, b = pad_centers(model, left, right)
    # The pad normals remain parallel through the 1:1 mimic linkage.
    normal = transform_by_quat(a.new_tensor((0, 1, 0)), (axis_link or left).get_quat())
    return ((b - a) * normal).sum(dim=-1).abs() * 1000


def configure_g2_drive(robot: Any, *, damping: float = 0.05, frictionloss: float = 0.0) -> tuple[list[int], list[int]]:
    """Only the master joint is actuated; all mimic joints remain passive."""
    drive = [robot.get_joint("drive_joint").dofs_idx_local[0]]
    all_idx = [robot.get_joint(name).dofs_idx_local[0] for name in G2_JOINT_NAMES]
    state = robot.get_dofs_position(drive)
    passive = [idx for idx in all_idx if idx not in drive]
    robot.control_dofs_force(state.new_zeros((*state.shape[:-1], len(passive))), passive)
    robot.set_dofs_kp(state.new_tensor([G2_KP]), drive)
    robot.set_dofs_kv(state.new_tensor([G2_KV]), drive)
    robot.set_dofs_armature(state.new_full((len(all_idx),), G2_ARMATURE), all_idx)
    robot.set_dofs_damping(state.new_full((len(all_idx),), damping), all_idx)
    robot.set_dofs_frictionloss(state.new_full((len(all_idx),), frictionloss), all_idx)
    return drive, all_idx


class G2GenesisController:
    """Single-environment adapter; call apply before every scene.step().

    The controller runs on the CPU in float64 with ``num_envs=1``; outputs are
    ``(1,)`` tensors.
    """

    def __init__(
        self, robot: Any, left: Any, right: Any, *, initial_position_mm: float = 84, axis_link: Any = None
    ) -> None:
        self.robot, self.left, self.right = robot, left, right
        self.axis_link = axis_link
        self.model = G2Model.load()
        self.drive_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]
        self.controller = G2Controller(self.model, initial_position_mm=initial_position_mm)
        self.last_limit: float | None = None
        self.output: ControlOutput | None = None

    def gap_mm(self) -> float:
        return float(measured_gap_mm(self.model, self.left, self.right, self.axis_link).reshape(-1)[0].item())

    def reset(self, position_mm: float = 84, **kwargs: Any) -> None:
        self.controller.reset(position_mm=position_mm, **kwargs)
        self.last_limit = None

    def command(self, position_mm: float, speed_percent: float = 50, force_percent: float = 50) -> None:
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


def apply_batch_control(
    robot: Any, controller: G2Controller, drive_idx: list[int], gap_mm: torch.Tensor, dt: float
) -> ControlOutput:
    """Requires RigidOptions(batch_dofs_info=True) for independent force caps."""
    q = robot.get_dofs_position(drive_idx).squeeze(-1)
    velocity = robot.get_dofs_velocity(drive_idx).squeeze(-1)
    output = controller.update(q, velocity, dt, measured_gap_mm=gap_mm)
    limit = output.torque_limit_nm.unsqueeze(-1)
    robot.set_dofs_force_range(-limit, limit, drive_idx)
    robot.control_dofs_position_velocity(
        output.position_rad.unsqueeze(-1), output.velocity_rad_s.unsqueeze(-1), drive_idx
    )
    return output


class G2ContactCollisionBackend:
    """Check the physical grasp envelope using the unchanged collision margins.

    The safety timeline uses actual joint angles. A two-frame-confirmed grasp
    cannot close through the rigid object. Subsequent arm motion is conditional
    on that confirmation in replay_g2; missing contact stops before lift.
    """

    def __init__(self, backend, *, object_width_m: float):
        self.backend = backend
        self.model = G2Model.load()
        if not 0 < object_width_m < 0.084:
            raise ValueError("G2 contact preflight requires object width within (0, 84) mm")
        self.object_width_m = object_width_m

    def _drive(self, drive):
        if drive is None:
            gap = 0.084
        else:
            gap = gripper_g2_sim_drive_to_gap_m(drive)
        return self.model.linkage.drive_for_gap_m(max(gap, self.object_width_m))

    def check(self, q_rad, *, stage, gripper_drive=None):
        return self.backend.check(q_rad, stage=stage, gripper_drive=self._drive(gripper_drive))

    def check_all(self, q_rad, *, stage, gripper_drive=None):
        return self.backend.check_all(q_rad, stage=stage, gripper_drive=self._drive(gripper_drive))

    def check_all_within_margin(self, q_rad, *, stage, margin_m, gripper_drive=None):
        return self.backend.check_all_within_margin(
            q_rad, stage=stage, margin_m=margin_m, gripper_drive=self._drive(gripper_drive)
        )
