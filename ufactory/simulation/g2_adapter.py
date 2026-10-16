"""Genesis joint control, pad measurements and contact preflight shared by G2 simulations."""

from __future__ import annotations

from typing import Any

import torch

from ufactory.grippers.g2 import gripper_g2_sim_drive_to_gap_m
from ufactory.simulation.g2_batch import G2BatchController
from ufactory.simulation.g2_controller import G2Controller
from ufactory.simulation.g2_model import G2Model, G2_JOINT_NAMES, G2_KP, G2_KV, G2_ARMATURE


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


def normal_contact_forces(obj: Any, robot: Any, left: Any, right: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from genesis.utils.geom import transform_by_quat

    contacts = obj.get_contacts(with_entity=robot, exclude_self_contact=True)
    valid = contacts.get("valid_mask", torch.ones_like(contacts["link_a"], dtype=torch.bool))
    forces = []
    for link in (left, right):
        mask_a = (valid & (contacts["link_a"] == int(link.idx))).unsqueeze(-1)
        mask_b = (valid & (contacts["link_b"] == int(link.idx))).unsqueeze(-1)
        applied = (
            torch.where(mask_a, contacts["force_a"], torch.zeros_like(contacts["force_a"]))
            + torch.where(mask_b, contacts["force_b"], torch.zeros_like(contacts["force_b"]))
        ).sum(dim=-2)
        normal = transform_by_quat(applied.new_tensor((0, 1, 0)), link.get_quat())
        forces.append((applied * normal).sum(dim=-1).abs())
    return forces[0], forces[1]


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
    """Single-environment adapter; call apply before every scene.step()."""

    def __init__(
        self, robot: Any, left: Any, right: Any, *, initial_position_mm: float = 84, axis_link: Any = None
    ) -> None:
        self.robot, self.left, self.right = robot, left, right
        self.axis_link = axis_link
        self.model = G2Model.load()
        self.drive_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]
        self.controller = G2Controller(self.model, initial_position_mm)
        self.last_limit: float | None = None
        self.output: Any = None

    def gap_mm(self) -> float:
        return float(measured_gap_mm(self.model, self.left, self.right, self.axis_link).reshape(-1)[0].item())

    def reset(self, position_mm: float = 84, **kwargs: Any) -> None:
        self.controller.reset(position_mm, **kwargs)
        self.last_limit = None

    def command(self, position_mm: float, speed_percent: float = 50, force_percent: float = 50) -> None:
        self.controller.command(position_mm, speed_percent, force_percent)

    def apply(self, dt: float = 0.02) -> Any:
        q = self.robot.get_dofs_position(self.drive_idx)
        velocity = self.robot.get_dofs_velocity(self.drive_idx)
        self.output = output = self.controller.update(
            float(q.reshape(-1)[0]), float(velocity.reshape(-1)[0]), dt, measured_gap_mm=self.gap_mm()
        )
        if output.torque_limit_nm != self.last_limit:
            self.robot.set_dofs_force_range(
                q.new_tensor([-output.torque_limit_nm]), q.new_tensor([output.torque_limit_nm]), self.drive_idx
            )
            self.last_limit = output.torque_limit_nm
        self.robot.control_dofs_position_velocity(
            torch.full_like(q, output.position_rad), torch.full_like(q, output.velocity_rad_s), self.drive_idx
        )
        return output


def apply_batch_control(
    robot: Any, controller: G2BatchController, drive_idx: list[int], gap_mm: torch.Tensor, dt: float
) -> Any:
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
