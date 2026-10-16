"""Genesis adapter and independent empty, sensor and cube contact scenes."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat

from ufactory.simulation.g2_confirmation import G2Telemetry
from ufactory.simulation.g2_controller import G2Controller
from ufactory.simulation.g2_model import G2Model, checked, sdk_position_counts
from ufactory.robots.paths import gripper_g2_movable_visual_urdf
from ufactory.robots.runtime import G2_GRIPPER_PARAMS
from ufactory.simulation import (
    DEFAULT_NOSLIP_ITERATIONS,
    DEFAULT_SOLVER_ITERATIONS,
    configure_g2_mimic_constraints,
    make_rigid_options,
    validate_g2_contact_substeps,
)
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface


RATE_HZ = 50
DT = 1 / RATE_HZ
SUBSTEPS = 32
GRIPPER_POS = (0.0, 0.0, 0.45)
SUPPORT_SIZE_M = (0.020, 0.012, 0.040)


def scalar(value) -> float:
    return float(torch.as_tensor(value).detach().flatten()[0].item())


def tensor(value) -> torch.Tensor:
    return torch.tensor(value, device=gs.device, dtype=gs.tc_float)


@dataclass(frozen=True)
class Sample(G2Telemetry):
    """One standalone Gripper G2 sample; adds the held object height."""

    object_z_mm: float


def write_samples(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Sample.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)


class G2Scene:
    def __init__(
        self,
        model: G2Model,
        *,
        show_viewer: bool = False,
        object_gap_mm: float | None = None,
        free_cube: bool = False,
        sensor_block: bool = False,
    ):
        if free_cube and object_gap_mm is None:
            raise ValueError("cube scene requires object_gap_mm")
        self.model = model
        self.object_gap_mm = object_gap_mm
        self.free_cube = free_cube
        validate_g2_contact_substeps(dt=DT, substeps=SUBSTEPS)
        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
            rigid_options=make_rigid_options(
                gs,
                iterations=DEFAULT_SOLVER_ITERATIONS,
                noslip_iterations=DEFAULT_NOSLIP_ITERATIONS,
                constraint_timeconst=2 * DT / SUBSTEPS,
                enable_collision=True,
                enable_joint_limit=True,
                use_gjk_collision=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=RATE_HZ,
                camera_pos=(0.38, -0.48, 0.28),
                camera_lookat=(0.0, 0.0, 0.14),
                camera_fov=40,
            ),
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        with glb_pbr_surfaces():
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=gripper_g2_movable_visual_urdf(),
                    pos=GRIPPER_POS,
                    euler=(180, 0, 0),
                    fixed=True,
                    # Numerical regularization of the soft mimic constraints;
                    # this is not an identified G2 rotor inertia.
                    default_armature=0.01,
                ),
                surface=glb_view_surface(),
            )
        self.object = None
        self.support = None
        if object_gap_mm is not None:
            width_m = checked(object_gap_mm, 1, 83, "object_gap_mm") / 1000
            size = (width_m, width_m, width_m) if free_cube and not sensor_block else (0.020, width_m, 0.030)
            self.object_height_m = size[2]
            self.object = self.scene.add_entity(
                gs.morphs.Box(size=size, pos=(0.80, 0.0, 0.20), fixed=not free_cube),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.90, 0.10, 0.10))),
            )
        if free_cube:
            self.support = self.scene.add_entity(
                gs.morphs.Box(
                    size=(SUPPORT_SIZE_M[0], min(SUPPORT_SIZE_M[1], width_m / 2), SUPPORT_SIZE_M[2]),
                    pos=(0.80, 0.0, 0.05),
                    fixed=True,
                ),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.55, 0.55, 0.55))),
            )
        self.scene.build(n_envs=1)
        configure_g2_mimic_constraints(self.robot)
        grip = G2_GRIPPER_PARAMS
        self.drive_idx = self.robot.get_joint(grip.drive_joint).dofs_idx_local[0]
        self.all_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in grip.all_joint_names]
        self.left, self.right = [self.robot.get_link(name) for name in grip.finger_link_names]
        self.robot.set_dofs_kp(tensor([40]), [self.drive_idx])
        self.robot.set_dofs_kv(tensor([5]), [self.drive_idx])
        self.robot.set_dofs_damping(tensor([grip.damping] * len(self.all_idx)), self.all_idx)
        self.robot.set_dofs_frictionloss(tensor([grip.frictionloss] * len(self.all_idx)), self.all_idx)
        self.left.set_friction(1.2)
        self.right.set_friction(1.2)
        if self.object is not None:
            self.object.set_friction(1.0)
            if free_cube:
                self.object.set_links_mass(tensor([0.017]))
        self._last_limit = None
        self.reset()

    def _pad_centers(self) -> tuple[torch.Tensor, torch.Tensor]:
        linkage = self.model.linkage
        centers = []
        for link, y in ((self.left, linkage.pad_inner_y_m), (self.right, -linkage.pad_inner_y_m)):
            point = tensor((0, y, linkage.pad_center_z_m))
            centers.append(link.get_pos() + transform_by_quat(point, link.get_quat()))
        return centers[0], centers[1]

    def gap_mm(self) -> float:
        a, b = self._pad_centers()
        return scalar((b - a)[..., 1].abs()) * 1000

    def reset(self, initial_position_mm: float = 84) -> None:
        self.controller = G2Controller(self.model, initial_position_mm)
        self.time_s = 0.0
        self._last_limit = None
        self.support_removed = False
        if self.object is not None:
            # Read pad center at the specimen's nominal contact posture.
            contact_q = self.model.linkage.drive_for_gap_m(self.object_gap_mm / 1000)
            self.robot.set_dofs_position(tensor([[contact_q] * len(self.all_idx)]), self.all_idx, zero_velocity=True)
            a, b = self._pad_centers()
            center = (0.5 * (a + b)).detach().cpu().numpy().reshape(-1, 3)[0]
            self.object.set_pos(tensor(center), zero_velocity=True)
            if self.support is not None:
                support_center = center.copy()
                support_center[2] -= self.object_height_m / 2 + SUPPORT_SIZE_M[2] / 2
                self.support.set_pos(tensor(support_center), zero_velocity=True)
        q = self.model.linkage.drive_for_gap_m(initial_position_mm / 1000)
        self.robot.set_dofs_position(tensor([[q] * len(self.all_idx)]), self.all_idx, zero_velocity=True)
        self._previous_gap_mm = self.gap_mm()

    def command(self, position_mm: float, speed_percent: float = 50, force_percent: float = 50) -> None:
        self.controller.command(position_mm, speed_percent, force_percent)

    def remove_support(self) -> None:
        if self.support is None:
            raise ValueError("scene has no support")
        position = self.support.get_pos().detach().clone()
        position[..., 2] -= 0.25
        self.support.set_pos(position, zero_velocity=True)
        self.support_removed = True

    def _contacts(self) -> tuple[float, float]:
        if self.object is None:
            return 0.0, 0.0
        contacts = self.object.get_contacts(with_entity=self.robot, exclude_self_contact=True)
        valid = contacts.get("valid_mask", torch.ones_like(contacts["link_a"], dtype=torch.bool))
        forces = []
        for link in (self.left, self.right):
            applied = torch.where(
                (valid & (contacts["link_a"] == int(link.idx))).unsqueeze(-1),
                contacts["force_a"],
                torch.zeros_like(contacts["force_a"]),
            ) + torch.where(
                (valid & (contacts["link_b"] == int(link.idx))).unsqueeze(-1),
                contacts["force_b"],
                torch.zeros_like(contacts["force_b"]),
            )
            forces.append(scalar(applied.sum(dim=-2)[..., 1].abs()))
        return forces[0], forces[1]

    def tick(self, phase: str = "hold") -> Sample:
        q = scalar(self.robot.get_dofs_position([self.drive_idx]))
        velocity = scalar(self.robot.get_dofs_velocity([self.drive_idx]))
        output = self.controller.update(q, velocity, DT, measured_gap_mm=self.gap_mm())
        if output.torque_limit_nm != self._last_limit:
            self.robot.set_dofs_force_range(
                tensor([-output.torque_limit_nm]), tensor([output.torque_limit_nm]), [self.drive_idx]
            )
            self._last_limit = output.torque_limit_nm
        self.robot.control_dofs_position_velocity(
            tensor([[output.position_rad]]), tensor([[output.velocity_rad_s]]), [self.drive_idx]
        )
        self.scene.step()
        self.time_s += DT
        q = scalar(self.robot.get_dofs_position([self.drive_idx]))
        velocity = scalar(self.robot.get_dofs_velocity([self.drive_idx]))
        gap = self.gap_mm()
        gap_velocity = (gap - self._previous_gap_mm) / DT
        self._previous_gap_mm = gap
        torque = scalar(self.robot.get_dofs_control_force([self.drive_idx]))
        left, right = self._contacts()
        object_z_mm = scalar(self.object.get_pos()[..., 2]) * 1000 if self.object is not None else float("nan")
        c = self.controller
        lookup = output.force_lookup
        return Sample(
            self.time_s,
            phase,
            c.position_mm,
            sdk_position_counts(c.position_mm),
            c.speed_percent,
            c.force_percent,
            c.requested_speed_mm_s,
            c.requested_speed_rpm,
            lookup.calibration_rpm,
            lookup.outside_calibration,
            lookup.force_n,
            self.model.static_force_n(lookup, max(0, min(84, gap))),
            output.reference_gap_mm,
            gap,
            gap_velocity,
            q,
            velocity,
            torque,
            output.torque_limit_nm,
            output.integral_torque_nm,
            self.model.motor_torque_estimate_nm(torque),
            left,
            right,
            (left + right) / 2,
            object_z_mm,
        )

    def advance(self, duration_s: float, phase: str = "hold") -> list[Sample]:
        return [self.tick(phase) for _ in range(round(duration_s * RATE_HZ))]
