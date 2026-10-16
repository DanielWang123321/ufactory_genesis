"""Standalone Gripper G2 acceptance: empty, sensor-block and free-cube scenes.

Used by ``verify.py`` (33-case suite) and ``grasp_cube.py`` (free-cube grasp
with support removal). The G2 model, controller and Genesis integration live in
``ufactory.simulation``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from importlib import metadata
import json
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np
import torch

from ufactory.robots.paths import gripper_g2_movable_visual_urdf
from ufactory.robots.runtime import G2_GRIPPER_PARAMS
from ufactory.simulation import (
    DEFAULT_NOSLIP_ITERATIONS,
    DEFAULT_SOLVER_ITERATIONS,
    genesis_backend_constant,
    make_rigid_options,
)
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.simulation.g2_adapter import (
    configure_g2_mimic_constraints,
    object_finger_contact_forces_n,
    pad_centers,
    validate_g2_contact_substeps,
)
from ufactory.simulation.g2_controller import G2Controller
from ufactory.simulation.g2_model import (
    G2Model,
    checked,
    g2_physics_summary,
    sdk_position_counts,
    sdk_speed_rpm,
    speed_mm_s,
)
from ufactory.trajectory.g2_confirmation import G2Telemetry
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

RATE_HZ = 50
DT = 1 / RATE_HZ
SUBSTEPS = 32
GRIPPER_POS = (0.0, 0.0, 0.45)
SUPPORT_SIZE_M = (0.020, 0.012, 0.040)
WORLD_Y = torch.tensor((0.0, 1.0, 0.0))


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
        return pad_centers(self.model, self.left, self.right)

    def gap_mm(self) -> float:
        # The fixed standalone gripper closes along world Y.
        a, b = self._pad_centers()
        return scalar((b - a)[..., 1].abs()) * 1000

    def reset(self, initial_position_mm: float = 84) -> None:
        self.controller = G2Controller(self.model, initial_position_mm=initial_position_mm)
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
        left, right = object_finger_contact_forces_n(
            self.object.get_contacts(with_entity=self.robot, exclude_self_contact=True),
            left_link_idx=int(self.left.idx),
            right_link_idx=int(self.right.idx),
            # The fixed standalone gripper closes along world Y.
            left_normal=WORLD_Y,
            right_normal=WORLD_Y,
        )
        return scalar(left), scalar(right)

    def tick(self, phase: str = "hold") -> Sample:
        q = scalar(self.robot.get_dofs_position([self.drive_idx]))
        velocity = scalar(self.robot.get_dofs_velocity([self.drive_idx]))
        output = self.controller.update(q, velocity, DT, measured_gap_mm=self.gap_mm())
        limit = scalar(output.torque_limit_nm)
        if limit != self._last_limit:
            self.robot.set_dofs_force_range(tensor([-limit]), tensor([limit]), [self.drive_idx])
            self._last_limit = limit
        self.robot.control_dofs_position_velocity(
            tensor([[scalar(output.position_rad)]]), tensor([[scalar(output.velocity_rad_s)]]), [self.drive_idx]
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
        lookup = c.force_lookup()
        position_mm = scalar(c.position_mm)
        return Sample(
            self.time_s,
            phase,
            position_mm,
            sdk_position_counts(position_mm),
            scalar(c.speed_percent),
            scalar(c.force_percent),
            scalar(c.requested_speed_mm_s),
            int(scalar(c.requested_speed_rpm)),
            lookup.calibration_rpm,
            lookup.outside_calibration,
            lookup.force_n,
            self.model.static_force_n(lookup, max(0, min(84, gap))),
            scalar(output.reference_gap_mm),
            gap,
            gap_velocity,
            q,
            velocity,
            torque,
            limit,
            scalar(output.integral_torque_nm),
            self.model.motor_torque_estimate_nm(torque),
            left,
            right,
            (left + right) / 2,
            object_z_mm,
        )

    def advance(self, duration_s: float, phase: str = "hold") -> list[Sample]:
        return [self.tick(phase) for _ in range(round(duration_s * RATE_HZ))]


# ---------------------------------------------------------------------------
# Acceptance cases and command-line entries
# ---------------------------------------------------------------------------


def summarize(samples, *, support_removed: bool, release_z_mm: float | None) -> dict:
    hold = [sample for sample in samples if sample.phase == "hold"]
    steady = hold[-50:]
    force = [sample.sensor_force_n for sample in steady]
    expected = steady[-1].expected_static_force_n if steady else 0.0
    stable_force = sum(force) / len(force) if force else 0.0
    drop = release_z_mm - hold[-1].object_z_mm if hold and release_z_mm is not None else None
    held = bool(support_removed and hold and drop < 15 and min(hold[-1].left_normal_n, hold[-1].right_normal_n) > 0.2)
    return {
        "held": held,
        "support_removed": support_removed,
        "steady_window_s": 1.0,
        "stable_force_n": stable_force,
        "stable_force_min_n": min(force, default=0.0),
        "stable_force_max_n": max(force, default=0.0),
        "expected_static_force_n": expected,
        "stable_force_error_n": abs(stable_force - expected),
        "contact_peak_n": max((sample.sensor_force_n for sample in samples), default=0.0),
        "left_contact_peak_n": max((sample.left_normal_n for sample in samples), default=0.0),
        "right_contact_peak_n": max((sample.right_normal_n for sample in samples), default=0.0),
        "drive_torque_peak_nm": max((abs(sample.drive_torque_nm) for sample in samples), default=0.0),
        "motor_torque_estimate_peak_nm": max((abs(sample.motor_torque_estimate_nm) for sample in samples), default=0.0),
        "drop_mm": drop,
    }


def grasp_cube_samples(
    model: G2Model,
    *,
    position_mm: float = 0,
    speed_percent: float = 50,
    force_percent: float = 50,
    cube_size_mm: float = 30,
    show_viewer: bool = False,
    timeout_s: float | None = None,
) -> tuple[Any, list[Sample], dict]:
    """Close on a free cube, remove the support after a held contact, then hold."""

    simulation = G2Scene(model, show_viewer=show_viewer, object_gap_mm=cube_size_mm, free_cube=True)
    simulation.command(84, speed_percent, force_percent)
    samples = simulation.advance(0.4, "settle")
    simulation.command(position_mm, speed_percent, force_percent)
    if timeout_s is None:
        speed = speed_mm_s(speed_percent)
        timeout_s = max(3.0, 84 / speed + 2.0) if speed else 3.0
    stable_steps = 0
    release_z_mm = None
    for _ in range(round(timeout_s / DT)):
        sample = simulation.tick("close")
        samples.append(sample)
        bilateral = min(sample.left_normal_n, sample.right_normal_n) >= 0.2
        stable_steps = stable_steps + 1 if bilateral else 0
        if stable_steps * DT >= 0.4:
            release_z_mm = sample.object_z_mm
            simulation.remove_support()
            print(
                f"撤支撑 t={sample.time_s:.2f} s，较弱法向力 {min(sample.left_normal_n, sample.right_normal_n):.2f} N",
                flush=True,
            )
            break
    if simulation.support_removed:
        samples.extend(simulation.advance(3.0, "hold"))
    result = summarize(samples, support_removed=simulation.support_removed, release_z_mm=release_z_mm)
    return simulation, samples, result


def run(
    *,
    show_viewer: bool,
    model: G2Model,
    position_mm: float = 0,
    speed_percent: float = 50,
    force_percent: float = 50,
    cube_size_mm: float = 30,
    csv_path: Path | None = None,
) -> dict:
    simulation, samples, result = grasp_cube_samples(
        model,
        position_mm=position_mm,
        speed_percent=speed_percent,
        force_percent=force_percent,
        cube_size_mm=cube_size_mm,
        show_viewer=show_viewer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if csv_path is not None:
        write_samples(csv_path, samples)
        summary_path = csv_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"记录：{csv_path}")
    if show_viewer:
        print("保持位置指令，Ctrl+C 结束。")
        try:
            while True:
                simulation.tick("hold")
        except KeyboardInterrupt:
            pass
    return result


def force_result(samples, *, reference=False) -> dict:
    steady = samples[-50:]
    expected = steady[-1].curve_force_n if reference else steady[-1].expected_static_force_n
    force = np.array([s.sensor_force_n for s in steady])
    error = abs(float(force.mean()) - expected)
    tolerance = max(2.0, 0.05 * expected)
    torque_ok = all(abs(s.drive_torque_nm) <= s.torque_limit_nm + 1e-5 for s in samples)
    bilateral = all(min(s.left_normal_n, s.right_normal_n) > 0.2 for s in steady)
    return {
        "kind": "force",
        "passed": error <= tolerance and torque_ok and bilateral,
        "speed_percent": steady[-1].speed_percent,
        "force_percent": steady[-1].force_percent,
        "stable_force_n": float(force.mean()),
        "stable_force_min_n": float(force.min()),
        "stable_force_max_n": float(force.max()),
        "stable_force_std_n": float(force.std()),
        "stable_left_normal_n": float(np.mean([s.left_normal_n for s in steady])),
        "stable_right_normal_n": float(np.mean([s.right_normal_n for s in steady])),
        "bilateral_contact": bilateral,
        "expected_force_n": expected,
        "force_error_n": error,
        "force_tolerance_n": tolerance,
        "contact_peak_n": max(s.sensor_force_n for s in samples),
        "left_contact_peak_n": max(s.left_normal_n for s in samples),
        "right_contact_peak_n": max(s.right_normal_n for s in samples),
        "gap_mm": steady[-1].gap_mm,
        "torque_within_limit": torque_ok,
        "torque_peak_nm": max(abs(s.drive_torque_nm) for s in samples),
        "calibration_speed_rpm": steady[-1].calibration_speed_rpm,
        "outside_calibration": steady[-1].outside_calibration,
    }


def free_cases(model, save):
    simulation = G2Scene(model)
    for speed in (10, 50, 100):
        simulation.reset()
        simulation.command(84, speed, 100)
        simulation.advance(0.5, "settle")
        for direction, target in (("close", 0), ("open", 84)):
            simulation.command(target, speed, 100)
            samples = simulation.advance(84 / speed_mm_s(speed) + 1.5, direction)
            central = [s for s in samples if 25 <= s.gap_mm <= 65]
            measured_speed = (
                abs(float(np.polyfit([s.time_s for s in central], [s.gap_mm for s in central], 1)[0]))
                if len(central) >= 3
                else 0.0
            )
            speed_error = abs(measured_speed / speed_mm_s(speed) - 1)
            central_peak_error = max(
                (abs(abs(s.gap_velocity_mm_s) / speed_mm_s(speed) - 1) for s in central), default=1.0
            )
            position_error = abs(samples[-1].gap_mm - target)
            save(
                f"free_s{speed}_{direction}",
                samples,
                {
                    "kind": "free",
                    "passed": speed_error <= 0.10 and central_peak_error <= 0.10 and position_error <= 1,
                    "speed_percent": speed,
                    "position_mm": target,
                    "measured_speed_mm_s": measured_speed,
                    "requested_speed_mm_s": speed_mm_s(speed),
                    "speed_relative_error": speed_error,
                    "central_speed_max_relative_error": central_peak_error,
                    "position_error_mm": position_error,
                    "gap_mm": samples[-1].gap_mm,
                    "outside_calibration": samples[-1].outside_calibration,
                },
            )
    simulation.reset()
    simulation.command(0, 50, 100)
    before = simulation.advance(0.3, "moving")
    simulation.command(0, 0, 100)
    paused = simulation.advance(1.5, "pause")
    pause_drift = max(abs(s.gap_mm - before[-1].gap_mm) for s in paused)
    reference_frozen = max(s.reference_gap_mm for s in paused) - min(s.reference_gap_mm for s in paused) <= 1e-9
    stopped = abs(paused[-1].gap_velocity_mm_s) <= 1.0
    simulation.command(0, 50, 100)
    resumed = simulation.advance(2.0, "resume")
    save(
        "free_pause_resume",
        before + paused + resumed,
        {
            "kind": "behavior",
            "passed": reference_frozen and stopped and resumed[-1].gap_mm <= 1,
            "pause_drift_mm": pause_drift,
            "position_error_mm": resumed[-1].gap_mm,
            "reference_frozen": reference_frozen,
            "pause_final_velocity_mm_s": paused[-1].gap_velocity_mm_s,
        },
    )


def reference_cases(model, save):
    # Let the specimen center itself along Y; a fixed specimen would absorb
    # unequal lateral reactions, despite having the same average force.
    simulation = G2Scene(model, object_gap_mm=model.reference_gap_mm, free_cube=True, sensor_block=True)
    for speed in (10, 50, 90):
        for force in (10, 30, 50, 80, 100):
            simulation.reset()
            simulation.command(84, speed, force)
            simulation.advance(0.2, "settle")
            simulation.command(0, speed, force)
            samples = simulation.advance(84 / speed_mm_s(speed) + 3.0, "hold")
            save(f"reference_s{speed}_f{force}", samples, force_result(samples, reference=True))
    simulation.reset()
    simulation.command(0, 50, 30)
    samples = simulation.advance(4.0, "close")
    for force in (80, 10):
        simulation.command(0, 50, force)
        changed = simulation.advance(2.0, "force_change")
        samples.extend(changed)
        save(f"change_force_f{force}", changed, force_result(changed, reference=True))
    simulation.command(0, 0, 10)
    paused = simulation.advance(1.5, "pause")
    samples.extend(paused)
    result = force_result(paused, reference=True)
    result["pause_drift_mm"] = abs(paused[-1].gap_mm - paused[0].gap_mm)
    result["passed"] = result["passed"] and result["pause_drift_mm"] <= 1 and paused[-1].motor_speed_command_rpm == 0
    save("grasp_pause", paused, result)
    simulation.command(0, 0, 80)
    changed = simulation.advance(2.0, "paused_force_change")
    samples.extend(changed)
    save("paused_change_force_f80", changed, force_result(changed, reference=True))
    simulation.command(84, 0, 80)
    queued = simulation.advance(1.5, "paused_open_command")
    samples.extend(queued)
    result = force_result(queued, reference=True)
    result["reference_frozen"] = (
        max(s.reference_gap_mm for s in queued) - min(s.reference_gap_mm for s in queued) <= 1e-9
    )
    result["passed"] = result["passed"] and result["reference_frozen"]
    save("paused_position_change", queued, result)
    simulation.command(0, 50, 50)
    resumed = simulation.advance(1.5, "resume")
    samples.extend(resumed)
    save("grasp_resume", resumed, force_result(resumed, reference=True))
    simulation.command(84, 50, 50)
    opened = simulation.advance(2.0, "open")
    samples.extend(opened)
    save(
        "grasp_reverse_open",
        samples,
        {
            "kind": "behavior",
            "passed": abs(opened[-1].gap_mm - 84) <= 1 and opened[-1].sensor_force_n <= 0.2,
            "position_error_mm": abs(opened[-1].gap_mm - 84),
            "stable_force_n": opened[-1].sensor_force_n,
        },
    )


def thickness_cases(model, save):
    for thickness in (10, 50, 70):
        simulation = G2Scene(model, object_gap_mm=thickness, free_cube=True, sensor_block=True)
        simulation.command(0, 50, 80)
        samples = simulation.advance(4.0, "hold")
        result = force_result(samples)
        result["object_gap_mm"] = thickness
        save(f"thickness_{thickness}mm", samples, result)


def cube_case(model, save):
    """Free-cube grasp with support removal, on the same path as the grasp CLI."""
    _, samples, result = grasp_cube_samples(model, timeout_s=4.0)
    result["kind"] = "cube"
    result["passed"] = result["held"] and result["stable_force_error_n"] <= max(
        2, result["expected_static_force_n"] * 0.05
    )
    save("cube_30mm_support_removed", samples, result)


def grasp_main() -> None:
    parser = argparse.ArgumentParser(description="Gripper G2 百分比控制，夹持方块后撤支撑")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--position-mm", type=float, default=0, help="指垫间隙目标，0–84 mm")
    parser.add_argument("--speed-percent", type=float, default=50, help="0–100%%；0 暂停位置进给")
    parser.add_argument("--force-percent", type=float, default=50, help="1–100%%，按官方曲线限制关节力矩")
    parser.add_argument("--cube-size-mm", type=float, default=30)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    try:
        checked(args.position_mm, 0, 84, "position_mm")
        speed = speed_mm_s(args.speed_percent)
        checked(args.force_percent, 1, 100, "force_percent")
        checked(args.cube_size_mm, 1, 83, "cube_size_mm")
    except ValueError as exc:
        parser.error(str(exc))
    model = G2Model.load()
    rpm = sdk_speed_rpm(speed)
    calibration_rpm = rpm or sdk_speed_rpm(speed_mm_s(50))
    lookup = model.curve.lookup(calibration_rpm, args.force_percent)
    print(
        f"位置 {args.position_mm:g} mm，速度 {args.speed_percent:g}% → {speed:g} mm/s / {rpm} rpm，力 {args.force_percent:g}%"
    )
    print(
        f"参考原始位置 230 → {model.reference_gap_mm:.6f} mm（SDK 源码推断）；标定稳定力 {lookup.force_n:.2f} N，关节限幅 ±{model.torque_limit_nm(lookup):.3f} N·m"
    )
    if lookup.outside_calibration:
        print("转速超出标定范围：力标定使用 3500 rpm 边界曲线，运动速度按输入执行。")
    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, args.backend), logging_level="error")
    result = run(
        show_viewer=not args.headless,
        model=model,
        position_mm=args.position_mm,
        speed_percent=args.speed_percent,
        force_percent=args.force_percent,
        cube_size_mm=args.cube_size_mm,
        csv_path=args.csv,
    )
    if args.headless:
        raise SystemExit(0 if result["held"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 百分比控制仿真验收")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--suite", choices=("all", "free", "reference", "thickness", "cube"), default="all")
    parser.add_argument("--out", type=Path, default=Path("reports/g2/validation"))
    args = parser.parse_args()
    model = G2Model.load()
    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, args.backend), logging_level="error")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend == "gpu" and str(gs.device) == "cpu":
        raise RuntimeError("GPU acceptance cannot fall back to CPU")

    (args.out / "run.json").write_text(
        json.dumps(
            {
                "genesis_version": metadata.version("genesis-world"),
                "device": str(gs.device),
                "requested_backend": args.backend,
                "physics": g2_physics_summary(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results = []

    def save(name, samples, result):
        result["case"] = name
        result["csv"] = f"{name}.csv"
        write_samples(args.out / result["csv"], samples)
        results.append(result)
        (args.out / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fields = sorted({key for row in results for key in row})
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    suites = {"free": free_cases, "reference": reference_cases, "thickness": thickness_cases, "cube": cube_case}
    for name, suite in suites.items():
        if args.suite in ("all", name):
            suite(model, save)
    passed = all(row["passed"] for row in results)
    print(f"验收：{sum(row['passed'] for row in results)}/{len(results)} 通过；记录 {args.out}", flush=True)
    raise SystemExit(0 if passed else 1)
