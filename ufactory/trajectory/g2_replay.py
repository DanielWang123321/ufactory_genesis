"""State-confirmed trajectory replay for the calibrated G2 drive."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ufactory.config.models import G2_PHYSICS_PROFILE, G2SimulationConfig
from ufactory.simulation.g2_adapter import (
    G2GenesisController,
    object_finger_contact_forces_n,
    pad_centers,
    pad_normal,
)
from ufactory.simulation.g2_model import g2_physics_summary, sdk_position_counts
from ufactory.trajectory.g2_confirmation import (
    G2Telemetry,
    GripForceGate,
    ReleaseGate,
    grip_timeout_s,
    release_timeout_s,
)


def save_sim_report(report, csv_path=None, sim_report_path=None) -> None:
    if not report.completed and (csv_path is None or sim_report_path is None):
        from datetime import datetime
        from ufactory.config.assets import discover_asset_store

        stem = discover_asset_store().root / "reports/g2/failures" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        csv_path = csv_path or stem.with_suffix(".csv")
        sim_report_path = sim_report_path or stem.with_suffix(".json")
    report.record_paths = {
        "csv": None if csv_path is None else str(csv_path),
        "sim_report": None if sim_report_path is None else str(sim_report_path),
    }
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(report.records[0]) if report.records else ["time_s", "phase"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report.records)
    if sim_report_path is not None:
        path = Path(sim_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload.pop("records")
        payload["csv"] = None if csv_path is None else str(csv_path)

        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite(item) for item in value]
            return value

        path.write_text(
            json.dumps(finite(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )


def _scalar(value) -> float:
    return float(value.detach().reshape(-1)[0].item())


def _record(ctx, adapter, output, phase: str, time_s: float, previous_gap: float, q_arm, dt: float) -> dict:
    from genesis.utils.geom import transform_by_quat

    c, model = adapter.controller, adapter.model
    lookup = c.force_lookup()
    position_mm = _scalar(c.position_mm)
    gap = adapter.gap_mm()
    q = _scalar(ctx.robot.get_dofs_position(ctx.gripper_dof_idx))
    torque = _scalar(ctx.robot.get_dofs_control_force(ctx.gripper_dof_idx))
    left, right = object_finger_contact_forces_n(
        ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True),
        left_link_idx=int(ctx.left_finger.idx),
        right_link_idx=int(ctx.right_finger.idx),
        left_normal=pad_normal(ctx.left_finger),
        right_normal=pad_normal(ctx.right_finger),
    )
    left, right = _scalar(left), _scalar(right)
    row = {
        "time_s": time_s,
        "phase": phase,
        "position_mm": position_mm,
        "position_raw_counts": sdk_position_counts(position_mm),
        "speed_percent": _scalar(c.speed_percent),
        "force_percent": _scalar(c.force_percent),
        "speed_command_mm_s": _scalar(c.requested_speed_mm_s),
        "motor_speed_command_rpm": int(_scalar(c.requested_speed_rpm)),
        "calibration_speed_rpm": lookup.calibration_rpm,
        "outside_calibration": lookup.outside_calibration,
        "curve_force_n": lookup.force_n,
        "expected_static_force_n": model.static_force_n(lookup, max(0, min(84, gap))),
        "reference_gap_mm": _scalar(output.reference_gap_mm),
        "gap_mm": gap,
        "gap_velocity_mm_s": (gap - previous_gap) / dt,
        "drive_rad": q,
        "drive_velocity_rad_s": _scalar(ctx.robot.get_dofs_velocity(ctx.gripper_dof_idx)),
        "drive_torque_nm": torque,
        "torque_limit_nm": _scalar(output.torque_limit_nm),
        "integral_torque_nm": _scalar(output.integral_torque_nm),
        "motor_torque_estimate_nm": model.motor_torque_estimate_nm(torque),
        "left_normal_n": left,
        "right_normal_n": right,
        "sensor_force_n": (left + right) / 2,
    }
    # Enforce the shared record schema before the arm-specific keys are added.
    row = asdict(G2Telemetry(**row))
    position, tool = ctx.obj.get_pos(), ctx.ik_link.get_pos()
    inverse = ctx.ik_link.get_quat().clone()
    inverse[..., 1:] *= -1
    relative = transform_by_quat(position - tool, inverse).detach().cpu().numpy().reshape(3)
    a, b = pad_centers(model, ctx.left_finger, ctx.right_finger)
    center = (a + b) / 2
    lq, rq = ctx.left_finger.get_quat(), ctx.right_finger.get_quat()
    sign = ((lq * rq).sum(dim=-1, keepdim=True) >= 0).to(lq.dtype) * 2 - 1
    quat = lq + sign * rq
    quat = quat / quat.norm(dim=-1, keepdim=True)
    quat[..., 1:] *= -1
    pad_relative = transform_by_quat(position - center, quat).detach().cpu().numpy().reshape(3)
    pad_in_tool = transform_by_quat(center - tool, inverse).detach().cpu().numpy().reshape(3)
    object_base = ctx.world_to_base(position.detach().cpu().numpy().reshape(3))
    tool_base = ctx.world_to_base(tool.detach().cpu().numpy().reshape(3))
    for name, values in (
        ("object_base", object_base),
        ("tool_base", tool_base),
        ("object_in_tool", relative),
        ("object_in_pad", pad_relative),
        ("pad_in_tool", pad_in_tool),
    ):
        row.update({f"{name}_{axis}_mm": float(values[i] * 1000) for i, axis in enumerate("xyz")})
    row["object_z_mm"] = _scalar(position[..., 2]) * 1000
    row["object_speed_mm_s"] = _scalar(ctx.obj.get_vel().norm(dim=-1)) * 1000
    row["home_error_mm"] = float(np.linalg.norm(np.asarray(tool_base) - ctx.home_pos_base) * 1000)
    actual = ctx.robot.get_dofs_position(ctx.arm_dof_idx).detach().cpu().numpy().reshape(-1)
    for i, (target, measured) in enumerate(zip(q_arm, actual, strict=True), start=1):
        row[f"arm_target_q{i}_rad"], row[f"arm_q{i}_rad"] = float(target), float(measured)
    return row


def replay_g2(
    program, ctx, *, final_metric_settle_s=0.04, on_phase=None, on_tick=None, csv_path=None, sim_report_path=None
):
    from ufactory.trajectory.g2_confirmation import summarize
    from ufactory.trajectory.mirror_executor import resolve_tick_arm_q
    from ufactory.trajectory.sim_executor import PhaseStatus, SimReport, _assert_finite_replay_state

    if not math.isfinite(final_metric_settle_s) or final_metric_settle_s < 0:
        raise ValueError("final_metric_settle_s must be finite and non-negative")
    settings = ctx.g2_settings or G2SimulationConfig()
    if settings.speed_percent <= 0:
        raise ValueError("automatic G2 manipulation requires speed_percent > 0")
    adapter = ctx.g2_controller
    if adapter is None:
        adapter = ctx.g2_controller = G2GenesisController(
            ctx.robot, ctx.left_finger, ctx.right_finger, axis_link=ctx.ik_link
        )
    adapter.reset(max(0, min(84, adapter.gap_mm())))
    adapter.command(84, settings.speed_percent, settings.force_percent)
    report = SimReport(physics_profile=G2_PHYSICS_PROFILE, physics_parameters=g2_physics_summary())
    report.physics_parameters["command"] = asdict(settings)
    report.physics_parameters["rate_hz"] = program.rate
    report.physics_parameters.update(getattr(ctx, "simulation_parameters", {}))
    rate, dt = float(program.rate), 1 / float(program.rate)
    q_arm = ctx.home_qpos[ctx.arm_dof_idx].astype(np.float64).copy()
    previous_gap = adapter.gap_mm()
    pending_confirmation = None

    def tick(segment, tick_index):
        nonlocal previous_gap, pending_confirmation
        actual_q = ctx.robot.get_dofs_position(ctx.arm_dof_idx)
        ctx.robot.control_dofs_position(actual_q.new_tensor(q_arm).reshape(actual_q.shape), ctx.arm_dof_idx)
        output = adapter.apply(dt)
        ctx.scene.step()
        _assert_finite_replay_state(ctx, phase=segment.label, tick=tick_index)
        row = _record(
            ctx,
            adapter,
            output,
            segment.label or segment.kind,
            round((len(report.records) + 1) * dt, 12),
            previous_gap,
            q_arm,
            dt,
        )
        previous_gap = row["gap_mm"]
        report.records.append(row)
        if pending_confirmation is not None:
            confirmation = report.confirmations[pending_confirmation]
            confirmation["next_phase"] = row["phase"]
            confirmation["next_action_time_s"] = row["time_s"]
            confirmation["next_action_delay_s"] = round(row["time_s"] - confirmation["confirmed_time_s"], 12)
            pending_confirmation = None
        if on_tick is not None:
            on_tick(segment, tick_index)
        return row

    try:
        for si, segment in enumerate(program.segments):
            samples, count = segment.samples(rate)
            if count == 0:
                continue
            status = PhaseStatus(segment.label or segment.kind, si, 0, 0, segment.v_max, segment.a_max)
            gate = None
            if segment.kind == "gripper" and segment.label in {"grip", "release"}:
                reference = _scalar(adapter.controller.reference_gap_mm)
                if segment.label == "grip":
                    deadline = grip_timeout_s(reference, settings.position_mm, settings.speed_percent)
                    gate = GripForceGate(rate)
                    target = settings.position_mm
                else:
                    deadline = release_timeout_s(reference, settings.speed_percent)
                    gate = ReleaseGate(rate)
                    target = 84
                # The approved duration is the maximum reference stroke deadline.
                # Reversal can only reduce the remaining reference travel.
                count = int(math.ceil(min(deadline, segment.duration) * rate - 1e-9))
                adapter.command(target, settings.speed_percent, settings.force_percent)
            for t in range(count):
                if segment.kind != "gripper":
                    started = time.perf_counter_ns()
                    try:
                        q_arm = resolve_tick_arm_q(ctx, segment, samples, t, init_qpos=q_arm)
                    except Exception:
                        status.ik_failures += 1
                        raise
                    finally:
                        if segment.kind == "movel":
                            status.ik_duration_ms += (time.perf_counter_ns() - started) / 1e6
                row = tick(segment, t)
                status.ticks += 1
                if gate is not None and gate.observe(row):
                    report.confirmations[segment.label] = {
                        "confirmed_time_s": row["time_s"],
                        "deadline_s": count / rate,
                        "samples": [
                            {
                                key: item[key]
                                for key in (
                                    "time_s",
                                    "gap_mm",
                                    "expected_static_force_n",
                                    "left_normal_n",
                                    "right_normal_n",
                                )
                            }
                            for item in report.records[-2:]
                        ],
                    }
                    pending_confirmation = segment.label
                    break
            status.duration = status.ticks / rate
            status.obj_pos_mm = tuple(float(v) * 1000 for v in ctx.obj.get_pos().detach().cpu().numpy().reshape(3))
            status.link6_mm = tuple(float(v) * 1000 for v in ctx.ik_link.get_pos().detach().cpu().numpy().reshape(3))
            if segment.kind == "movel" and segment.pose_end is not None:
                status.eside_arm_mm = float(
                    np.linalg.norm(np.asarray(status.link6_mm) - np.asarray(ctx.base_to_world(segment.pose_end)) * 1000)
                )
            report.phases.append(status)
            if on_phase is not None:
                on_phase(status, segment)
            if gate is not None and gate.confirmed_time_s is None:
                report.completed = False
                report.timeout_phase = segment.label
                report.failure_reason = f"{segment.label} confirmation timeout after {status.duration:.2f} s"
                break
        if report.completed:
            from ufactory.trajectory.segments import Segment

            settle = Segment("gripper", final_metric_settle_s, 0, 0, label="final-settle")
            for t in range(int(math.ceil(final_metric_settle_s * rate))):
                tick(settle, t)
                report.metric_settle_ticks += 1
    except Exception as exc:
        report.completed = False
        report.failure_reason = f"{type(exc).__name__}: {exc}"

    report.total_ticks = len(report.records)
    report.total_duration = report.total_ticks / rate
    report.place_error_mm = float(
        np.linalg.norm(ctx.obj.get_pos().detach().cpu().numpy().reshape(3) - ctx.base_to_world(ctx.place_pos_base))
        * 1000
    )
    report.home_drift_mm = float(
        np.linalg.norm(ctx.ik_link.get_pos().detach().cpu().numpy().reshape(3) - ctx.base_to_world(ctx.home_pos_base))
        * 1000
    )
    if report.records:
        config = SimpleNamespace(
            motion=SimpleNamespace(rate_hz=rate),
            task=SimpleNamespace(
                parameters={
                    "fixed_target_position_m": ctx.place_pos_base,
                    "fixed_object_position_m": ctx.obj_pos_base,
                    "object_size_m": ctx.obj_size,
                }
            ),
        )
        packaging = any(s.label == "post-release-settle" for s in program.segments)
        report.acceptance = summarize(report.records, config, packaging=packaging)
        report.timing_passed = report.completed and report.acceptance["timing_passed"]
        report.transport_passed = report.completed and report.acceptance["transport_passed"]
    else:
        report.timing_passed = report.transport_passed = False
    save_sim_report(report, csv_path, sim_report_path)
    return report
