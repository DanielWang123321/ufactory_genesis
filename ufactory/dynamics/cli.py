"""Dynamics validation CLI entry points.

Console scripts point here (``ufactory.dynamics.cli:cli_*``). The heavy
real-robot / kinematics stack is imported lazily inside each entry so that
``import ufactory.dynamics.cli`` does not require the xArm SDK or a robot.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ufactory.dynamics.probe import (
    build_genesis_scene,
    read_joint_frictionloss,
    compute_ee_z_table_from_sim,
    genesis_pd_hold_torque_at_q,
    check_genesis_path_z,
)
from ufactory.dynamics.plot import write_torque_plot
from ufactory.dynamics.reference import load_reference_backend
from ufactory.dynamics.analysis import (
    STATIC_LAYERS,
    StaticPoseAnalysis,
    build_dynamics_sample,
    build_static_pose_analysis,
    format_torque_row,
    parse_strict_static_layers,
    static_layer_l2,
    summarize_static_layers,
    validate_urdf_dynamics,
)
from ufactory.dynamics.report import (
    DynamicsRunConfig,
    DynamicsSample,
    TorqueCompareResult,
    ValidationStatus,
    compare_report_records,
    dynamics_report_paths,
    joint_torque_rows,
    make_run_config,
    now_stamp,
    read_report_records,
    torque_summary,
    write_csv_report,
    write_jsonl_report,
)
from ufactory.dynamics.poses import (
    check_joint_limit_path,
    dynamics_default_configs,
    filter_safe_configs,
    merge_test_configs,
    parse_joint_limits,
    SafePose,
)
from ufactory.paths import robot_urdf
from ufactory.robot_params import get_robot_runtime_profile, robot_runtime_cli_choices


def _select_poses(
    configs: Sequence[tuple[str, np.ndarray]],
    names_csv: str | None,
) -> tuple[list[tuple[str, np.ndarray]], set[str]]:
    if not names_csv:
        return list(configs), set()
    wanted = {p.strip() for p in names_csv.split(",") if p.strip()}
    selected = [(name, q) for name, q in configs if name in wanted]
    missing = wanted - {name for name, _ in selected}
    return selected, missing


def print_safe_pose_table(safe_poses: Sequence[SafePose], rejected: Sequence[tuple[str, float]]) -> None:
    print(f"\n{'Name':<20} {'EE z (mm)':>12}  Status")
    print("-" * 48)
    for pose in safe_poses:
        print(f"{pose.name:<20} {pose.ee_z_mm:>12.2f}  SAFE")
    for name, ee_z in rejected:
        z_str = f"{ee_z:>12.2f}" if np.isfinite(ee_z) else f"{'n/a':>12}"
        print(f"{name:<20} {z_str}  REJECTED (z < z_min)")


def _display_float(value: object, width: int = 10, precision: int = 3) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except Exception:
        return f"{'n/a':>{width}}"
    if not np.isfinite(v):
        return f"{'n/a':>{width}}"
    return f"{v:{width}.{precision}f}"


def print_compare_table(
    results: Sequence[DynamicsSample | TorqueCompareResult],
    *,
    runtime_profile: Any | None = None,
) -> None:
    print("\n--- Torque validation summary ---")
    print("torque_l2_err_nm = sqrt(sum((genesis_tau_nm - sdk_tau_mean_nm)^2))")
    print(
        f"{'Pose':<20} {'EE z mm':>8} {'Status':<18} {'torque_l2':>10} "
        f"{'Worst':>6} {'abs Nm':>9} {'rel':>7} {'N':>4} {'Warn':>4}"
    )
    print("-" * 96)
    for r in results:
        summary = torque_summary(r, runtime_profile=runtime_profile)
        pose_name = r.pose if isinstance(r, DynamicsSample) else r.name
        print(
            f"{pose_name:<20} {r.ee_z_mm:8.1f} {summary['status']:<18} "
            f"{_display_float(summary['torque_l2_err_nm'], width=10)} "
            f"{str(summary['worst_joint'] or '-'):>6} "
            f"{_display_float(summary['worst_abs_err_nm'], width=9)} "
            f"{_display_float(summary['worst_rel_err'], width=7, precision=4)} "
            f"{summary['n_real_samples']:4d} {summary['warning_count']:4d}"
        )

    failures = [
        r
        for r in results
        if torque_summary(r, runtime_profile=runtime_profile)["status"] != ValidationStatus.PASS.value
    ]
    if not failures:
        return

    print("\n--- Non-PASS pose details ---")
    for r in failures:
        summary = torque_summary(r, runtime_profile=runtime_profile)
        pose_name = r.pose if isinstance(r, DynamicsSample) else r.name
        reason = summary["status_reason"] or "non-PASS"
        print(f"\n[{pose_name}] {summary['status']}: {reason}")
        print(
            f"{'Joint':<5} {'genesis_tau_nm':>15} {'sdk_tau_mean_nm':>16} "
            f"{'sdk_tau_std_nm':>15} {'abs_err_nm':>12} {'rel_err':>9} "
            f"{'abs_limit_nm':>13} {'rel_limit':>9}"
        )
        print("-" * 104)
        rows = joint_torque_rows(r, runtime_profile=runtime_profile)
        printed = False
        for row in rows:
            if row["joint_status"] == "PASS" and summary["status"] in {ValidationStatus.FAIL_BIAS.value, ValidationStatus.FAIL_MODEL.value}:
                continue
            printed = True
            print(
                f"{row['joint']:<5} "
                f"{_display_float(row['genesis_tau_nm'], width=15, precision=6)} "
                f"{_display_float(row['sdk_tau_mean_nm'], width=16, precision=6)} "
                f"{_display_float(row['sdk_tau_std_nm'], width=15, precision=6)} "
                f"{_display_float(row['abs_err_nm'], width=12, precision=6)} "
                f"{_display_float(row['rel_err'], width=9, precision=4)} "
                f"{_display_float(row['abs_limit_nm'], width=13, precision=3)} "
                f"{_display_float(row['rel_limit'], width=9, precision=3)}"
            )
        if not printed and isinstance(r, DynamicsSample) and r.notes:
            for note in r.notes[:3]:
                print(f"  note: {note}")


def _static_analyses_from_results(results: Sequence[DynamicsSample]) -> list[StaticPoseAnalysis]:
    analyses: list[StaticPoseAnalysis] = []
    for result in results:
        if not isinstance(result, DynamicsSample):
            continue
        if not result.static_layers and not result.static_warnings:
            continue
        analyses.append(
            StaticPoseAnalysis(
                pose=result.pose,
                layers=list(result.static_layers),
                clamp_slack_est=result.clamp_slack_est,
                armature=result.armature,
                reference_mass_matrix=result.reference_mass_matrix,
                mass_rel_fro=result.l3a_mass_rel_fro,
                warnings=list(result.static_warnings),
            )
        )
    return analyses


def print_static_layer_summary(
    results: Sequence[DynamicsSample],
    *,
    strict_layers: frozenset[str] | None = None,
) -> tuple[dict[str, dict[str, int]], bool]:
    analyses = _static_analyses_from_results(results)
    if not analyses:
        print("\n--- Static layer summary (L2/L3) ---")
        print("  (no static layer data)")
        return {layer: {"pass": 0, "warn": 0, "fail": 0} for layer in STATIC_LAYERS}, False

    counts, strict_fail = summarize_static_layers(analyses, strict_layers=strict_layers or frozenset())
    print("\n--- Static layer summary (L2/L3) ---")
    print("  L2a/L2b/L3a/L3b are diagnostics; they affect Overall only with --strict-static.")
    print("  L2a/L2b/L3b are Nm L2 residuals; L3a is mass-matrix relative Frobenius error.")
    for layer in STATIC_LAYERS:
        bucket = counts.get(layer, {"pass": 0, "warn": 0, "fail": 0})
        print(
            f"  {layer}: pass={bucket.get('pass', 0)} "
            f"warn={bucket.get('warn', 0)} fail={bucket.get('fail', 0)}"
        )
    if strict_layers:
        print(f"  strict layers: {','.join(sorted(strict_layers))} -> {'FAIL' if strict_fail else 'PASS'}")
    return counts, strict_fail


def _run_genesis_samples(
    scene,
    robot,
    ee_link,
    dof_idx,
    safe_poses: Sequence[SafePose],
    reference: Any,
    runtime_profile: Any | None = None,
):
    runtime = runtime_profile or get_robot_runtime_profile("xarm6")
    out: dict[str, Any] = {}
    for pose in safe_poses:
        sample = genesis_pd_hold_torque_at_q(robot, scene, dof_idx, pose.q, runtime_profile=runtime)
        out[pose.name] = sample
        status = "settled" if sample.settled else "NOT settled"
        sat = " SATURATED" if sample.saturated else ""
        print(f"  [{pose.name}] pd_hold_tau={format_torque_row(sample.pd_hold_tau)}  ({status}{sat})")
        if reference is not None:
            try:
                g_ref = reference.gravity(sample.q_actual)
                print(f"    pinocchio_G={format_torque_row(g_ref)}")
                static = build_static_pose_analysis(
                    pose.name,
                    pose.q,
                    sample,
                    runtime_profile=runtime,
                    reference=reference,
                )
                parts = []
                for layer in static.layers:
                    tag = layer.severity.upper()
                    if layer.severity != "pass":
                        parts.append(f"{layer.layer}={tag}({layer.l2_err:.3f})")
                if parts:
                    print(f"    static: {' '.join(parts)}")
            except Exception as exc:
                print(f"    [WARN] Pinocchio gravity failed: {exc}")
    return out


def _sdk_path_z_reasons(session, start_q: Sequence[float], target_q: Sequence[float], z_min_mm: float, steps: int = 10) -> list[str]:
    reasons: list[str] = []
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        code, sdk_pose = session.arm.get_forward_kinematics(
            angles=q.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        if code != 0:
            reasons.append(f"SDK FK failed at path step {i}: code={code}")
            break
        z_mm = float(sdk_pose[2])
        if z_mm < z_min_mm:
            reasons.append(f"SDK path step {i}: EE z {z_mm:.2f} mm < z_min {z_min_mm:.2f} mm")
            break
    return reasons


def _hardware_path_reasons_by_waypoint(
    *,
    session,
    robot,
    scene,
    ee_link,
    dof_idx,
    start_q: Sequence[float],
    target_q: Sequence[float],
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    z_min_mm: float,
    move_strategy: str,
) -> list[str]:
    from ufactory.real_robot_session import build_motion_waypoints

    waypoints = build_motion_waypoints(start_q, target_q, strategy=move_strategy)
    reasons: list[str] = []
    segment_start = np.asarray(start_q, dtype=np.float64)
    for i, waypoint in enumerate(waypoints, start=1):
        segment_reasons: list[str] = []
        segment_reasons.extend(check_joint_limit_path(segment_start, waypoint, joint_lower, joint_upper))
        segment_reasons.extend(
            check_genesis_path_z(
                robot,
                scene,
                ee_link,
                dof_idx,
                segment_start,
                waypoint,
                z_min_mm=z_min_mm,
            )
        )
        segment_reasons.extend(_sdk_path_z_reasons(session, segment_start, waypoint, z_min_mm))
        if segment_reasons:
            reasons.extend(
                f"waypoint {i}/{len(waypoints)} [{move_strategy}]: {reason}"
                for reason in segment_reasons
            )
            break
        segment_start = waypoint
    return reasons


def cli_hardware_check(argv: Sequence[str] | None = None) -> int:
    from ufactory.kinematics import (
        get_robot_sn,
        log_kinematics_sn_status,
        prepare_robot_model_for_verification,
        validate_kinematics_calibration_request,
    )
    from ufactory.real_robot_session import (
        MOVE_STRATEGIES,
        MOVE_STRATEGY_DIRECT,
        RealRobotSession,
        RobotMotionError,
    )

    parser = argparse.ArgumentParser(description="UFACTORY Genesis vs real static torque validation")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", type=str, default=None, help="xArm IP (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Genesis torques + safe poses only")
    parser.add_argument("--hold-current-only", action="store_true", help="Read real torque at current pose")
    parser.add_argument("--kinematics-suffix", type=str, default=None)
    parser.add_argument("--kinematics-yaml", type=str, default=None)
    parser.add_argument("--kinematics-yaml-dir", type=str, default=None)
    parser.add_argument(
        "--force-kinematics",
        action="store_true",
        help="Use --kinematics-suffix/yaml even when SN rules out factory compensation",
    )
    parser.add_argument("--robot-model", type=str, default=None)
    parser.add_argument("--calibrated-output-dir", type=str, default=None)
    parser.add_argument("--z-min-mm", type=float, default=None, help="Minimum EE z (mm); default from robot profile")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed (rad/s) for real moves; default from robot profile",
    )
    parser.add_argument(
        "--move-strategy",
        choices=MOVE_STRATEGIES,
        default=MOVE_STRATEGY_DIRECT,
        help="Real robot joint move strategy; default direct",
    )
    parser.add_argument("--sample-duration", type=float, default=3.0, help="Seconds of hold samples per pose")
    parser.add_argument("--sample-poll", type=float, default=0.1, help="Seconds between hold samples")
    parser.add_argument("--repeats", type=int, default=1, help="Hardware repeats per pose")
    parser.add_argument("--poses", type=str, default=None, help="Comma-separated pose names to run")
    parser.add_argument("--include-stress", action="store_true", help="Include stress/saturation pose set")
    parser.add_argument("--require-reference", action="store_true", help="Fail if Pinocchio is unavailable")
    parser.add_argument(
        "--zero-joint-frictionloss",
        action="store_true",
        help="Set per-joint frictionloss to 0 for this run (friction-noise-floor control run)",
    )
    parser.add_argument(
        "--strict-static",
        action="store_true",
        help="Treat L2/L3 static layer warnings as FAIL (default: L1 only)",
    )
    parser.add_argument(
        "--strict-static-layers",
        type=str,
        default=None,
        help="Subset for --strict-static: l2a,l2b,l3a,l3b or l2,l3,all (default: all L2/L3)",
    )
    parser.add_argument("--report", type=str, default=None, help="CSV report output path")
    parser.add_argument("--jsonl-report", type=str, default=None, help="JSONL report output path")
    parser.add_argument("-v", "--vis", action="store_true", help="Genesis viewer")
    args = parser.parse_args(argv)
    runtime = get_robot_runtime_profile(args.robot)
    strict_layers = parse_strict_static_layers(args.strict_static_layers) if args.strict_static else frozenset()
    if args.z_min_mm is None:
        args.z_min_mm = runtime.dynamics.default_z_min_mm
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s

    if not args.dry_run and not args.hold_current_only and not args.ip:
        parser.error("--ip is required unless --dry-run or --hold-current-only")
    if not args.dry_run and not args.hold_current_only and not runtime.dynamics.supports_hardware_validation:
        parser.error(f"{runtime.model.key} has no hardware dynamics validation pose profile yet")

    if args.z_min_mm < 50.0:
        print(f"[WARN] z_min={args.z_min_mm:.1f} mm is a low-margin hardware mode")

    resolved_sn: str | None = None
    if args.ip and args.kinematics_yaml is None:
        from ufactory.kinematics import resolve_kinematics_suffix_from_ip

        suffix, sn = resolve_kinematics_suffix_from_ip(
            args.ip,
            runtime.model.robot_name,
            kinematics_suffix=args.kinematics_suffix,
            kinematics_yaml=args.kinematics_yaml,
        )
        resolved_sn = sn or None
        args.kinematics_suffix = suffix
        if suffix:
            print(f"kinematics_suffix: {suffix} (auto from SN {sn})")

    urdf_path_str, kinematics_yaml_path = prepare_robot_model_for_verification(
        args.robot_model,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=robot_urdf(runtime.model.key),
        robot_name=runtime.model.robot_name,
        joint_count=runtime.model.dof,
        output_dir=args.calibrated_output_dir,
    )

    print("=" * 78)
    print(f"{runtime.model.key} Static Dynamics Validation: Genesis PD hold vs Real Robot")
    print("=" * 78)
    print(f"URDF : {urdf_path_str}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")
    print(
        f"z_min: {args.z_min_mm:.1f} mm  speed: {args.speed:.4f} rad/s  move_strategy: {args.move_strategy}  "
        f"zero_joint_frictionloss: {args.zero_joint_frictionloss}"
    )

    urdf_issues = validate_urdf_dynamics(urdf_path_str)
    errors = [i for i in urdf_issues if i.severity == "ERROR"]
    if urdf_issues:
        print("\n--- URDF dynamics static checks ---")
        for issue in urdf_issues:
            print(f"  [{issue.severity}] {issue.item}: {issue.message}")
    if errors:
        print("[FAIL] URDF dynamics static checks contain errors")
        return 1

    configs = dynamics_default_configs(runtime.model.key, include_stress=args.include_stress)
    configs, missing = _select_poses(configs, args.poses)
    if missing:
        print(f"[WARN] Requested poses not found: {sorted(missing)}")
    if not configs:
        print("[FAIL] No poses selected")
        return 1

    print("\n--- Building Genesis scene ---")
    scene, robot, ee_link, dof_idx = build_genesis_scene(
        urdf_path_str,
        runtime_profile=runtime,
        show_viewer=args.vis,
        zero_joint_frictionloss=args.zero_joint_frictionloss,
    )
    ee_z_table = compute_ee_z_table_from_sim(robot, scene, ee_link, dof_idx, configs)
    safe_poses, rejected = filter_safe_configs(configs, ee_z_table, args.z_min_mm)

    print("\n--- Safe pose filter (Genesis link6 z) ---")
    print_safe_pose_table(safe_poses, rejected)
    if not safe_poses:
        print("[FAIL] No safe poses after z filter")
        return 1

    reference = load_reference_backend(urdf_path_str, required=args.require_reference)
    if reference is None:
        print("[WARN] Pinocchio reference unavailable; continuing without independent G(q)/M(q)")

    print("\n--- Genesis PD hold torques ---")
    genesis_data = _run_genesis_samples(scene, robot, ee_link, dof_idx, safe_poses, reference, runtime)

    session: RealRobotSession | None = None
    results: list[DynamicsSample] = []
    run_config = make_run_config(
        robot_key=runtime.model.key,
        urdf_path=urdf_path_str,
        kinematics_yaml_path=kinematics_yaml_path,
        mode="dry-run" if args.dry_run else "hardware",
    )
    run_config.robot_sn = resolved_sn or run_config.robot_sn

    try:
        if args.hold_current_only:
            session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
            session.configure_for_dynamics()
            session.print_config()
            q, qvel, tau = session.get_joint_states()
            tau_direct = session.get_joints_torque()
            print(f"\nCurrent q          : {format_torque_row(q)}")
            print(f"Current dq         : {format_torque_row(qvel)}")
            print(f"joint_states effort: {format_torque_row(tau)}")
            if tau_direct is not None:
                print(f"get_joints_torque  : {format_torque_row(tau_direct)}")
            return 0

        if args.dry_run:
            for pose in safe_poses:
                ref_g = reference.gravity(genesis_data[pose.name].q_actual) if reference is not None else None
                results.append(
                    build_dynamics_sample(
                        pose,
                        genesis_data[pose.name],
                        runtime_profile=runtime,
                        reference_gravity_tau=ref_g,
                        reference=reference,
                        skip_reason="dry-run",
                    )
                )
        else:
            session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
            session.configure_for_dynamics()
            session.print_config()
            run_config = make_run_config(
                robot_key=runtime.model.key,
                urdf_path=urdf_path_str,
                kinematics_yaml_path=kinematics_yaml_path,
                mode="hardware",
                session=session,
            )

            sn = get_robot_sn(session.arm)
            run_config.robot_sn = sn or run_config.robot_sn
            validate_kinematics_calibration_request(
                sn,
                runtime.model.robot_name,
                kinematics_yaml=args.kinematics_yaml,
                kinematics_suffix=args.kinematics_suffix,
                allow_sn_override=args.force_kinematics,
            )
            log_kinematics_sn_status(
                sn,
                runtime.model.robot_name,
                kinematics_yaml=kinematics_yaml_path,
                kinematics_suffix=args.kinematics_suffix,
                allow_sn_override=args.force_kinematics,
            )

            joint_lower, joint_upper = parse_joint_limits(urdf_path_str, runtime.arm.joint_names)
            print("\n--- Hardware sampling ---")
            hardware_abort = False
            for pose in safe_poses:
                if hardware_abort:
                    break
                for repeat_i in range(args.repeats):
                    q_now, _, _ = session.get_joint_states()
                    unsafe = _hardware_path_reasons_by_waypoint(
                        session=session,
                        robot=robot,
                        scene=scene,
                        ee_link=ee_link,
                        dof_idx=dof_idx,
                        start_q=q_now,
                        target_q=pose.q,
                        joint_lower=joint_lower,
                        joint_upper=joint_upper,
                        z_min_mm=args.z_min_mm,
                        move_strategy=args.move_strategy,
                    )
                    genesis_sample = genesis_data[pose.name]
                    if unsafe:
                        print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; " + "; ".join(unsafe[:2]))
                        results.append(
                            DynamicsSample(
                                pose=pose.name,
                                q=pose.q,
                                ee_z_mm=pose.ee_z_mm,
                                status=ValidationStatus.UNSAFE,
                                settled=False,
                                saturated=genesis_sample.saturated,
                                q_actual=genesis_sample.q_actual,
                                qvel=genesis_sample.qvel,
                                pd_hold_tau=genesis_sample.pd_hold_tau,
                                actual_dof_force=genesis_sample.actual_dof_force,
                                mass_matrix=genesis_sample.mass_matrix,
                                skip_reason="unsafe",
                                notes=unsafe,
                            )
                        )
                        continue
                    if not genesis_sample.settled or genesis_sample.saturated:
                        status = ValidationStatus.NOT_SETTLED if not genesis_sample.settled else ValidationStatus.SATURATED
                        print(f"  [{pose.name}] repeat {repeat_i + 1}: {status.value}; not moving hardware")
                        results.append(
                            DynamicsSample(
                                pose=pose.name,
                                q=pose.q,
                                ee_z_mm=pose.ee_z_mm,
                                status=status,
                                settled=genesis_sample.settled,
                                saturated=genesis_sample.saturated,
                                q_actual=genesis_sample.q_actual,
                                qvel=genesis_sample.qvel,
                                pd_hold_tau=genesis_sample.pd_hold_tau,
                                actual_dof_force=genesis_sample.actual_dof_force,
                                mass_matrix=genesis_sample.mass_matrix,
                                skip_reason=status.value.lower(),
                            )
                        )
                        continue

                    print(f"  Moving to [{pose.name}] repeat {repeat_i + 1}/{args.repeats} ...")
                    try:
                        real = session.sample_at_hold(
                            pose.q,
                            speed_rad_s=args.speed,
                            move_strategy=args.move_strategy,
                            sample_duration_s=args.sample_duration,
                            sample_poll_s=args.sample_poll,
                        )
                    except RobotMotionError as exc:
                        print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; motion failed: {exc}")
                        results.append(
                            DynamicsSample(
                                pose=pose.name,
                                q=pose.q,
                                ee_z_mm=pose.ee_z_mm,
                                status=ValidationStatus.UNSAFE,
                                settled=False,
                                saturated=genesis_sample.saturated,
                                q_actual=genesis_sample.q_actual,
                                qvel=genesis_sample.qvel,
                                pd_hold_tau=genesis_sample.pd_hold_tau,
                                actual_dof_force=genesis_sample.actual_dof_force,
                                mass_matrix=genesis_sample.mass_matrix,
                                skip_reason="motion_error",
                                notes=[str(exc)],
                            )
                        )
                        hardware_abort = True
                        break
                    ref_g = reference.gravity(genesis_sample.q_actual) if reference is not None else None
                    sample = build_dynamics_sample(
                        pose,
                        genesis_sample,
                        runtime_profile=runtime,
                        tau_real=real.tau,
                        tau_real_median=real.tau_median,
                        tau_real_std=real.tau_std,
                        tau_real_min=real.tau_min,
                        tau_real_max=real.tau_max,
                        tau_direct=real.tau_direct,
                        n_real_samples=real.n_samples,
                        reference_gravity_tau=ref_g,
                        reference=reference,
                        skip_reason="" if real.settled else "not settled",
                    )
                    if not real.settled and sample.status == ValidationStatus.INSUFFICIENT_DATA:
                        sample.status = ValidationStatus.NOT_SETTLED
                    print(f"  [{pose.name}] tau_real_mean={format_torque_row(real.tau)}  [{sample.status.value}]")
                    results.append(sample)

    finally:
        if session is not None and not args.hold_current_only:
            try:
                print("\n--- Returning to home ---")
                session.return_home(
                    speed_rad_s=args.speed,
                    move_strategy=args.move_strategy,
                )
            except Exception as exc:
                print(f"[WARN] return_home failed: {exc}")
            session.disconnect()

    print_compare_table(results, runtime_profile=runtime)
    _, strict_static_fail = print_static_layer_summary(results, strict_layers=strict_layers)

    stamp = now_stamp()
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity=run_config.robot_sn or runtime.model.key,
        report=args.report,
        jsonl_report=args.jsonl_report,
        stamp=stamp,
    )
    run_config.joint_frictionloss = read_joint_frictionloss(robot, dof_idx).tolist()
    write_csv_report(results, csv_path, runtime_profile=runtime)
    write_jsonl_report(results, jsonl_path, run_config=run_config, urdf_issues=urdf_issues, runtime_profile=runtime)
    plot_written = write_torque_plot(results, plot_path, runtime_profile=runtime)
    print(f"\nCSV report  : {csv_path}")
    print(f"JSONL report: {jsonl_path}")
    if plot_written:
        print(f"Torque plot : {plot_path}")
    else:
        print("Torque plot : skipped (no SDK torque data)")

    if args.dry_run:
        print("\n[OK] Dry-run complete (Genesis quantities only)")
        if strict_static_fail:
            print("[FAIL] --strict-static: one or more L2/L3 layers exceeded thresholds")
            return 1
        return 0

    eval_results = [r for r in results if r.status not in {ValidationStatus.NOT_SETTLED, ValidationStatus.SATURATED, ValidationStatus.UNSAFE}]
    n_pass = sum(1 for r in eval_results if r.status == ValidationStatus.PASS)
    n_total = len(eval_results)
    n_unsafe = sum(1 for r in results if r.status == ValidationStatus.UNSAFE)
    all_passed = n_total > 0 and n_pass == n_total and n_unsafe == 0
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{n_total} evaluated poses passed")
    if n_unsafe:
        print(f"UNSAFE: {n_unsafe} poses skipped/aborted")
    print(f"Overall: {'PASS' if all_passed and not strict_static_fail else 'FAIL'}")
    if strict_static_fail:
        print("STATIC: FAIL (--strict-static L2/L3 thresholds exceeded)")
    print("=" * 78)
    return 0 if all_passed and not strict_static_fail else 1


def cli_sim_check(argv: Sequence[str] | None = None) -> int:
    from ufactory.kinematics import prepare_robot_model_for_verification, resolve_kinematics_suffix_from_ip

    parser = argparse.ArgumentParser(description="UFACTORY Genesis dynamics simulation regression")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", type=str, default=None, help="Optional robot IP to auto-resolve kinematics suffix from SN")
    parser.add_argument("--robot-model", type=str, default=None)
    parser.add_argument("--kinematics-suffix", type=str, default=None)
    parser.add_argument("--kinematics-yaml", type=str, default=None)
    parser.add_argument("--kinematics-yaml-dir", type=str, default=None)
    parser.add_argument("--calibrated-output-dir", type=str, default=None)
    parser.add_argument("--z-min-mm", type=float, default=None)
    parser.add_argument("--random-count", type=int, default=100)
    parser.add_argument("--require-reference", action="store_true", help="Fail if Pinocchio is unavailable")
    parser.add_argument(
        "--zero-joint-frictionloss",
        action="store_true",
        help="Set per-joint frictionloss to 0 for this run (friction-noise-floor control run)",
    )
    parser.add_argument(
        "--strict-static",
        action="store_true",
        help="Treat L2/L3 static layer warnings as FAIL (default: L1 only)",
    )
    parser.add_argument(
        "--strict-static-layers",
        type=str,
        default=None,
        help="Subset for --strict-static: l2a,l2b,l3a,l3b or l2,l3,all (default: all L2/L3)",
    )
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--jsonl-report", type=str, default=None)
    args = parser.parse_args(argv)
    runtime = get_robot_runtime_profile(args.robot)
    strict_layers = parse_strict_static_layers(args.strict_static_layers) if args.strict_static else frozenset()
    if args.z_min_mm is None:
        args.z_min_mm = runtime.dynamics.default_z_min_mm

    resolved_sn: str | None = None
    if args.ip and args.kinematics_yaml is None:
        suffix, sn = resolve_kinematics_suffix_from_ip(
            args.ip,
            runtime.model.robot_name,
            kinematics_suffix=args.kinematics_suffix,
            kinematics_yaml=args.kinematics_yaml,
        )
        resolved_sn = sn or None
        args.kinematics_suffix = suffix
        if suffix:
            print(f"kinematics_suffix: {suffix} (auto from SN {sn})")

    urdf_path_str, kinematics_yaml_path = prepare_robot_model_for_verification(
        args.robot_model,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=robot_urdf(runtime.model.key),
        robot_name=runtime.model.robot_name,
        joint_count=runtime.model.dof,
        output_dir=args.calibrated_output_dir,
    )
    issues = validate_urdf_dynamics(urdf_path_str)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        for issue in errors:
            print(f"[ERROR] {issue.item}: {issue.message}")
        return 1

    lower, upper = parse_joint_limits(urdf_path_str, runtime.arm.joint_names)
    lower = np.where(np.isfinite(lower), lower, -1.0)
    upper = np.where(np.isfinite(upper), upper, 1.0)
    rng = np.random.default_rng(42)
    random_configs = [
        (f"random_{i:03d}", rng.uniform(lower + 0.05, upper - 0.05).astype(np.float64))
        for i in range(max(0, args.random_count))
    ]
    configs = merge_test_configs(dynamics_default_configs(runtime.model.key), random_configs)

    scene, robot, ee_link, dof_idx = build_genesis_scene(
        urdf_path_str,
        runtime_profile=runtime,
        zero_joint_frictionloss=args.zero_joint_frictionloss,
    )
    ee_z_table = compute_ee_z_table_from_sim(robot, scene, ee_link, dof_idx, configs)
    safe_poses, rejected = filter_safe_configs(configs, ee_z_table, args.z_min_mm)
    print_safe_pose_table(safe_poses, rejected[:10])
    reference = load_reference_backend(urdf_path_str, required=args.require_reference)

    results: list[DynamicsSample] = []
    for pose in safe_poses:
        gs_sample = genesis_pd_hold_torque_at_q(robot, scene, dof_idx, pose.q, runtime_profile=runtime)
        ref_g = reference.gravity(gs_sample.q_actual) if reference is not None else None
        results.append(
            build_dynamics_sample(
                pose,
                gs_sample,
                runtime_profile=runtime,
                reference_gravity_tau=ref_g,
                reference=reference,
                skip_reason="sim-only",
            )
        )

    print_compare_table(results, runtime_profile=runtime)
    _, strict_static_fail = print_static_layer_summary(results, strict_layers=strict_layers)

    stamp = now_stamp()
    run_config = make_run_config(
        robot_key=runtime.model.key,
        urdf_path=urdf_path_str,
        kinematics_yaml_path=kinematics_yaml_path,
        mode="sim",
    )
    run_config.robot_sn = resolved_sn or run_config.robot_sn
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity=run_config.robot_sn or runtime.model.key,
        report=args.report,
        jsonl_report=args.jsonl_report,
        stamp=stamp,
    )
    run_config.joint_frictionloss = read_joint_frictionloss(robot, dof_idx).tolist()
    write_csv_report(results, csv_path, runtime_profile=runtime)
    write_jsonl_report(results, jsonl_path, run_config=run_config, urdf_issues=issues, runtime_profile=runtime)
    plot_written = write_torque_plot(results, plot_path, runtime_profile=runtime)
    n_bad = sum(1 for r in results if r.status in {ValidationStatus.NOT_SETTLED, ValidationStatus.SATURATED})
    print(f"Simulation samples: {len(results)}, unstable/saturated: {n_bad}")
    print(f"CSV report  : {csv_path}")
    print(f"JSONL report: {jsonl_path}")
    if plot_written:
        print(f"Torque plot : {plot_path}")
    else:
        print("Torque plot : skipped (no SDK torque data)")
    if strict_static_fail:
        print("[FAIL] --strict-static: one or more L2/L3 layers exceeded thresholds")
        return 1
    return 0 if n_bad == 0 else 1


@dataclass
class SimCollisionResult:
    pose_name: str
    passed: bool
    error_code: int | None = None
    waypoint_index: int | None = None
    message: str = ""


def run_sim_collision_chain(
    session,
    poses: Sequence[tuple[str, np.ndarray]],
    *,
    speed_rad_s: float,
    move_strategy: str,
) -> list[SimCollisionResult]:
    """Move through poses in order without returning home between them."""
    from ufactory.real_robot_session import RobotMotionError

    results: list[SimCollisionResult] = []
    for name, q in poses:
        print(f"\n--- [{name}] ---")
        try:
            session.move_to(
                q,
                speed_rad_s=speed_rad_s,
                wait=True,
                move_strategy=move_strategy,
            )
            error_code = int(session.arm.error_code)
            if error_code == 22:
                results.append(
                    SimCollisionResult(
                        pose_name=name,
                        passed=False,
                        error_code=error_code,
                        message="self-collision (error_code=22) after move",
                    )
                )
                print(f"  FAIL: self-collision error_code=22")
                session.recover_after_motion_error()
                continue
            results.append(SimCollisionResult(pose_name=name, passed=True))
            print("  PASS")
        except RobotMotionError as exc:
            error_code = int(exc.code)
            results.append(
                SimCollisionResult(
                    pose_name=name,
                    passed=False,
                    error_code=error_code,
                    waypoint_index=exc.waypoint_index,
                    message=str(exc),
                )
            )
            print(f"  FAIL: {exc}")
            session.recover_after_motion_error()
    return results


def write_sim_collision_report(results: Sequence[SimCollisionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["pose", "passed", "error_code", "waypoint_index", "message"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "pose": row.pose_name,
                    "passed": row.passed,
                    "error_code": row.error_code if row.error_code is not None else "",
                    "waypoint_index": row.waypoint_index if row.waypoint_index is not None else "",
                    "message": row.message,
                }
            )


def cli_sim_collision_check(argv: Sequence[str] | None = None) -> int:
    from ufactory.real_robot_session import MOVE_STRATEGIES, MOVE_STRATEGY_DIRECT, RealRobotSession

    parser = argparse.ArgumentParser(
        description="xArm simulation-mode chained self-collision check for dynamics poses",
    )
    parser.add_argument("--ip", type=str, required=True, help="xArm IP (simulation mode)")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--poses", type=str, default=None, help="Comma-separated pose names to run")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed (rad/s); default from robot profile",
    )
    parser.add_argument(
        "--move-strategy",
        choices=MOVE_STRATEGIES,
        default=MOVE_STRATEGY_DIRECT,
        help="Joint move strategy; default direct",
    )
    parser.add_argument("--report", type=str, default=None, help="CSV report output path")
    args = parser.parse_args(argv)

    runtime = get_robot_runtime_profile(args.robot)
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s
    if not runtime.dynamics.supports_hardware_validation:
        parser.error(f"{runtime.model.key} has no hardware dynamics validation pose profile yet")

    configs = dynamics_default_configs(runtime.model.key)
    configs, missing = _select_poses(configs, args.poses)
    if missing:
        print(f"[WARN] Requested poses not found: {sorted(missing)}")
    if not configs:
        print("[FAIL] No poses selected")
        return 1

    print("=" * 78)
    print(f"{runtime.model.key} Simulation Self-Collision Chain Check")
    print("=" * 78)
    print(f"IP             : {args.ip}")
    print(f"poses          : {len(configs)}")
    print(f"speed          : {args.speed:.4f} rad/s")
    print(f"move_strategy  : {args.move_strategy}")

    session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
    try:
        session.configure_for_simulation_collision_check()
        session.print_config()

        print("\n--- Moving to home ---")
        session.move_to(
            runtime.arm.home_qpos,
            speed_rad_s=args.speed,
            wait=True,
            move_strategy=args.move_strategy,
        )

        results = run_sim_collision_chain(
            session,
            configs,
            speed_rad_s=args.speed,
            move_strategy=args.move_strategy,
        )
    finally:
        try:
            session.arm.set_simulation_robot(on_off=False)
        except Exception as exc:
            print(f"[WARN] set_simulation_robot(False) failed: {exc}")
        session.disconnect()

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{len(results)} poses passed, {n_fail} failed")
    for row in results:
        if not row.passed:
            wp = f" waypoint={row.waypoint_index}" if row.waypoint_index is not None else ""
            code = f" error_code={row.error_code}" if row.error_code is not None else ""
            print(f"  FAIL [{row.pose_name}]{wp}{code}: {row.message[:120]}")
    print("=" * 78)

    stamp = now_stamp()
    report_path = Path(args.report) if args.report else Path("reports") / f"dynamics_sim_collision_{stamp}.csv"
    write_sim_collision_report(results, report_path)
    print(f"CSV report: {report_path}")

    return 0 if n_fail == 0 else 1


def cli_report_compare(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two dynamics validation reports")
    parser.add_argument("old_report")
    parser.add_argument("new_report")
    args = parser.parse_args(argv)

    old_records = read_report_records(args.old_report)
    new_records = read_report_records(args.new_report)
    stats = compare_report_records(old_records, new_records)
    print(f"{'Joint':>5} {'old_bias':>10} {'new_bias':>10} {'d_bias':>10} {'old_rmse':>10} {'new_rmse':>10} {'d_rmse':>10}")
    for row in stats:
        print(
            f"J{int(row['joint']):<4} "
            f"{row['old_bias']:10.4f} {row['new_bias']:10.4f} {row['bias_delta']:10.4f} "
            f"{row['old_rmse']:10.4f} {row['new_rmse']:10.4f} {row['rmse_delta']:10.4f}"
        )
    return 0
