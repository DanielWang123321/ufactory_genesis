"""Long-duration static hold observation: joint torque vs motor current time series."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ufactory.dynamics.poses import dynamics_default_configs
from ufactory.dynamics.reference import load_reference_backend
from ufactory.dynamics.report import now_stamp
from ufactory.kinematics.calibration import (
    get_robot_sn,
    log_kinematics_sn_status,
    prepare_robot_model_for_verification,
    validate_kinematics_calibration_request,
)
from ufactory.robots.paths import robot_urdf
from ufactory.hardware.session import (
    MOVE_STRATEGY_DIRECT,
    HoldTimeSeriesSample,
    RealRobotSession,
    RobotMotionError,
)
from ufactory.robots.runtime import get_robot_runtime_profile, robot_runtime_cli_choices


@dataclass
class SegmentStats:
    t_start_s: float
    t_end_s: float
    n: int
    tau_mean: float
    tau_std: float
    tau_min: float
    tau_max: float
    current_mean: float
    current_std: float
    current_min: float
    current_max: float
    qvel_max_abs: float


@dataclass
class HoldObserveSummary:
    context: str
    pose: str
    joint: int
    duration_s: float
    poll_s: float
    pin_g_j: float | None
    abs_err_limit_j: float | None
    segment_a: SegmentStats
    segment_b: SegmentStats
    segment_c: SegmentStats
    tau_current_corr_all: float | None
    tau_current_corr_segment_a: float | None
    current_zero_crossings: int
    current_peak_count: int
    l1_fail_segment_a: bool | None
    interpretation: str


def _select_pose_q(configs: Sequence[tuple[str, np.ndarray]], name: str) -> np.ndarray:
    for pose_name, q in configs:
        if pose_name == name:
            return np.asarray(q, dtype=np.float64)
    raise KeyError(f"Pose not found: {name}")


def _chain_poses_up_to(
    configs: Sequence[tuple[str, np.ndarray]],
    stop_at: str,
) -> list[tuple[str, np.ndarray]]:
    chain: list[tuple[str, np.ndarray]] = []
    for name, q in configs:
        chain.append((name, np.asarray(q, dtype=np.float64)))
        if name == stop_at:
            break
    if not chain or chain[-1][0] != stop_at:
        raise KeyError(f"chain-stop-at pose not in default configs: {stop_at}")
    return chain


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _segment_stats(
    samples: Sequence[HoldTimeSeriesSample],
    *,
    joint_idx: int,
    t_start_s: float,
    t_end_s: float,
) -> SegmentStats:
    subset = [s for s in samples if t_start_s <= s.t_s <= t_end_s]
    if not subset:
        return SegmentStats(
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            n=0,
            tau_mean=float("nan"),
            tau_std=float("nan"),
            tau_min=float("nan"),
            tau_max=float("nan"),
            current_mean=float("nan"),
            current_std=float("nan"),
            current_min=float("nan"),
            current_max=float("nan"),
            qvel_max_abs=float("nan"),
        )
    tau = np.asarray([s.tau[joint_idx] for s in subset], dtype=np.float64)
    current = np.asarray([s.current[joint_idx] for s in subset], dtype=np.float64)
    qvel = np.asarray([s.qvel[joint_idx] for s in subset], dtype=np.float64)
    return SegmentStats(
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        n=len(subset),
        tau_mean=float(tau.mean()),
        tau_std=float(tau.std()),
        tau_min=float(tau.min()),
        tau_max=float(tau.max()),
        current_mean=float(current.mean()),
        current_std=float(current.std()),
        current_min=float(current.min()),
        current_max=float(current.max()),
        qvel_max_abs=float(np.abs(qvel).max()),
    )


def _count_zero_crossings(values: np.ndarray) -> int:
    if values.size < 2:
        return 0
    centered = values - float(values.mean())
    signs = np.sign(centered)
    signs[signs == 0.0] = 1.0
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def _count_local_peaks(values: np.ndarray) -> int:
    if values.size < 3:
        return 0
    count = 0
    for i in range(1, values.size - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            count += 1
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            count += 1
    return count


def _interpret_summary(summary: HoldObserveSummary) -> str:
    a = summary.segment_a
    c = summary.segment_c
    if not math.isfinite(a.tau_mean) or not math.isfinite(c.tau_mean):
        return "insufficient samples"
    tau_drift = abs(a.tau_mean - c.tau_mean)
    current_late_std = c.current_std
    if summary.current_zero_crossings >= 4 or summary.current_peak_count >= 6:
        if summary.tau_current_corr_all is not None and abs(summary.tau_current_corr_all) > 0.8:
            return "supports motor PID current instability (oscillation + tau/current coupling)"
    if current_late_std < 0.05 * max(abs(c.current_mean), 1e-6) and tau_drift > 0.5:
        return "supports transmission-side relaxation (current stable, tau still drifting)"
    if tau_drift > 0.5 and summary.tau_current_corr_all is not None and abs(summary.tau_current_corr_all) > 0.9:
        return "slow coupled decay (current and tau converge together; fix via sampling protocol)"
    if tau_drift <= 0.2:
        return "stable within observation window"
    return "slow convergence; extend sampling or use tail-window mean"


def analyze_hold_timeseries(
    samples: Sequence[HoldTimeSeriesSample],
    *,
    joint: int,
    context: str,
    pose: str,
    duration_s: float,
    poll_s: float,
    pin_g_j: float | None,
    abs_err_limit_j: float | None,
) -> HoldObserveSummary:
    joint_idx = joint - 1
    if joint_idx < 0:
        raise ValueError(f"joint must be >= 1, got {joint}")
    if samples and joint_idx >= samples[0].tau.size:
        raise ValueError(f"joint {joint} exceeds DOF {samples[0].tau.size}")

    seg_a = _segment_stats(samples, joint_idx=joint_idx, t_start_s=0.0, t_end_s=3.0)
    seg_b = _segment_stats(samples, joint_idx=joint_idx, t_start_s=3.0, t_end_s=8.0)
    tail_start = max(0.0, duration_s - 10.0)
    seg_c = _segment_stats(samples, joint_idx=joint_idx, t_start_s=tail_start, t_end_s=duration_s)

    tau_all = np.asarray([s.tau[joint_idx] for s in samples], dtype=np.float64)
    current_all = np.asarray([s.current[joint_idx] for s in samples], dtype=np.float64)
    tau_a = np.asarray([s.tau[joint_idx] for s in samples if s.t_s <= 3.0], dtype=np.float64)
    current_a = np.asarray([s.current[joint_idx] for s in samples if s.t_s <= 3.0], dtype=np.float64)

    l1_fail = None
    if pin_g_j is not None and abs_err_limit_j is not None and math.isfinite(seg_a.tau_mean):
        l1_fail = abs(seg_a.tau_mean - pin_g_j) > abs_err_limit_j

    summary = HoldObserveSummary(
        context=context,
        pose=pose,
        joint=joint,
        duration_s=duration_s,
        poll_s=poll_s,
        pin_g_j=pin_g_j,
        abs_err_limit_j=abs_err_limit_j,
        segment_a=seg_a,
        segment_b=seg_b,
        segment_c=seg_c,
        tau_current_corr_all=_pearson(tau_all, current_all),
        tau_current_corr_segment_a=_pearson(tau_a, current_a),
        current_zero_crossings=_count_zero_crossings(current_all),
        current_peak_count=_count_local_peaks(current_all),
        l1_fail_segment_a=l1_fail,
        interpretation="",
    )
    summary.interpretation = _interpret_summary(summary)
    return summary


def write_timeseries_csv(
    path: Path,
    samples: Sequence[HoldTimeSeriesSample],
    *,
    context: str,
    pose: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dof = samples[0].tau.size if samples else 0
    fieldnames = ["context", "pose", "t_s"]
    for i in range(1, dof + 1):
        fieldnames.extend([f"q{i}", f"qvel{i}", f"tau{i}", f"current{i}", f"tau_direct{i}"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row: dict[str, float | str] = {
                "context": context,
                "pose": pose,
                "t_s": f"{sample.t_s:.4f}",
            }
            for i in range(dof):
                j = i + 1
                row[f"q{j}"] = f"{sample.q[i]:.6f}"
                row[f"qvel{j}"] = f"{sample.qvel[i]:.6f}"
                row[f"tau{j}"] = f"{sample.tau[i]:.6f}"
                row[f"current{j}"] = f"{sample.current[i]:.6f}"
                direct = sample.tau_direct[i] if sample.tau_direct is not None else float("nan")
                row[f"tau_direct{j}"] = f"{direct:.6f}"
            writer.writerow(row)


def print_summary(summary: HoldObserveSummary) -> None:
    j = summary.joint
    print(f"\n=== Hold observe: context={summary.context} pose={summary.pose} J{j} ===")
    if summary.pin_g_j is not None:
        print(f"pin_G[J{j}] = {summary.pin_g_j:.4f} Nm  abs_err_limit = {summary.abs_err_limit_j:.4f} Nm")
    for label, seg in [("A [0,3s]", summary.segment_a), ("B [3,8s]", summary.segment_b), ("C tail", summary.segment_c)]:
        print(
            f"  segment {label}: n={seg.n} "
            f"tau mean={seg.tau_mean:+.4f} std={seg.tau_std:.4f} min={seg.tau_min:+.4f} max={seg.tau_max:+.4f} | "
            f"I mean={seg.current_mean:+.4f} std={seg.current_std:.4f} min={seg.current_min:+.4f} max={seg.current_max:+.4f}"
        )
    print(
        f"  corr(tau,I) all={summary.tau_current_corr_all} segA={summary.tau_current_corr_segment_a} "
        f"I zero_cross={summary.current_zero_crossings} I peaks={summary.current_peak_count}"
    )
    if summary.l1_fail_segment_a is not None:
        verdict = "FAIL" if summary.l1_fail_segment_a else "PASS"
        print(f"  L1 using segment-A tau mean vs pin_G: {verdict}")
    print(f"  interpretation: {summary.interpretation}")


def _run_context(
    session: RealRobotSession,
    *,
    context: str,
    pose_name: str,
    target_q: np.ndarray,
    chain: Sequence[tuple[str, np.ndarray]] | None,
    speed_rad_s: float,
    post_move_wait_s: float,
    duration_s: float,
    poll_s: float,
) -> list[HoldTimeSeriesSample]:
    if context == "direct":
        print(f"\n--- [{context}] home -> {pose_name} -> observe {duration_s:.0f}s ---")
        return session.observe_at_hold(
            target_q,
            speed_rad_s=speed_rad_s,
            move_strategy=MOVE_STRATEGY_DIRECT,
            post_move_wait_s=post_move_wait_s,
            duration_s=duration_s,
            poll_s=poll_s,
        )

    if context != "chain":
        raise ValueError(f"Unknown context: {context}")

    if chain is None:
        raise ValueError("chain poses required for context=chain")
    print(f"\n--- [{context}] {len(chain)} poses -> {pose_name} -> observe {duration_s:.0f}s ---")
    for i, (name, q) in enumerate(chain, start=1):
        print(f"  chain {i}/{len(chain)}: {name}")
        try:
            session.move_to(q, speed_rad_s=speed_rad_s, wait=True, move_strategy=MOVE_STRATEGY_DIRECT)
        except RobotMotionError as exc:
            raise RuntimeError(f"chain motion failed at {name}: {exc}") from exc
    time.sleep(post_move_wait_s)
    session.wait_until_settled(poll_s=poll_s)
    return session.collect_hold_timeseries(duration_s=duration_s, poll_s=poll_s)


def cli_observe_hold_current(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static hold observation: joint torque and motor current time series",
    )
    parser.add_argument("--robot", default="lite6", choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", required=True, help="xArm IP")
    parser.add_argument("--pose", default="0", help="Target pose index (comma-separated for multiple)")
    parser.add_argument("--duration", type=float, default=60.0, help="Observation duration after settle (seconds)")
    parser.add_argument("--poll", type=float, default=0.1, help="Poll interval (seconds)")
    parser.add_argument("--post-move-wait", type=float, default=1.5, help="Wait after motion before settle/observe")
    parser.add_argument("--joint", type=int, default=2, help="Analysis focus joint (1-indexed)")
    parser.add_argument(
        "--context",
        choices=("direct", "chain", "both"),
        default="both",
        help="Arrival path: direct from home, full default chain, or both",
    )
    parser.add_argument(
        "--chain-stop-at",
        default=None,
        help="Last pose in chain mode (defaults to --pose first name)",
    )
    parser.add_argument("--speed", type=float, default=None, help="Joint speed rad/s (robot profile default)")
    parser.add_argument("--kinematics-suffix", type=str, default=None)
    parser.add_argument("--kinematics-yaml", type=str, default=None)
    parser.add_argument("--kinematics-yaml-dir", type=str, default=None)
    parser.add_argument("--robot-model", type=str, default=None)
    parser.add_argument("--calibrated-output-dir", type=str, default=None)
    parser.add_argument("--force-kinematics", action="store_true")
    parser.add_argument("--require-reference", action="store_true", help="Fail if Pinocchio unavailable")
    parser.add_argument("--output", type=str, default=None, help="CSV output path (auto if omitted)")
    parser.add_argument("--summary-json", type=str, default=None, help="JSON summary path (auto if omitted)")
    args = parser.parse_args(argv)

    runtime = get_robot_runtime_profile(args.robot)
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s

    pose_names = [p.strip() for p in args.pose.split(",") if p.strip()]
    if not pose_names:
        parser.error("No pose names provided")

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
    reference = load_reference_backend(urdf_path_str, required=args.require_reference)
    configs = dynamics_default_configs(runtime.model.key)
    joint_idx = args.joint - 1
    abs_limits = runtime.dynamics.abs_err_limits
    abs_err_j = float(abs_limits[joint_idx]) if joint_idx < len(abs_limits) else None

    contexts: list[str]
    if args.context == "both":
        contexts = ["direct", "chain"]
    else:
        contexts = [args.context]

    stamp = now_stamp()
    csv_path = Path(args.output) if args.output else Path("reports") / f"hold_observe_{stamp}.csv"
    json_path = Path(args.summary_json) if args.summary_json else csv_path.with_suffix(".summary.json")

    session = RealRobotSession(
        args.ip,
        dof=runtime.model.dof,
        home_qpos=runtime.arm.home_qpos,
    )
    summaries: list[HoldObserveSummary] = []
    all_rows: list[tuple[str, str, list[HoldTimeSeriesSample]]] = []
    try:
        session.configure_for_dynamics()
        session.print_config()
        sn = get_robot_sn(session.arm)
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
        print(f"observe: robot={runtime.model.key} duration={args.duration}s poll={args.poll}s joint=J{args.joint}")

        for pose_name in pose_names:
            target_q = _select_pose_q(configs, pose_name)
            chain_stop = args.chain_stop_at or pose_name
            chain = _chain_poses_up_to(configs, chain_stop)
            pin_g_j = None
            if reference is not None:
                pin_g_j = float(reference.gravity(target_q)[joint_idx])

            for context in contexts:
                samples = _run_context(
                    session,
                    context=context,
                    pose_name=pose_name,
                    target_q=target_q,
                    chain=chain,
                    speed_rad_s=args.speed,
                    post_move_wait_s=args.post_move_wait,
                    duration_s=args.duration,
                    poll_s=args.poll,
                )
                all_rows.append((context, pose_name, samples))
                summary = analyze_hold_timeseries(
                    samples,
                    joint=args.joint,
                    context=context,
                    pose=pose_name,
                    duration_s=args.duration,
                    poll_s=args.poll,
                    pin_g_j=pin_g_j,
                    abs_err_limit_j=abs_err_j,
                )
                summaries.append(summary)
                print_summary(summary)
    finally:
        try:
            session.return_home(speed_rad_s=args.speed)
        finally:
            session.disconnect()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        dof = all_rows[0][2][0].tau.size
        fieldnames = ["context", "pose", "t_s"]
        for i in range(1, dof + 1):
            fieldnames.extend([f"q{i}", f"qvel{i}", f"tau{i}", f"current{i}", f"tau_direct{i}"])
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for context, pose_name, samples in all_rows:
                for sample in samples:
                    row: dict[str, float | str] = {
                        "context": context,
                        "pose": pose_name,
                        "t_s": f"{sample.t_s:.4f}",
                    }
                    for i in range(dof):
                        j = i + 1
                        row[f"q{j}"] = f"{sample.q[i]:.6f}"
                        row[f"qvel{j}"] = f"{sample.qvel[i]:.6f}"
                        row[f"tau{j}"] = f"{sample.tau[i]:.6f}"
                        row[f"current{j}"] = f"{sample.current[i]:.6f}"
                        direct = sample.tau_direct[i] if sample.tau_direct is not None else float("nan")
                        row[f"tau_direct{j}"] = f"{direct:.6f}"
                    writer.writerow(row)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot": runtime.model.key,
        "ip": args.ip,
        "duration_s": args.duration,
        "poll_s": args.poll,
        "joint": args.joint,
        "csv": str(csv_path),
        "summaries": [asdict(s) for s in summaries],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote CSV   : {csv_path}")
    print(f"Wrote JSON  : {json_path}")
    return 0
