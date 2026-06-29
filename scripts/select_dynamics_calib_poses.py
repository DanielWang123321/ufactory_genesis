#!/usr/bin/env python3
"""Select dynamics calibration poses with balanced EE y+ / y- coverage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ufactory.dynamics_pose_selection import (  # noqa: E402
    EE_Y_SIDE_NEG,
    EE_Y_SIDE_POS,
    PoseCandidate,
    classify_ee_y_side,
    order_poses_greedy,
    select_stratified_by_ee_y,
)
from ufactory.dynamics_validation import (  # noqa: E402
    build_genesis_scene,
    compute_ee_xyz_table_from_sim,
    filter_safe_configs,
    genesis_pd_hold_torque_at_q,
    run_sim_collision_chain,
)
from ufactory.kinematics import prepare_robot_model_for_verification  # noqa: E402
from ufactory.paths import robot_urdf  # noqa: E402
from ufactory.real_robot_session import (  # noqa: E402
    MOVE_STRATEGY_DIRECT,
    RealRobotSession,
)
from ufactory.robot_params import (  # noqa: E402
    DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
    get_robot_runtime_profile,
)


def load_calib_poses(path: Path) -> list[tuple[str, np.ndarray]]:
    poses: list[tuple[str, np.ndarray]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        vals = [float(v.strip()) for v in line.split(",")]
        if len(vals) != 6:
            raise ValueError(f"Line {i}: expected 6 joints, got {len(vals)}")
        poses.append((f"calib_{i:03d}", np.asarray(vals, dtype=np.float64)))
    return poses


def resolve_urdf(kinematics_suffix: str | None) -> str:
    urdf_path, _ = prepare_robot_model_for_verification(
        None,
        None,
        kinematics_suffix,
        None,
        default_base_urdf=robot_urdf("xarm6"),
        robot_name="xarm6",
    )
    return urdf_path


def genesis_filter(
    poses: list[tuple[str, np.ndarray]],
    *,
    urdf: str,
    z_min_mm: float,
) -> list[PoseCandidate]:
    runtime = get_robot_runtime_profile("xarm6")
    scene, robot, ee_link, dof_idx = build_genesis_scene(urdf, runtime_profile=runtime, backend="cpu")
    ee_xyz_table = compute_ee_xyz_table_from_sim(robot, scene, ee_link, dof_idx, poses)
    ee_z_table = {name: xyz[2] for name, xyz in ee_xyz_table.items()}
    safe, rejected = filter_safe_configs(poses, ee_z_table, z_min_mm)

    rejected_names = {name for name, _ in rejected}
    out: list[PoseCandidate] = []
    for name, q in poses:
        if name in rejected_names:
            continue
        sample = genesis_pd_hold_torque_at_q(robot, scene, dof_idx, q, runtime_profile=runtime)
        if not sample.settled or sample.saturated:
            continue
        x_mm, y_mm, z_mm = ee_xyz_table[name]
        tau = np.asarray(sample.pd_hold_tau, dtype=np.float64)
        out.append(
            PoseCandidate(
                name=name,
                q=q,
                ee_x_mm=float(x_mm),
                ee_y_mm=float(y_mm),
                ee_z_mm=float(z_mm),
                tau_norm=float(np.linalg.norm(tau)),
                ee_y_side=classify_ee_y_side(y_mm),
            )
        )
    return out


def _reset_sim_state(session: RealRobotSession) -> None:
    session.arm.set_simulation_robot(on_off=False)
    time.sleep(0.15)
    session.arm.set_simulation_robot(on_off=True)
    time.sleep(0.15)
    session.recover_after_motion_error()


def check_collision_from_home(
    session: RealRobotSession,
    q_target: np.ndarray,
    *,
    speed_rad_s: float,
) -> tuple[bool, str]:
    _reset_sim_state(session)
    if not session.recover_after_motion_error():
        return False, "recover failed"
    try:
        session.move_to(
            session.home_qpos,
            speed_rad_s=speed_rad_s,
            wait=True,
            move_strategy=MOVE_STRATEGY_DIRECT,
        )
        session.move_to(
            q_target,
            speed_rad_s=speed_rad_s,
            wait=True,
            move_strategy=MOVE_STRATEGY_DIRECT,
        )
    except Exception as exc:
        return False, str(exc)
    if int(session.arm.error_code) == 22:
        return False, "self-collision error_code=22"
    return True, ""


def collision_filter(
    candidates: list[PoseCandidate],
    *,
    ip: str,
    speed_rad_s: float,
) -> list[PoseCandidate]:
    runtime = get_robot_runtime_profile("xarm6")
    session = RealRobotSession(ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
    try:
        session.configure_for_simulation_collision_check()
        passed: list[PoseCandidate] = []
        for candidate in candidates:
            ok, note = check_collision_from_home(session, candidate.q, speed_rad_s=speed_rad_s)
            status = "OK [direct]" if ok else f"FAIL {note}"
            print(f"  collision [{candidate.name}]: {status}")
            if ok:
                passed.append(
                    PoseCandidate(
                        name=candidate.name,
                        q=candidate.q,
                        ee_x_mm=candidate.ee_x_mm,
                        ee_y_mm=candidate.ee_y_mm,
                        ee_z_mm=candidate.ee_z_mm,
                        tau_norm=candidate.tau_norm,
                        ee_y_side=candidate.ee_y_side,
                        collision_ok=True,
                        collision_note=note,
                        move_strategy=MOVE_STRATEGY_DIRECT,
                    )
                )
    finally:
        try:
            session.arm.set_simulation_robot(on_off=False)
        except Exception:
            pass
        session.disconnect()
    return passed


def verify_chain_collision(
    ordered: list[PoseCandidate],
    *,
    ip: str,
    speed_rad_s: float,
) -> bool:
    runtime = get_robot_runtime_profile("xarm6")
    configs = [("home", runtime.arm.home_qpos)] + [(c.name, c.q) for c in ordered]
    session = RealRobotSession(ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
    try:
        session.configure_for_simulation_collision_check()
        session.move_to(
            runtime.arm.home_qpos,
            speed_rad_s=speed_rad_s,
            wait=True,
            move_strategy=MOVE_STRATEGY_DIRECT,
        )
        results = run_sim_collision_chain(
            session,
            configs,
            speed_rad_s=speed_rad_s,
            move_strategy=MOVE_STRATEGY_DIRECT,
        )
    finally:
        try:
            session.arm.set_simulation_robot(on_off=False)
        except Exception:
            pass
        session.disconnect()
    n_pass = sum(1 for row in results if row.passed)
    print(f"\nChain verification: {n_pass}/{len(results)} passed")
    return n_pass == len(results)


def format_python_configs(ordered: list[PoseCandidate]) -> str:
    lines = ["XARM6_DEFAULT_DYNAMICS_CONFIGS: NamedPoseTuple = ("]
    lines.append('    ("home", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),')
    for candidate in ordered:
        qstr = ", ".join(f"{v:.6f}" for v in candidate.q)
        lines.append(f'    ("{candidate.name}", ({qstr})),')
    lines.append(")")
    return "\n".join(lines)


def _count_y_sides(candidates: Sequence[PoseCandidate], y_tol_mm: float) -> tuple[int, int, int]:
    y_pos = y_neg = neutral = 0
    for candidate in candidates:
        side = classify_ee_y_side(candidate.ee_y_mm, y_tol_mm=y_tol_mm)
        if side == EE_Y_SIDE_POS:
            y_pos += 1
        elif side == EE_Y_SIDE_NEG:
            y_neg += 1
        else:
            neutral += 1
    return y_pos, y_neg, neutral


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select y-balanced dynamics calibration poses")
    parser.add_argument(
        "--calib-file",
        type=Path,
        default=Path("/home/uf/Desktop/xarm6_joint_pos.txt"),
        help="Calibration pose source file (rad, 6 joints per line)",
    )
    parser.add_argument("--ip", type=str, default="192.168.1.65", help="xArm IP for simulation collision")
    parser.add_argument("--kinematics-suffix", type=str, default="xi1305", help="Per-unit kinematics suffix")
    parser.add_argument("--z-min-mm", type=float, default=0.0, help="Minimum EE z (mm)")
    parser.add_argument("--target-count", type=int, default=20, help="Total calib poses to select (excl. home)")
    parser.add_argument("--y-pos-count", type=int, default=10, help="Target y+ pose count")
    parser.add_argument("--y-neg-count", type=int, default=10, help="Target y- pose count")
    parser.add_argument("--y-tol-mm", type=float, default=10.0, help="EE y hemisphere threshold (mm)")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed (rad/s); default from robot profile",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/selected_calib_poses_ybalanced.json"),
        help="JSON report output path",
    )
    parser.add_argument("--skip-collision", action="store_true", help="Skip robot simulation collision checks")
    parser.add_argument("--skip-chain-verify", action="store_true", help="Skip chained collision verification")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.y_pos_count + args.y_neg_count != args.target_count:
        parser.error("y-pos-count + y-neg-count must equal target-count")

    speed_rad_s = args.speed if args.speed is not None else DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
    urdf = resolve_urdf(args.kinematics_suffix)

    print("Loading calibration poses...")
    poses = load_calib_poses(args.calib_file)
    print(f"  {len(poses)} poses from {args.calib_file}")

    print("\nGenesis filter (settled, not saturated, z_min)...")
    genesis_ok = genesis_filter(poses, urdf=urdf, z_min_mm=args.z_min_mm)
    y_pos, y_neg, neutral = _count_y_sides(genesis_ok, args.y_tol_mm)
    print(f"  {len(genesis_ok)}/{len(poses)} passed Genesis (y+={y_pos}, y-={y_neg}, neutral={neutral})")

    if args.skip_collision:
        collision_ok = genesis_ok
        print("\nSkipping simulation collision filter (--skip-collision)")
    else:
        print("\nSimulation collision filter (direct home -> target)...")
        collision_ok = collision_filter(genesis_ok, ip=args.ip, speed_rad_s=speed_rad_s)
        print(f"  {len(collision_ok)}/{len(genesis_ok)} passed collision")
        y_pos, y_neg, neutral = _count_y_sides(collision_ok, args.y_tol_mm)
        print(f"  collision pool: y+={y_pos}, y-={y_neg}, neutral={neutral}")

    print(f"\nStratified selection (y+={args.y_pos_count}, y-={args.y_neg_count})...")
    selected = select_stratified_by_ee_y(
        collision_ok,
        n_y_pos=args.y_pos_count,
        n_y_neg=args.y_neg_count,
        y_tol_mm=args.y_tol_mm,
    )
    y_pos, y_neg, neutral = _count_y_sides(selected, args.y_tol_mm)
    print(f"  selected {len(selected)} poses (y+={y_pos}, y-={y_neg}, neutral={neutral})")

    print("\nGreedy ordering...")
    ordered = order_poses_greedy(selected)
    for i, candidate in enumerate(ordered, start=1):
        print(
            f"  {i:2d}. {candidate.name} y={candidate.ee_y_mm:+.1f}mm "
            f"tau_norm={candidate.tau_norm:.2f} ee_z={candidate.ee_z_mm:.1f}mm"
        )

    if not args.skip_collision and not args.skip_chain_verify:
        print("\nChained collision verification (home + ordered poses)...")
        if not verify_chain_collision(ordered, ip=args.ip, speed_rad_s=speed_rad_s):
            print("[WARN] Chain verification failed; review ordering or candidates")

    payload = {
        "source": str(args.calib_file),
        "kinematics_suffix": args.kinematics_suffix,
        "z_min_mm": args.z_min_mm,
        "y_tol_mm": args.y_tol_mm,
        "y_pos_count": args.y_pos_count,
        "y_neg_count": args.y_neg_count,
        "speed_rad_s": speed_rad_s,
        "selected": [
            {
                "name": candidate.name,
                "q": candidate.q.tolist(),
                "ee_x_mm": candidate.ee_x_mm,
                "ee_y_mm": candidate.ee_y_mm,
                "ee_z_mm": candidate.ee_z_mm,
                "ee_y_side": candidate.ee_y_side,
                "tau_norm": candidate.tau_norm,
                "collision_note": candidate.collision_note,
                "move_strategy": candidate.move_strategy,
            }
            for candidate in ordered
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON saved: {args.output_json}")

    py_block = format_python_configs(ordered)
    py_out = args.output_json.with_suffix(".py.txt")
    py_out.write_text(py_block, encoding="utf-8")
    print(f"Python snippet saved: {py_out}")
    print("\n" + py_block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
