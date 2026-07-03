"""
xArm6 Reach — real-robot deploy, smoke tests, and EE alignment.

Usage:
    # Static Genesis vs SDK FK alignment (no motion)
    python examples/xarm6/xarm6_reach_deploy.py --mode align --ip 192.168.1.65

    # Zero-action servo smoke (50 Hz, default 50 steps)
    python examples/xarm6/xarm6_reach_deploy.py --mode smoke-zero --ip 192.168.1.65

    # Small random actions
    python examples/xarm6/xarm6_reach_deploy.py --mode smoke-random --ip 192.168.1.65 --steps 30

    # Firmware mode-6 joint online trajectory planning smoke
    python examples/xarm6/xarm6_reach_deploy.py --mode smoke-zero --ip 192.168.1.65 \\
        --executor online-joint --steps 20

    # Open-loop joint replay (position mode, safe calib poses)
    python examples/xarm6/xarm6_reach_deploy.py --mode replay --ip 192.168.1.65 --poses home,default

    # Closed-loop policy deploy (omit --checkpoint to use latest model_*.pt)
    python examples/xarm6/xarm6_reach_deploy.py --mode deploy --ip 192.168.1.65 \\
        --exp-name xarm6-reach-300-online \\
        --checkpoint logs/xarm6-reach-300-online/model_299.pt \\
        --target 0.4,0.0,0.3 --executor online-joint --action-contract checkpoint --z-min-mm 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from ufactory.deploy.obs_adapter import parse_target_xyz
from ufactory.deploy.obs_align import cli_align
from ufactory.deploy.policy_runner import ReachPolicyRunner
from ufactory.deploy.reach_config import REACH_EXECUTORS
from ufactory.deploy.session import ReachDeploySession
from ufactory.hardware.session import MOVE_STRATEGY_DIRECT
from ufactory.kinematics.calibration import resolve_kinematics_suffix_from_ip
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.hardware.xarm import MODE_POSITION

ACTION_CONTRACT_CHECKPOINT = "checkpoint"
ACTION_CONTRACT_RUNTIME = "runtime"
ACTION_CONTRACTS = (ACTION_CONTRACT_CHECKPOINT, ACTION_CONTRACT_RUNTIME)


def _default_ip(args_ip: str | None) -> str:
    ip = args_ip or os.environ.get("XARM_IP")
    if not ip:
        raise SystemExit("Set --ip or XARM_IP")
    return ip


def _resolve_kinematics_suffix(ip: str, robot_key: str, kinematics_suffix: str | None) -> str | None:
    if not ip:
        return kinematics_suffix
    runtime = get_robot_runtime_profile(robot_key)
    cli_suffix = kinematics_suffix
    suffix, sn = resolve_kinematics_suffix_from_ip(
        ip,
        runtime.model.robot_name,
        kinematics_suffix=cli_suffix,
    )
    if suffix and not cli_suffix:
        print(f"kinematics_suffix: {suffix} (auto from SN {sn})")
    return suffix


def _resolve_poses(runtime, names: str) -> list[tuple[str, np.ndarray]]:
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    configs: dict[str, np.ndarray] = {
        "home": np.asarray(runtime.arm.home_qpos, dtype=np.float64),
        "default": np.asarray(runtime.arm.default_qpos, dtype=np.float64),
    }
    for name, q in runtime.dynamics.default_configs:
        configs[name] = np.asarray(q, dtype=np.float64)
    missing = sorted(wanted - set(configs))
    if missing:
        raise SystemExit(f"Unknown pose(s): {', '.join(missing)}")
    return [(name, configs[name]) for name in names.split(",") if name.strip()]


def run_smoke_zero(session: ReachDeploySession, steps: int) -> None:
    print(
        f"smoke-zero: {steps} steps @ {session.config.ctrl_dt}s, action=0, "
        f"executor={session.config.executor}, max_delta={session.config.max_joint_delta_rad:.4f}rad"
    )
    session.configure_for_deploy()

    def _step(_sess: ReachDeploySession, _idx: int) -> np.ndarray:
        return np.zeros(session.config.num_actions, dtype=np.float64)

    session.run_control_loop(_step, steps=steps)


def run_smoke_random(session: ReachDeploySession, steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    print(
        f"smoke-random: {steps} steps, action ~ U(-0.1, 0.1), "
        f"executor={session.config.executor}, max_delta={session.config.max_joint_delta_rad:.4f}rad"
    )
    session.configure_for_deploy()

    def _step(_sess: ReachDeploySession, _idx: int) -> np.ndarray:
        return rng.uniform(-0.1, 0.1, session.config.num_actions)

    session.run_control_loop(_step, steps=steps)


def run_replay(session: ReachDeploySession, pose_names: str) -> None:
    runtime = session.runtime
    poses = _resolve_poses(runtime, pose_names)
    print(f"replay: {len(poses)} poses via position mode (direct)")
    session.configure_for_dynamics()
    session._motion_mode = MODE_POSITION
    for name, q in poses:
        print(f"  -> {name}: q={[round(float(v), 4) for v in q]}")
        session.move_to(q, wait=True, move_strategy=MOVE_STRATEGY_DIRECT)
    session.return_home()


def _list_checkpoints(log_dir: Path) -> list[Path]:
    return sorted(
        log_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def _resolve_checkpoint(log_dir: Path, checkpoint: str | Path | None) -> Path:
    """Resolve checkpoint path; rsl-rl saves 0-based iteration indices (300 iters -> model_299.pt)."""
    if checkpoint is None:
        pts = _list_checkpoints(log_dir)
        if not pts:
            raise SystemExit(f"No checkpoints in {log_dir} (run xarm6_reach_train.py first)")
        ckpt = pts[-1]
        print(f"Using latest checkpoint: {ckpt}")
        return ckpt

    ckpt = Path(checkpoint)
    if ckpt.exists():
        return ckpt

    # Common off-by-one: --max_iterations 300 saves model_299.pt, not model_300.pt
    if ckpt.name.startswith("model_") and ckpt.suffix == ".pt":
        try:
            iter_n = int(ckpt.stem.split("_")[1])
        except (IndexError, ValueError):
            iter_n = None
        if iter_n is not None and iter_n > 0:
            alt = ckpt.parent / f"model_{iter_n - 1}.pt"
            if alt.exists():
                print(
                    f"[INFO] {ckpt} not found; using {alt} "
                    f"(rsl-rl checkpoints are 0-based: iteration {iter_n - 1})"
                )
                return alt

    pts = _list_checkpoints(log_dir if ckpt.parent == Path(".") else ckpt.parent)
    if not pts and log_dir.exists():
        pts = _list_checkpoints(log_dir)
    hint = ""
    if pts:
        names = ", ".join(p.name for p in pts)
        hint = f"\nAvailable in {log_dir}: {names}\nTip: omit --checkpoint to use the latest."
    raise SystemExit(f"Missing checkpoint: {checkpoint}{hint}")


def _warn_action_contract_mismatch(checkpoint_config, runtime_config) -> None:
    ckpt_limit = float(checkpoint_config.max_joint_delta_rad)
    runtime_limit = float(runtime_config.max_joint_delta_rad)
    if np.isclose(ckpt_limit, runtime_limit, rtol=1e-6, atol=1e-9):
        return
    ratio = runtime_limit / ckpt_limit if ckpt_limit != 0.0 else float("inf")
    print(
        "[WARN] action contract delta limit mismatch: "
        f"checkpoint={ckpt_limit:.6f} runtime={runtime_limit:.6f} ratio={ratio:.3f}"
    )


def _print_action_contract(session: ReachDeploySession, source: str) -> None:
    cfg = session.config
    print(
        "action_contract: "
        f"source={source} executor={cfg.executor} robot_mode={session.arm.mode} "
        f"ctrl_dt={cfg.ctrl_dt:.4f}s action_scale={cfg.action_scale:.6f} "
        f"action_clip={cfg.action_clip:.6f} max_joint_delta_rad={cfg.max_joint_delta_rad:.6f}"
    )


def _resolve_report_csv(report_csv: str | None) -> Path | None:
    if not report_csv:
        return None
    if report_csv == "auto":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("reports") / f"reach_deploy_{stamp}.csv"
    return Path(report_csv)


def _deploy_report_fieldnames(dof: int) -> list[str]:
    names = [
        "step",
        "dist_mm",
        "ee_x_m",
        "ee_y_m",
        "ee_z_m",
        "max_abs_action",
        "action_saturation_fraction",
        "delta_saturation_fraction",
        "max_abs_dq_rad",
    ]
    names.extend(f"q_{idx + 1}" for idx in range(dof))
    names.extend(f"q_cmd_{idx + 1}" for idx in range(dof))
    names.extend(f"action_{idx + 1}" for idx in range(dof))
    return names


def _deploy_report_row(step: int, obs: np.ndarray, action: np.ndarray, q_cmd: np.ndarray, command, target: np.ndarray) -> dict:
    dof = q_cmd.size
    q = obs[:dof]
    ee_start = dof * 2
    ee = obs[ee_start:ee_start + 3]
    dist = float(np.linalg.norm(target - ee))
    row = {
        "step": step,
        "dist_mm": dist * 1000.0,
        "ee_x_m": float(ee[0]),
        "ee_y_m": float(ee[1]),
        "ee_z_m": float(ee[2]),
        "max_abs_action": float(np.abs(action).max()),
        "action_saturation_fraction": float(command.action_saturation_fraction),
        "delta_saturation_fraction": float(command.delta_saturation_fraction),
        "max_abs_dq_rad": float(np.abs(command.joint_delta).max()),
    }
    for idx, value in enumerate(q):
        row[f"q_{idx + 1}"] = float(value)
    for idx, value in enumerate(q_cmd):
        row[f"q_cmd_{idx + 1}"] = float(value)
    for idx, value in enumerate(action):
        row[f"action_{idx + 1}"] = float(value)
    return row


def run_deploy(
    session: ReachDeploySession,
    checkpoint: Path,
    log_dir: Path,
    steps: int,
    success_dist_m: float,
    action_contract: str,
    report_csv: str | None,
) -> None:
    cfgs = log_dir / "cfgs.pkl"
    if not cfgs.exists():
        raise SystemExit(f"Missing training config: {cfgs}")
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")

    policy = ReachPolicyRunner.from_checkpoint(checkpoint, cfgs, device="cpu")
    runtime_config = session.config
    _warn_action_contract_mismatch(policy.config, runtime_config)
    if action_contract == ACTION_CONTRACT_CHECKPOINT:
        session.use_action_contract(policy.config)
    elif action_contract != ACTION_CONTRACT_RUNTIME:
        raise ValueError(f"Unknown action contract: {action_contract}")

    session.configure_for_deploy()
    target = session.target_pos_m
    print(
        f"deploy: target={target.tolist()} steps={steps} success_dist<{success_dist_m}m "
        f"executor={session.config.executor}"
    )
    _print_action_contract(session, action_contract)

    report_path = _resolve_report_csv(report_csv)
    report_file = None
    writer = None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_file = report_path.open("w", newline="")
        writer = csv.DictWriter(report_file, fieldnames=_deploy_report_fieldnames(session.config.dof))
        writer.writeheader()
        print(f"deploy_report_csv: {report_path}")

    try:
        for idx in range(steps):
            t0 = time.monotonic()
            obs = session.read_reach_obs()
            action = policy.act(obs)
            q_cmd = session.step_servo(action)
            command = session.last_action_command
            if command is None:
                raise RuntimeError("Missing action command after deploy step")
            row = _deploy_report_row(idx, obs, action, q_cmd, command, target)
            if writer is not None:
                writer.writerow(row)
            if idx % 10 == 0:
                print(
                    f"  step {idx}: "
                    f"dist={row['dist_mm']:.1f} mm ee_z={row['ee_z_m'] * 1000.0:.1f} mm "
                    f"max_abs_action={row['max_abs_action']:.3f} "
                    f"action_sat={row['action_saturation_fraction']:.2f} "
                    f"delta_sat={row['delta_saturation_fraction']:.2f} "
                    f"max_abs_dq={row['max_abs_dq_rad']:.4f} rad"
                )
            elapsed = time.monotonic() - t0
            sleep_s = session.config.ctrl_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        if report_file is not None:
            report_file.close()

    ee = session.get_ee_pos_m()
    dist = float(np.linalg.norm(target - ee))
    print(f"Final EE distance: {dist * 1000:.1f} mm ({'OK' if dist < success_dist_m else 'MISS'})")


def main() -> int:
    parser = argparse.ArgumentParser(description="xArm6 Reach sim-to-real deploy")
    parser.add_argument("--mode", required=True,
                        choices=("align", "smoke-zero", "smoke-random", "replay", "deploy"))
    parser.add_argument("--ip", default=None)
    parser.add_argument("--robot", default="xarm6")
    parser.add_argument("--kinematics-suffix", default=os.environ.get("XARM_KINEMATICS_SUFFIX"))
    parser.add_argument("--z-min-mm", type=float, default=0.0)
    parser.add_argument(
        "--executor",
        default=os.environ.get("XARM_REACH_EXECUTOR", "servo-j"),
        choices=REACH_EXECUTORS,
        help="Joint command executor: servo-j uses mode 1 set_servo_angle_j; online-joint uses mode 6 set_servo_angle",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poses", default="home,default", help="Comma pose names for replay mode")
    parser.add_argument("--target", default="0.4,0.0,0.3", help="Target xyz metres for deploy/smoke")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--exp-name", default="xarm6-reach")
    parser.add_argument("--success-dist-mm", type=float, default=50.0)
    parser.add_argument("--align-poses", default="validation", choices=("validation", "dynamics", "all"))
    parser.add_argument(
        "--action-contract",
        default=ACTION_CONTRACT_CHECKPOINT,
        choices=ACTION_CONTRACTS,
        help="checkpoint uses cfgs.pkl action scale/limit; runtime uses executor defaults",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Optional deploy CSV path; use 'auto' to write reports/reach_deploy_<timestamp>.csv",
    )
    args = parser.parse_args()

    if args.mode == "align":
        ip = _default_ip(args.ip)
        kinematics_suffix = _resolve_kinematics_suffix(ip, args.robot, args.kinematics_suffix)
        argv = [
            "--robot", args.robot,
            "--ip", ip,
            "--poses", args.align_poses,
        ]
        if kinematics_suffix:
            argv.extend(["--kinematics-suffix", kinematics_suffix])
        return cli_align(argv)

    ip = _default_ip(args.ip)
    args.kinematics_suffix = _resolve_kinematics_suffix(ip, args.robot, args.kinematics_suffix)
    target_pos = parse_target_xyz(args.target)
    log_dir = Path("logs") / args.exp_name
    ckpt: Path | None = None
    if args.mode == "deploy":
        cfgs = log_dir / "cfgs.pkl"
        if not cfgs.exists():
            raise SystemExit(f"Missing training config: {cfgs}")
        ckpt = _resolve_checkpoint(log_dir, args.checkpoint)

    session = ReachDeploySession(ip, robot_key=args.robot, z_min_mm=args.z_min_mm, executor=args.executor)
    try:
        session.set_target_position(target_pos)
        if args.mode == "smoke-zero":
            run_smoke_zero(session, args.steps)
        elif args.mode == "smoke-random":
            run_smoke_random(session, args.steps, args.seed)
        elif args.mode == "replay":
            run_replay(session, args.poses)
        elif args.mode == "deploy":
            assert ckpt is not None
            run_deploy(
                session,
                ckpt,
                log_dir,
                args.steps,
                args.success_dist_mm / 1000.0,
                args.action_contract,
                args.report_csv,
            )
    finally:
        if args.mode != "replay" and session.deploy_configured:
            try:
                session.return_home()
            except Exception as exc:
                print(f"[WARN] return_home: {exc}")
        session.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
