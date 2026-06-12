"""Utility to run repeated FK/IK validation cycles for kinematics calibration.

Usage examples:
  python examples/xarm6/run_fk_alignment_cycle.py \
    --robot-ip 192.168.1.60 \
    --real-ip 192.168.1.60 \
    --suffixes SUFFIX_A,SUFFIX_B \
    --gen-script scripts/gen_kinematics_params.py \
    --log-dir logs/kinematics
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run xArm6 FK alignment verification loops.")
    parser.add_argument("--robot-ip", type=str, default=None, help="Robot IP for gen_kinematics_params.py")
    parser.add_argument(
        "--real-ip",
        type=str,
        required=True,
        help="Real robot IP for verify_xarm6.py (IK/FK comparison)",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default="",
        help="Comma separated kinematics suffixes to try, e.g. SUFFIX_A,SUFFIX_B",
    )
    parser.add_argument(
        "--kinematics-yaml",
        type=str,
        default=None,
        help="Use a fixed kinematics YAML for every round, skip suffix mode.",
    )
    parser.add_argument(
        "--kinematics-yaml-dir",
        type=str,
        default=None,
        help="Directory to search when auto-finding kinematics yaml by suffix.",
    )
    parser.add_argument(
        "--gen-script",
        type=str,
        default=None,
        help=(
            "Path to xarm_ros2 gen_kinematics_params.py. "
            "If omitted, existing YAMLs are reused."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/kinematics_alignment",
        help="Directory to save per-run logs.",
    )
    parser.add_argument(
        "--max-pos-mm",
        type=float,
        default=1.0,
        help="Pass criteria threshold for position error (mm).",
    )
    parser.add_argument(
        "--max-rpy-deg",
        type=float,
        default=0.1,
        help="Pass criteria threshold for orientation error (deg).",
    )
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Skip real-robot IK verification and only evaluate FK.",
    )
    parser.add_argument(
        "--max-ik-joint-deg",
        type=float,
        default=0.20,
        help="Pass criteria threshold for max IK joint diff (deg) in comparison mode.",
    )
    parser.add_argument(
        "--max-ik-gs-err-mm",
        type=float,
        default=1.0,
        help="Pass criteria threshold for max Genesis FK error from Genesis IK (mm).",
    )
    parser.add_argument(
        "--max-ik-sdk-err-mm",
        type=float,
        default=1.0,
        help="Pass criteria threshold for max SDK FK error from SDK IK (mm).",
    )
    parser.add_argument(
        "--max-ik-fail-count",
        type=int,
        default=0,
        help="Maximum allowed IK failure count in comparison mode.",
    )
    return parser.parse_args()


def run_generator(gen_script: str, robot_ip: str, suffix: str, workspace: Path) -> Path:
    """Run xarm_ros2 gen_kinematics_params.py to generate YAML."""
    cmd = [sys.executable, gen_script, robot_ip, suffix]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(workspace),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Generate kinematics failed for suffix={suffix}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
        )

    # Most output formats include an absolute or relative saved path.
    for line in proc.stdout.splitlines():
        if "Success" in line and "save to" in line:
            path = Path(line.split("save to", 1)[1].strip())
            if path.exists():
                return path
    raise FileNotFoundError(
        f"Failed to locate generated kinematics file for suffix={suffix}. "
        f"Generator output:\n{proc.stdout}\n{proc.stderr}"
    )


def parse_metrics(log_text: str):
    max_pos = None
    max_rpy = None
    max_ik_joint_diff_deg = None
    max_ik_gs_verify_err_mm = None
    max_ik_sdk_verify_err_mm = None
    ik_fail_count = None

    num_pattern = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    fk_pos_pattern = re.compile(r"^\s*[\w\s]+\[.*\]\s+\[.*\]\s+(" + num_pattern + r")")
    fk_pos_summary_pattern = re.compile(r"^FK_SUMMARY_MAX_POS_MM\s*[:=]\s*(" + num_pattern + r")")
    rpy_pattern = re.compile(r"max RPY diff: (" + num_pattern + r")")
    rpy_summary_pattern = re.compile(r"^FK_SUMMARY_MAX_RPY_DIFF_DEG\s*[:=]\s*(" + num_pattern + r")")
    ik_joint_pattern = re.compile(r"^IK_SUMMARY_MAX_JOINT_DIFF_DEG\s*[:=]\s*(" + num_pattern + r")")
    ik_gs_err_pattern = re.compile(r"^IK_SUMMARY_MAX_GS_VERIFY_ERR_MM\s*[:=]\s*(" + num_pattern + r")")
    ik_sdk_err_pattern = re.compile(r"^IK_SUMMARY_MAX_SDK_VERIFY_ERR_MM\s*[:=]\s*(" + num_pattern + r")")
    ik_fail_pattern = re.compile(r"^IK_SUMMARY_FAIL_COUNT\s*[:=]\s*(\d+)")

    for line in log_text.splitlines():
        stripped = line.strip()
        m = fk_pos_summary_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_pos = value if max_pos is None else max(max_pos, value)
            continue

        m = rpy_summary_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_rpy = value if max_rpy is None else max(max_rpy, value)
            continue

        m = fk_pos_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_pos = value if max_pos is None else max(max_pos, value)
            continue

        m = rpy_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_rpy = value if max_rpy is None else max(max_rpy, value)
            continue

        m = ik_joint_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_ik_joint_diff_deg = (
                value if max_ik_joint_diff_deg is None else max(max_ik_joint_diff_deg, value)
            )
            continue

        m = ik_gs_err_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_ik_gs_verify_err_mm = (
                value if max_ik_gs_verify_err_mm is None else max(max_ik_gs_verify_err_mm, value)
            )
            continue

        m = ik_sdk_err_pattern.search(stripped)
        if m:
            value = float(m.group(1))
            max_ik_sdk_verify_err_mm = (
                value if max_ik_sdk_verify_err_mm is None else max(max_ik_sdk_verify_err_mm, value)
            )
            continue

        m = ik_fail_pattern.search(stripped)
        if m:
            ik_fail_count = int(float(m.group(1)))

    return (
        max_pos,
        max_rpy,
        max_ik_joint_diff_deg,
        max_ik_gs_verify_err_mm,
        max_ik_sdk_verify_err_mm,
        ik_fail_count,
    )


def run_verify(
    verify_script: Path,
    real_ip: str,
    robot_model: Path | None,
    kinematics_suffix: str | None,
    kinematics_yaml: str | None,
    kinematics_yaml_dir: str | None,
    skip_ik: bool,
) -> tuple[
    float | None, float | None, float | None, float | None, float | None, int | None, str, int
]:
    cmd = [sys.executable, str(verify_script), "--real-ip", real_ip]
    if robot_model:
        cmd.extend(["--robot-model", str(robot_model)])
    if kinematics_yaml:
        cmd.extend(["--kinematics-yaml", kinematics_yaml])
    elif kinematics_suffix:
        cmd.extend(["--kinematics-suffix", kinematics_suffix])
    if kinematics_yaml_dir:
        cmd.extend(["--kinematics-yaml-dir", kinematics_yaml_dir])
    if skip_ik:
        cmd.append("--skip-ik")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_text = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    max_pos, max_rpy, max_ik_joint_diff_deg, max_ik_gs_verify_err_mm, max_ik_sdk_verify_err_mm, ik_fail_count = parse_metrics(
        log_text
    )
    return (
        max_pos,
        max_rpy,
        max_ik_joint_diff_deg,
        max_ik_gs_verify_err_mm,
        max_ik_sdk_verify_err_mm,
        ik_fail_count,
        log_text,
        proc.returncode,
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    verify_script = repo_root / "examples" / "xarm6" / "verify_xarm6.py"
    os.makedirs(args.log_dir, exist_ok=True)
    log_dir = Path(args.log_dir)

    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    rounds = suffixes or ["baseline"]

    print(f"Will run {len(rounds)} alignment round(s): {rounds}")
    results = []

    for suffix in rounds:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = suffix if suffix else "base"
        log_file = log_dir / f"fk_alignment_{tag}_{ts}.log"
        print(f"\n== Round {tag} ==")

        yaml_path = args.kinematics_yaml if args.kinematics_yaml else None
        robot_model = None

        if args.gen_script and args.robot_ip and suffix != "baseline":
            yaml_path = str(run_generator(args.gen_script, args.robot_ip, suffix, repo_root))

        if yaml_path:
            robot_model = None
            kinematics_suffix = None
        else:
            kinematics_suffix = None if suffix == "baseline" else suffix

        (
            max_pos,
            max_rpy,
            max_ik_joint_diff_deg,
            max_ik_gs_verify_err_mm,
            max_ik_sdk_verify_err_mm,
            ik_fail_count,
            log_text,
            return_code,
        ) = run_verify(
            verify_script=verify_script,
            real_ip=args.real_ip,
            robot_model=robot_model,
            kinematics_suffix=kinematics_suffix,
            kinematics_yaml=yaml_path,
            kinematics_yaml_dir=args.kinematics_yaml_dir,
            skip_ik=args.skip_ik,
        )

        log_file.write_text(log_text, encoding="utf-8")
        valid = max_pos is not None and max_rpy is not None and return_code == 0
        if args.skip_ik:
            passed = bool(
                valid and (max_pos <= args.max_pos_mm) and (max_rpy <= args.max_rpy_deg)
            )
        else:
            passed = bool(
                valid
                and (max_pos <= args.max_pos_mm)
                and (max_rpy <= args.max_rpy_deg)
                and (max_ik_joint_diff_deg is not None)
                and (max_ik_gs_verify_err_mm is not None)
                and (max_ik_sdk_verify_err_mm is not None)
                and (ik_fail_count is not None)
                and (max_ik_joint_diff_deg <= args.max_ik_joint_deg)
                and (max_ik_gs_verify_err_mm <= args.max_ik_gs_err_mm)
                and (max_ik_sdk_verify_err_mm <= args.max_ik_sdk_err_mm)
                and (ik_fail_count <= args.max_ik_fail_count)
            )
        if max_pos is None:
            max_pos = 0.0
        if max_rpy is None:
            max_rpy = 0.0
        if max_ik_joint_diff_deg is None:
            max_ik_joint_diff_deg = 0.0
        if max_ik_gs_verify_err_mm is None:
            max_ik_gs_verify_err_mm = 0.0
        if max_ik_sdk_verify_err_mm is None:
            max_ik_sdk_verify_err_mm = 0.0
        if ik_fail_count is None:
            ik_fail_count = 0
        results.append(
            (
                suffix,
                max_pos,
                max_rpy,
                max_ik_joint_diff_deg,
                max_ik_gs_verify_err_mm,
                max_ik_sdk_verify_err_mm,
                ik_fail_count,
                passed,
                log_file,
            )
        )
        print(
            f"  valid={valid}, return_code={return_code}, "
            f"max_pos={max_pos:.2f} mm, max_rpy={max_rpy:.2f} deg, "
            f"max_ik_joint_diff={max_ik_joint_diff_deg:.2f} deg, "
            f"max_ik_gs_err={max_ik_gs_verify_err_mm:.2f} mm, "
            f"max_ik_sdk_err={max_ik_sdk_verify_err_mm:.2f} mm, "
            f"ik_fail={ik_fail_count}, "
            f"passed={passed}, log={log_file}"
        )

    print("\nSummary:")
    for (
        suffix,
        max_pos,
        max_rpy,
        max_ik_joint_diff_deg,
        max_ik_gs_verify_err_mm,
        max_ik_sdk_verify_err_mm,
        ik_fail_count,
        passed,
        log_file,
    ) in results:
        status = "PASS" if passed else "FAIL"
        print(
            f"- {suffix}: {status} | pos={max_pos:.2f} mm | rpy={max_rpy:.2f} deg | "
            f"ik_max={max_ik_joint_diff_deg:.2f} deg | ik_gs_err={max_ik_gs_verify_err_mm:.2f} mm | "
            f"ik_sdk_err={max_ik_sdk_verify_err_mm:.2f} mm | ik_fail={ik_fail_count} | {log_file}"
        )


if __name__ == "__main__":
    main()
