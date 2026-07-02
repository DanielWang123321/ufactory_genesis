"""Static EE alignment checks between Genesis FK and xArm SDK."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from ufactory.kinematics import prepare_robot_model_for_verification, resolve_kinematics_suffix_from_ip
from ufactory.kinematics_validation import (
    PASS_POS_MM,
    PASS_RPY_DEG,
    angle_diff_deg,
    build_genesis_robot,
    genesis_fk,
    quat_to_rpy,
    validation_configs,
    _connect_sdk,
)
from ufactory.deploy.sdk_units import MM_PER_M, sdk_position_m, sdk_position_mm
from ufactory.dynamics_validation import xarm6_default_dynamics_configs
from ufactory.paths import robot_urdf
from ufactory.robot_params import get_robot_runtime_profile, robot_runtime_cli_choices


def _dynamics_pose_configs(_runtime) -> list[tuple[str, np.ndarray]]:
    return [(name, q) for name, q in xarm6_default_dynamics_configs(include_stress=False)]


def compare_genesis_sdk_fk(
    *,
    robot_key: str,
    ip: str,
    kinematics_suffix: str | None = None,
    kinematics_yaml: str | None = None,
    poses: Sequence[tuple[str, np.ndarray]] | None = None,
    backend: str = "cpu",
) -> list[dict]:
    """Compare Genesis link FK vs SDK FK for named joint configurations."""
    runtime = get_robot_runtime_profile(robot_key)
    arm = _connect_sdk(ip)
    try:
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)

        urdf_path, _ = prepare_robot_model_for_verification(
            None,
            kinematics_yaml,
            kinematics_suffix,
            None,
            default_base_urdf=robot_urdf(runtime.model.key),
            robot_name=runtime.model.robot_name,
            joint_count=runtime.model.dof,
        )
        _, robot = build_genesis_robot(urdf_path, backend=backend, show_viewer=False)
        ee_link = next(l for l in robot.links if l.name.split("/")[-1] == runtime.arm.ee_link)

        if poses is None:
            pose_list = list(validation_configs(runtime))
            pose_list.extend(_dynamics_pose_configs(runtime))
        else:
            pose_list = list(poses)

        rows: list[dict] = []
        for name, q in pose_list:
            code, pose = arm.get_forward_kinematics(
                q.tolist(),
                input_is_radian=True,
                return_is_radian=True,
            )
            if code != 0:
                rows.append({"name": name, "ok": False, "error": f"SDK FK code={code}"})
                continue
            sdk_pos_mm = sdk_position_mm(pose)
            sdk_rpy = np.asarray(pose[3:6], dtype=np.float64)
            g_pos, g_quat = genesis_fk(robot, q, int(ee_link.idx_local))
            g_pos_mm = np.asarray(g_pos, dtype=np.float64) * MM_PER_M
            g_rpy = np.asarray(quat_to_rpy(g_quat), dtype=np.float64)
            pos_mm = float(np.linalg.norm(g_pos_mm - sdk_pos_mm))
            rpy_deg = max(angle_diff_deg(a, b) for a, b in zip(g_rpy, sdk_rpy))
            ok = pos_mm < PASS_POS_MM and rpy_deg < PASS_RPY_DEG
            rows.append(
                {
                    "name": name,
                    "ok": ok,
                    "pos_mm": pos_mm,
                    "rpy_deg": rpy_deg,
                    "sdk_pos_m": sdk_position_m(pose).tolist(),
                    "genesis_pos_m": g_pos.tolist(),
                    "sdk_pos_mm": sdk_pos_mm.tolist(),
                    "genesis_pos_mm": g_pos_mm.tolist(),
                }
            )
        return rows
    finally:
        arm.disconnect()


def print_alignment_report(rows: Sequence[dict]) -> int:
    failed = 0
    for row in rows:
        if row.get("error"):
            print(f"FAIL {row['name']}: {row['error']}")
            failed += 1
            continue
        ok = bool(row["ok"])
        print(
            f"{'PASS' if ok else 'FAIL'} {row['name']}: "
            f"pos={row['pos_mm']:.2f}mm rpy={row['rpy_deg']:.2f}deg"
        )
        failed += 0 if ok else 1
    if failed:
        print(f"Alignment check failed: {failed} pose(s) out of tolerance")
        return 1
    print("All alignment checks passed")
    return 0


def cli_align(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Genesis EE FK (link6) vs xArm SDK FK for reach deploy alignment",
    )
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", required=True)
    parser.add_argument("--kinematics-suffix", default=None)
    parser.add_argument("--kinematics-yaml", default=None)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--poses",
        default="validation",
        choices=("validation", "dynamics", "all"),
        help="Pose set: validation configs, dynamics calib poses, or both",
    )
    args = parser.parse_args(argv)

    runtime = get_robot_runtime_profile(args.robot)
    if args.kinematics_yaml is None:
        suffix, sn = resolve_kinematics_suffix_from_ip(
            args.ip,
            runtime.model.robot_name,
            kinematics_suffix=args.kinematics_suffix,
            kinematics_yaml=args.kinematics_yaml,
        )
        args.kinematics_suffix = suffix
        if suffix:
            print(f"kinematics_suffix: {suffix} (auto from SN {sn})")
    poses: list[tuple[str, np.ndarray]] = []
    if args.poses in ("validation", "all"):
        poses.extend(validation_configs(runtime))
    if args.poses in ("dynamics", "all"):
        poses.extend(_dynamics_pose_configs(runtime))

    rows = compare_genesis_sdk_fk(
        robot_key=args.robot,
        ip=args.ip,
        kinematics_suffix=args.kinematics_suffix,
        kinematics_yaml=args.kinematics_yaml,
        poses=poses,
        backend=args.backend,
    )
    return print_alignment_report(rows)
