"""Reusable FK/IK validation helpers for UFACTORY Genesis models."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ufactory.kinematics import (
    get_robot_sn,
    log_kinematics_sn_status,
    prepare_robot_model_for_verification,
    validate_kinematics_calibration_request,
)
from ufactory.paths import robot_urdf
from ufactory.robot_params import RobotRuntimeProfile, get_robot_runtime_profile, robot_runtime_cli_choices

PASS_POS_MM = 1.0
PASS_RPY_DEG = 0.5


def quat_to_rpy(quat: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) * 180.0 / math.pi


def _ensure_fk_scratch(robot) -> None:
    if getattr(robot, "_IK_qpos_orig", None) is not None:
        return
    if robot.n_qs == 0:
        return
    try:
        import genesis as gs
        import quadrants as qd
    except ImportError:
        return
    robot._IK_qpos_orig = qd.field(dtype=gs.qd_float, shape=(robot.n_qs, robot._solver._B))


def validation_configs(runtime: RobotRuntimeProfile) -> list[tuple[str, np.ndarray]]:
    dof = runtime.model.dof
    configs = [("home", np.asarray(runtime.arm.home_qpos, dtype=np.float64))]
    configs.append(("default", np.asarray(runtime.arm.default_qpos, dtype=np.float64)))
    if dof >= 5:
        configs.append(("A", np.asarray([0.5, -0.3, 0.0, 0.0, 0.3, *([0.0] * max(0, dof - 5))], dtype=np.float64)[:dof]))
    if dof >= 6:
        configs.append(("B", np.asarray([0.0, -0.5, -0.1, 0.5, 0.5, 0.0, *([0.0] * max(0, dof - 6))], dtype=np.float64)[:dof]))
    return configs


def build_genesis_robot(urdf_path: str, *, backend: str = "cpu", show_viewer: bool = False):
    import genesis as gs

    gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu)
    scene = gs.Scene(show_viewer=show_viewer)
    robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True, requires_jac_and_IK=True))
    scene.build()
    return scene, robot


def genesis_fk(robot, q: np.ndarray, ee_link_idx: int) -> tuple[np.ndarray, np.ndarray]:
    import genesis as gs

    _ensure_fk_scratch(robot)
    q_t = torch.tensor(q, dtype=torch.float32, device=gs.device)
    links_pos, links_quat = robot.forward_kinematics(qpos=q_t)
    if links_pos.ndim == 2:
        return links_pos[ee_link_idx].cpu().numpy(), links_quat[ee_link_idx].cpu().numpy()
    return links_pos[0, ee_link_idx].cpu().numpy(), links_quat[0, ee_link_idx].cpu().numpy()


def _connect_sdk(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    connect = getattr(arm, "connect", None)
    if connect is not None:
        connect()
    return arm


def _prepare_sdk_and_urdf(args, runtime: RobotRuntimeProfile) -> tuple[object, str, str | None]:
    arm = _connect_sdk(args.ip)
    sn = get_robot_sn(arm)
    validate_kinematics_calibration_request(
        sn,
        runtime.model.robot_name,
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
    )
    log_kinematics_sn_status(
        sn,
        runtime.model.robot_name,
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
    )
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    robot_model_arg = args.robot_model or args.urdf
    urdf_path, kinematics_yaml_path = prepare_robot_model_for_verification(
        robot_model_arg,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=robot_urdf(runtime.model.key),
        robot_name=runtime.model.robot_name,
        joint_count=runtime.model.dof,
    )
    return arm, urdf_path, kinematics_yaml_path


def run_fk_validation(args) -> int:
    runtime = get_robot_runtime_profile(args.robot)
    arm, urdf_path, kinematics_yaml_path = _prepare_sdk_and_urdf(args, runtime)
    print(f"Robot: {runtime.model.key}")
    print(f"URDF : {Path(urdf_path).resolve()}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")

    _, robot = build_genesis_robot(urdf_path, backend=args.backend, show_viewer=args.vis)
    ee_link = next(l for l in robot.links if l.name.split("/")[-1] == runtime.arm.ee_link)

    failed = 0
    for name, q in validation_configs(runtime):
        code, pose = arm.get_forward_kinematics(q.tolist(), input_is_radian=True, return_is_radian=True)
        if code != 0:
            raise RuntimeError(f"SDK FK failed for {name}: code={code}")
        sdk_pos = np.asarray(pose[:3], dtype=np.float64)
        sdk_rpy = np.asarray(pose[3:6], dtype=np.float64)
        g_pos, g_quat = genesis_fk(robot, q, int(ee_link.idx_local))
        g_rpy = np.asarray(quat_to_rpy(g_quat), dtype=np.float64)
        pos_mm = float(np.linalg.norm((g_pos - sdk_pos) * 1000.0))
        rpy_deg = max(angle_diff_deg(a, b) for a, b in zip(g_rpy, sdk_rpy))
        ok = pos_mm < PASS_POS_MM and rpy_deg < PASS_RPY_DEG
        print(f"{'PASS' if ok else 'FAIL'} {name}: pos={pos_mm:.2f}mm rpy={rpy_deg:.2f}deg")
        failed += 0 if ok else 1

    arm.disconnect()
    if failed:
        return 1
    print("All FK checks passed")
    return 0


def run_ik_validation(args) -> int:
    import genesis as gs

    runtime = get_robot_runtime_profile(args.robot)
    arm, urdf_path, kinematics_yaml_path = _prepare_sdk_and_urdf(args, runtime)
    print(f"Robot: {runtime.model.key}")
    print(f"URDF : {Path(urdf_path).resolve()}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")

    _, robot = build_genesis_robot(urdf_path, backend=args.backend, show_viewer=args.vis)
    ee_link = next(l for l in robot.links if l.name.split("/")[-1] == runtime.arm.ee_link)
    rng = np.random.default_rng(args.seed)
    failed = 0

    for i in range(args.samples):
        q_seed = rng.uniform(-0.5, 0.5, runtime.model.dof)
        code, pose = arm.get_forward_kinematics(q_seed.tolist(), input_is_radian=True, return_is_radian=True)
        if code != 0:
            print(f"SKIP ik_{i}: SDK FK code={code}")
            continue
        target_pos = np.asarray(pose[:3], dtype=np.float64)
        target_rpy = np.asarray(pose[3:6], dtype=np.float64)
        target_quat = np.asarray(rpy_to_quat(*target_rpy), dtype=np.float64)
        init_q = torch.tensor(q_seed, dtype=torch.float32, device=gs.device)
        result = robot.inverse_kinematics(
            link=ee_link,
            pos=torch.tensor(target_pos, dtype=gs.tc_float, device=gs.device),
            quat=torch.tensor(target_quat, dtype=gs.tc_float, device=gs.device),
            init_qpos=init_q,
            respect_joint_limit=True,
            return_error=True,
            max_solver_iters=args.max_iters,
        )
        if result is None:
            print(f"FAIL ik_{i}: no solution")
            failed += 1
            continue
        q_sol = result[0] if isinstance(result, tuple) else result
        q_np = q_sol.cpu().numpy().reshape(-1)[: runtime.model.dof]
        code2, pose2 = arm.get_forward_kinematics(q_np.tolist(), input_is_radian=True, return_is_radian=True)
        if code2 != 0:
            print(f"FAIL ik_{i}: SDK FK(solution) code={code2}")
            failed += 1
            continue
        pos_mm = float(np.linalg.norm((np.asarray(pose2[:3]) - target_pos) * 1000.0))
        rpy_deg = max(angle_diff_deg(a, b) for a, b in zip(pose2[3:6], target_rpy))
        ok = pos_mm < PASS_POS_MM and rpy_deg < PASS_RPY_DEG
        print(f"{'PASS' if ok else 'FAIL'} ik_{i}: pos={pos_mm:.2f}mm rpy={rpy_deg:.2f}deg")
        failed += 0 if ok else 1

    arm.disconnect()
    if failed:
        return 1
    print("IK checks passed")
    return 0


def _add_common_args(parser: argparse.ArgumentParser, *, default_robot: str, require_robot: bool) -> None:
    parser.add_argument("--robot", default=default_robot, required=require_robot, choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", required=True)
    parser.add_argument("--robot-model", default=None)
    parser.add_argument("--urdf", default=None, help="Alias for --robot-model")
    parser.add_argument("--kinematics-suffix", default=None)
    parser.add_argument("--kinematics-yaml", default=None)
    parser.add_argument("--kinematics-yaml-dir", default=None)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("-v", "--vis", action="store_true")


def cli_fk(argv: Sequence[str] | None = None, *, default_robot: str = "xarm6", require_robot: bool = True) -> int:
    parser = argparse.ArgumentParser(description="FK verification: Genesis URDF vs xArm Python SDK")
    _add_common_args(parser, default_robot=default_robot, require_robot=require_robot)
    args = parser.parse_args(argv)
    return run_fk_validation(args)


def cli_ik(argv: Sequence[str] | None = None, *, default_robot: str = "xarm6", require_robot: bool = True) -> int:
    parser = argparse.ArgumentParser(description="IK verification: Genesis URDF vs xArm Python SDK")
    _add_common_args(parser, default_robot=default_robot, require_robot=require_robot)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=20)
    args = parser.parse_args(argv)
    return run_ik_validation(args)
