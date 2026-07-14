"""Unified, configuration-driven grasp/place command."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from ufactory.config import RepositoryAssetStore, dump_runtime_config, load_runtime_config, resolve_grasp_object_spec
from ufactory.grippers import create_gripper_adapter
from ufactory.hardware import XArmTransport
from ufactory.hardware.xarm import wait_for_servo_motion_ready
from ufactory.kinematics import build_calibrated_urdf, get_robot_sn, load_kinematics_calibration
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW, GRIPPER_DOWN_RPY_RAD
from ufactory.safety import validate_sdk_simulation
from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend, PinocchioKinematicsBackend
from ufactory.safety.adapters.pinocchio import StageAwareObjectCollisionBackend
from ufactory.safety.gate import program_sha256, sha256_file
from ufactory.simulation import GenesisRuntimeManager
from ufactory.simulation.compat import require_genesis_capabilities
from ufactory.trajectory.execution import ExecutionBindings, execute_real, execute_sim
from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.planner import (
    CartesianWaypoint,
    JointWaypoint,
    TrajectoryPlannerConfig,
    plan_mixed_waypoints,
)
from ufactory.trajectory.preflight import create_safety_gate


G2_GRIP_SETTLE_S = 0.5
LITE6_GRIPPER_DURATION_S = 0.5
LITE6_PLACE_SETTLE_S = 0.18
# Keep in sync with ufactory.trajectory.scene grasp-gap defaults.
DEFAULT_GRIPPER_GRASP_GAP_M = 0.022
LITE6_DEFAULT_GRIPPER_GRASP_GAP_M = 0.020
OBJECT_RELOCATED_STAGES = (
    "place-descend",
    "place-settle",
    "release",
    "retreat",
    "return-home",
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def _model_and_hashes(config: Any, calibration_path: Path | None, serial_number: str | None) -> tuple[Path, str, str]:
    store = RepositoryAssetStore.discover()
    base = store.require(Path(config.robot.assets_dir) / config.robot.urdf)
    if calibration_path is None:
        calibration_hash = hashlib.sha256(f"NOMINAL:{config.robot.key}".encode()).hexdigest()
        return base, sha256_file(base), calibration_hash
    calibration = load_kinematics_calibration(
        str(calibration_path),
        robot_key=config.robot.key,
        serial_number=serial_number,
        joint_names=config.robot.joint_names,
    )
    output = Path(
        build_calibrated_urdf(
            str(base),
            calibration,
            suffix=serial_number or "offline",
            joint_count=config.robot.dof,
        )
    )
    return output, sha256_file(output), calibration.sha256


def _backends(config: Any, urdf: Path) -> tuple[Any, Any]:
    passive = {config.gripper.drive_joint: config.gripper.open_drive} if config.gripper is not None else {}
    kinematics = PinocchioKinematicsBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions=passive,
    )
    params = config.task.parameters
    object_size = tuple(map(float, params["object_size_m"]))
    object_position = tuple(map(float, params["fixed_object_position_m"]))
    target_position = tuple(map(float, params["fixed_target_position_m"]))
    obstacles = (
        EnvironmentObstacle("table", (1.2, 1.2, 0.05), (0.3, 0.0, -0.025)),
        EnvironmentObstacle("object", object_size, object_position),
    )
    collision = StageAwareObjectCollisionBackend(
        PinocchioCollisionBackend(
            urdf,
            joint_names=config.robot.joint_names,
            ee_link=config.robot.ee_link,
            passive_joint_positions=passive,
            adjacent_link_pairs=config.robot.adjacent_collision_pairs,
            obstacles=obstacles,
        ),
        spawn_center_m=object_position,
        place_center_m=target_position,
        relocated_stages=OBJECT_RELOCATED_STAGES,
    )
    return kinematics, collision


def _build_program(config: Any, kinematics: Any | None = None, *, q_home: np.ndarray | None = None) -> Any:
    """Build the v0.2.4-style grasp-place program starting at Cartesian home.

    ``start_xyz`` is always ``[obj_x, obj_y, home_z]`` (not FK of default_qpos).
    When ``q_home`` is provided (Genesis home IK), a zero-length MoveJ seed is
    prepended for ``servo_j`` compilation/preposition. Otherwise, if
    ``kinematics`` is provided, Pinocchio IK seeds a MoveJ at home for
    ``servo_cartesian`` preflight.
    """

    params = config.task.parameters
    object_pos = np.asarray(params["fixed_object_position_m"], dtype=np.float64)
    target_pos = np.asarray(params["fixed_target_position_m"], dtype=np.float64)
    from ufactory.trajectory.scene import dry_heights

    heights = dry_heights(config.robot.key)
    is_lite6 = str(getattr(config.gripper, "adapter", "")) == "lite6"
    obj_x, obj_y = float(object_pos[0]), float(object_pos[1])
    place_x, place_y = float(target_pos[0]), float(target_pos[1])
    grasp_z = float(heights.grasp_link6_z)
    pre_grasp_z = float(heights.pre_grasp_link6_z)
    lift_z = float(heights.lift_link6_z)
    home_z = float(heights.home_pos_base[2])
    open_gap = float(config.gripper.open_gap_m)
    close_gap = LITE6_DEFAULT_GRIPPER_GRASP_GAP_M if is_lite6 else DEFAULT_GRIPPER_GRASP_GAP_M
    gripper_duration_s = LITE6_GRIPPER_DURATION_S if is_lite6 else float(config.motion.gripper_duration_s)

    pre_grasp = [obj_x, obj_y, pre_grasp_z]
    grasp = [obj_x, obj_y, grasp_z]
    lift = [obj_x, obj_y, lift_z]
    place_top = [place_x, place_y, lift_z]
    place_grasp = [place_x, place_y, grasp_z]
    retreat = [place_x, place_y, lift_z]
    home = [obj_x, obj_y, home_z]

    grip_head: list[Any] = [
        {
            "type": "gripper",
            "gap_start": open_gap,
            "gap_end": close_gap,
            "duration": gripper_duration_s,
            "label": "grip",
        },
    ]
    if not is_lite6:
        grip_head.append(
            {
                "type": "gripper",
                "gap_start": close_gap,
                "gap_end": close_gap,
                "duration": G2_GRIP_SETTLE_S,
                "label": "grip-settle",
            }
        )

    place_tail: list[Any] = [
        CartesianWaypoint(place_grasp, label="place-descend"),
    ]
    if is_lite6:
        place_tail.append(
            {
                "type": "gripper",
                "gap_start": close_gap,
                "gap_end": close_gap,
                "duration": LITE6_PLACE_SETTLE_S,
                "label": "place-settle",
            }
        )
    place_tail.extend(
        [
            {
                "type": "gripper",
                "gap_start": close_gap,
                "gap_end": open_gap,
                "duration": gripper_duration_s,
                "label": "release",
            },
            CartesianWaypoint(retreat, label="retreat"),
            CartesianWaypoint(home, label="return-home"),
        ]
    )

    planner = TrajectoryPlannerConfig(
        robot_key=config.robot.key,
        rate=config.motion.rate_hz,
        speed_rad_s=config.motion.joint_speed_rad_s,
        mvacc_rad_s2=config.motion.joint_acceleration_rad_s2,
        linear_speed_m_s=config.motion.cartesian_speed_m_s,
        linear_acc_m_s2=config.motion.cartesian_acceleration_m_s2,
        z_min_m=config.safety.z_min_m,
        runtime_config=config,
    )
    waypoints: list[Any] = [
        CartesianWaypoint(pre_grasp, label="home->pregrasp"),
        CartesianWaypoint(grasp, label="descend"),
        *grip_head,
        CartesianWaypoint(lift, label="lift"),
        CartesianWaypoint(place_top, label="transit"),
        *place_tail,
    ]
    start_q = None
    if q_home is not None:
        q_home_arr = np.asarray(q_home, dtype=np.float64).reshape(-1)
        waypoints.insert(0, JointWaypoint(q_home_arr, label="start"))
        start_q = q_home_arr
    elif kinematics is not None:
        seed = np.asarray(config.arm.default_qpos_rad, dtype=np.float64).reshape(-1)
        home_pose = np.concatenate(
            (np.asarray(home, dtype=np.float64), np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64))
        )
        q_at_home = np.asarray(kinematics.inverse(home_pose, seed), dtype=np.float64)
        waypoints.insert(0, JointWaypoint(q_at_home, label="start"))
        start_q = q_at_home
    return plan_mixed_waypoints(
        planner,
        waypoints,
        start_q=start_q,
        start_xyz=home,
    )


def _home_xyz_m(config: Any) -> np.ndarray:
    from ufactory.trajectory.scene import dry_heights

    params = config.task.parameters
    object_pos = np.asarray(params["fixed_object_position_m"], dtype=np.float64)
    heights = dry_heights(config.robot.key)
    return np.asarray(
        [float(object_pos[0]), float(object_pos[1]), float(heights.home_pos_base[2])],
        dtype=np.float64,
    )


def _arm_q_from_scene(ctx: Any) -> np.ndarray:
    q_full = np.asarray(ctx.home_qpos, dtype=np.float64).reshape(-1)
    arm_idx = list(ctx.arm_dof_idx)
    return q_full[arm_idx].copy()


def _build_ik_scene(config: Any, *, calibration: Path | None, show_viewer: bool = False) -> Any:
    from ufactory.trajectory.scene import build_scene

    return build_scene(
        robot_key=config.robot.key,
        rate=config.motion.rate_hz,
        show_viewer=show_viewer,
        substeps=config.simulation.substeps,
        solver_iterations=config.simulation.solver_iterations,
        noslip_iterations=config.simulation.noslip_iterations,
        constraint_timeconst=config.simulation.constraint_time_constant_s,
        use_gjk_collision=config.simulation.use_gjk_collision,
        kinematics_yaml=str(calibration) if calibration is not None else None,
        **_scene_layout_kwargs(config),
    )


def _compile_servo_j_program(
    config: Any,
    *,
    calibration: Path | None,
    show_viewer: bool = False,
) -> tuple[Any, Any, np.ndarray]:
    """Build Cartesian program, Genesis-compile with down_quat, return (program, ctx, q_home)."""

    ctx = _build_ik_scene(config, calibration=calibration, show_viewer=show_viewer)
    q_home = _arm_q_from_scene(ctx)
    seeded = _build_program(config, q_home=q_home)
    compiled = compile_cartesian_program_to_joint_stream(seeded, ctx)
    return compiled, ctx, q_home


def _write_json(path: Path | None, value: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _connect(ip: str) -> Any:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    if not arm.connected:
        connect = getattr(arm, "connect", None)
        if callable(connect):
            connect()
    if not arm.connected:
        raise RuntimeError(f"cannot connect to {ip}")
    return arm


def _read_q(arm: Any, dof: int) -> np.ndarray:
    code, values = arm.get_servo_angle(is_radian=True)
    q = np.asarray(values, dtype=np.float64).reshape(-1)[:dof]
    if code != 0 or q.shape != (dof,) or not np.all(np.isfinite(q)):
        raise RuntimeError(f"invalid joint feedback (SDK code={code})")
    return q


def _ensure_program_start(
    transport: XArmTransport,
    q_start: np.ndarray,
    *,
    allow_preposition: bool,
    speed_rad_s: float = 0.35,
    mvacc_rad_s2: float = 2.0,
    tolerance_rad: float = 0.02,
) -> np.ndarray:
    """Require the arm at the planned joint start; optionally preposition with set_servo_angle."""

    current = transport.read_state().q_rad
    target = np.asarray(q_start, dtype=np.float64).reshape(-1)
    if current.size < target.size or not np.all(np.isfinite(current[: target.size])):
        raise RuntimeError("invalid joint feedback before program start")
    err = float(np.max(np.abs(current[: target.size] - target)))
    if err <= tolerance_rad:
        print(f"  [preposition] already at program start max_err={err:.4f}rad")
        return current[: target.size].copy()
    if not allow_preposition:
        raise RuntimeError(
            f"arm is {err:.4f} rad from program start (limit {tolerance_rad:.4f} rad); "
            "move there manually or pass --confirm-real to allow MODE_POSITION set_servo_angle preposition"
        )
    reached_err = transport.preposition_joints(
        target,
        speed_rad_s=speed_rad_s,
        mvacc_rad_s2=mvacc_rad_s2,
        tolerance_rad=tolerance_rad,
    )
    print(f"  [preposition] mode=0 joint target reached max_err={reached_err:.4f}rad")
    return transport.read_state().q_rad[: target.size].copy()


def _ensure_cartesian_program_start(
    transport: XArmTransport,
    xyz_m: np.ndarray,
    *,
    allow_preposition: bool,
    speed_mm_s: float = 100.0,
    mvacc_mm_s2: float = 500.0,
    tolerance_mm: float = 2.0,
) -> None:
    """Require the arm at Cartesian home (gripper-down); optionally preposition with set_position."""

    code, reported = transport.arm.get_position(is_radian=True)
    target = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    if int(code) != 0:
        raise RuntimeError(f"invalid Cartesian feedback before program start (SDK code={code})")
    reported_arr = np.asarray(reported, dtype=np.float64).reshape(-1)
    xyz_err = float(np.linalg.norm(reported_arr[:3] - target * 1000.0))
    if xyz_err <= tolerance_mm:
        print(f"  [preposition] already at Cartesian home xyz_err={xyz_err:.2f}mm")
        return
    if not allow_preposition:
        raise RuntimeError(
            f"arm is {xyz_err:.2f} mm from Cartesian home (limit {tolerance_mm:.2f} mm); "
            "move there manually or pass --confirm-real to allow MODE_POSITION set_position preposition"
        )
    xyz_err, rpy_err = transport.preposition_cartesian(
        target,
        speed_mm_s=speed_mm_s,
        mvacc_mm_s2=mvacc_mm_s2,
        tolerance_mm=tolerance_mm,
    )
    print(f"  [preposition] mode=0 Cartesian home reached xyz_err={xyz_err:.2f}mm rpy_err={rpy_err:.4f}rad")


def _sdk_sim_feedback(arm: Any, program: Any, config: Any, expected_start_q: np.ndarray) -> np.ndarray:
    """Replay against the controller's virtual robot from the approved start.

    The virtual preposition happens only after SDK simulation mode is enabled.
    This keeps ``sdk-sim`` independent of the physical arm pose and prevents a
    validation-only command from moving the real robot. The controller is
    always returned to non-simulation mode, including on replay failures.
    """

    target = np.asarray(expected_start_q, dtype=np.float64).reshape(-1)
    if target.shape != (config.robot.dof,) or not np.all(np.isfinite(target)):
        raise ValueError(f"SDK simulation start must contain {config.robot.dof} finite joint values")

    def require_ok(label: str, code: int) -> None:
        if int(code) != 0:
            raise RuntimeError(f"SDK simulation {label} failed with code {code}")

    simulation_requested = bool(getattr(arm, "is_simulation_robot", False))
    try:
        if not simulation_requested:
            # The SDK deliberately suppresses motion_enable once its report
            # stream says simulation mode is active. Prepare readiness first,
            # then switch modes and wait for that report bit to arrive.
            require_ok("motion_enable", arm.motion_enable(enable=True))
            require_ok("set_mode(position)", arm.set_mode(0))
            require_ok("set_state", arm.set_state(0))
            require_ok("enable mode", arm.set_simulation_robot(True))
            simulation_requested = True
        deadline = time.monotonic() + 2.0
        while not bool(getattr(arm, "is_simulation_robot", False)) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not bool(getattr(arm, "is_simulation_robot", False)):
            raise RuntimeError("SDK simulation mode was not confirmed by the controller report stream")

        require_ok("set_mode(position)", arm.set_mode(0))
        require_ok("set_state", arm.set_state(0))
        require_ok(
            "virtual preposition",
            arm.set_servo_angle(
                angle=target.tolist(),
                speed=0.35,
                mvacc=2.0,
                wait=True,
                is_radian=True,
            ),
        )
        virtual_start = _read_q(arm, config.robot.dof)
        start_error = float(np.max(np.abs(virtual_start - target)))
        if start_error > 0.02:
            raise RuntimeError(f"SDK simulation virtual start error {start_error:.4f} rad exceeds 0.0200 rad")
        require_ok("set_mode(servo)", arm.set_mode(1))
        require_ok("set_state", arm.set_state(0))
        wait_for_servo_motion_ready(arm)
        print(f"  [sdk-sim] virtual start reached max_err={start_error:.4f}rad; physical arm was not prepositioned")
        rows = [virtual_start]
        orientation = GRIPPER_DOWN_RPY_RAD
        dt = 1.0 / program.rate
        for segment in program.segments:
            samples, _ = segment.samples(program.rate)
            for target in samples:
                if segment.kind == "movej":
                    code = arm.set_servo_angle_j(angles=target.tolist(), speed=0, mvacc=0, mvtime=0, is_radian=True)
                elif segment.kind == "movel":
                    pose = [*(target * 1000.0).tolist(), *orientation]
                    code = arm.set_servo_cartesian(mvpose=pose, speed=0, mvacc=0, mvtime=0, is_radian=True)
                else:
                    code = 0
                if int(code) != 0:
                    arm.set_state(4)
                    raise RuntimeError(f"SDK simulation command failed with code {code}")
                time.sleep(dt)
                rows.append(_read_q(arm, config.robot.dof))
        return np.stack(rows)
    finally:
        if simulation_requested:
            cleanup_failures: list[str] = []
            pause_code = int(arm.set_state(3))
            if pause_code != 0:
                cleanup_failures.append(f"pause failed with code {pause_code}")
            disable_code = int(arm.set_simulation_robot(False))
            if disable_code != 0:
                cleanup_failures.append(f"disable mode failed with code {disable_code}")
            deadline = time.monotonic() + 2.0
            while bool(getattr(arm, "is_simulation_robot", False)) and time.monotonic() < deadline:
                time.sleep(0.05)
            if bool(getattr(arm, "is_simulation_robot", False)):
                cleanup_failures.append("controller report still says simulation mode is active")
            if cleanup_failures:
                raise RuntimeError("SDK simulation cleanup failed: " + "; ".join(cleanup_failures))


def _validate_robot_identity(arm: Any, config: Any, serial: str) -> None:
    if not serial or len(serial) < 8:
        raise RuntimeError("controller did not provide a complete serial number")
    axis = int(getattr(arm, "axis", config.robot.dof))
    if axis != config.robot.dof:
        raise RuntimeError(f"connected robot DOF {axis} does not match {config.robot.dof}")
    if int(getattr(arm, "error_code", 0)) != 0:
        raise RuntimeError("controller has an active error; inspect it manually in UFACTORY Studio")


def _print_summary(config: Any) -> None:
    print("configuration_sources:")
    for source in config.sources:
        print(f"  - {source}")
    print(f"robot={config.robot.key} rate_hz={config.motion.rate_hz} z_min_m={config.safety.z_min_m}")
    print(f"config_sha256={config.sha256}")


def _program_sample_count(program: Any) -> int:
    count = 0
    has_joint_state = False
    for segment in program.segments:
        count += int(segment.samples(program.rate)[1])
        if segment.kind == "movej":
            if not has_joint_state and segment.q_start is not None:
                count += 1
            has_joint_state = True
    return count


def _print_ik_compile_complete(program: Any, *, elapsed_s: float) -> None:
    print(
        f"[ik-compile] complete samples={_program_sample_count(program)} elapsed_s={elapsed_s:.2f}",
        flush=True,
    )


def _print_preflight_start(program: Any, config: Any) -> None:
    print(
        f"[preflight] checking samples={_program_sample_count(program)} "
        f"collision_margin_mm={float(config.safety.min_collision_distance_m) * 1000.0:.1f}...",
        flush=True,
    )


def _print_preflight_complete(preflight: Any) -> None:
    checks = {check.name: check for check in preflight.checks}
    collision_check = checks.get("collision")
    total_check = checks.get("total")
    collision_s = float(collision_check.duration_ms) / 1000.0 if collision_check is not None else 0.0
    total_s = float(total_check.duration_ms) / 1000.0 if total_check is not None else 0.0
    print(
        f"[preflight] complete status={'PASS' if preflight.passed else 'FAIL'} "
        f"collision_s={collision_s:.3f} total_s={total_s:.3f}",
        flush=True,
    )


def _print_preflight_violation_summary(preflight: Any, report_path: Path | None) -> None:
    """Print every violation group without flooding the terminal per sample."""

    groups: dict[tuple[str, str, str], list[Any]] = {}
    for violation in preflight.violations:
        subject = violation.link or violation.joint or "-"
        key = (violation.type.value, violation.stage, subject)
        groups.setdefault(key, []).append(violation)
    print(
        f"violation_summary=grouped total={len(preflight.violations)} groups={len(groups)}",
        file=sys.stderr,
    )
    for (kind, stage, subject), items in sorted(groups.items()):
        samples = [item.sample_index for item in items if item.sample_index is not None]
        actuals = [float(item.actual) for item in items if item.actual is not None]
        limits = [float(item.limit) for item in items if item.limit is not None]
        sample_text = "n/a" if not samples else f"{min(samples)}-{max(samples)}"
        actual_text = "n/a" if not actuals else f"{min(actuals):.9g}"
        limit_text = "n/a" if not limits else f"{limits[0]:.9g}"
        if kind in {"clearance", "self_collision", "environment_collision"}:
            measurement_text = f"min_distance_m={actual_text} threshold_m={limit_text}"
        else:
            measurement_text = f"min_actual={actual_text} limit={limit_text}"
        print(
            f"  {kind}: {stage}: {subject} count={len(items)} samples={sample_text} {measurement_text}",
            file=sys.stderr,
        )
    if report_path is None:
        print("  full per-sample details: rerun with --report PATH", file=sys.stderr)
    else:
        print(f"  full per-sample details: {report_path}", file=sys.stderr)


def _close_gap_m(config: Any) -> float:
    adapter = str(getattr(config.gripper, "adapter", ""))
    return LITE6_DEFAULT_GRIPPER_GRASP_GAP_M if adapter == "lite6" else DEFAULT_GRIPPER_GRASP_GAP_M


def _scene_layout_kwargs(config: Any) -> dict[str, Any]:
    """Map runtime task layout into ``build_scene`` kwargs (base frame)."""

    params = config.task.parameters
    object_pos = tuple(map(float, params["fixed_object_position_m"]))
    target_pos = tuple(map(float, params["fixed_target_position_m"]))
    object_spec = resolve_grasp_object_spec(config)
    return {
        "obj_pos_base": object_pos,
        "place_pos_base": target_pos,
        "obj_size": object_spec.size_m,
        "obj_mass_kg": object_spec.mass_kg,
    }


def _hold_mirror_at_program_start(mirror: Any, *, hold_s: float) -> None:
    if hold_s <= 0.0:
        return
    tick_s = 1.0 / float(mirror.rate)
    steps = max(1, int(np.ceil(float(hold_s) / tick_s)))
    next_deadline = time.monotonic()
    for _ in range(steps):
        mirror.hold_step()
        next_deadline += tick_s
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def _warm_real_visual_tracker(tracker: Any) -> None:
    """Warm lazy Genesis mirror kernels before hardware motion is enabled."""

    print("[visual-warmup] compiling kinematic mirror object updates...")
    started = time.monotonic()
    tracker.warm_up()
    print(f"[visual-warmup] complete elapsed_s={time.monotonic() - started:.2f}")


def _hold_viewer(ctx: Any, on_step: Any = None) -> None:
    """Keep the viewer open until Ctrl+C; re-teleport when ``on_step`` is set."""

    visualizer = getattr(ctx.scene, "visualizer", None)
    viewer = getattr(visualizer, "viewer", None)
    if viewer is None:
        raise RuntimeError("viewer is unavailable")

    print("Viewer open. Press Ctrl+C to exit...")
    try:
        while viewer.is_alive():
            started = time.monotonic()
            try:
                if on_step is not None:
                    on_step()
                else:
                    ctx.scene.step()
            except Exception:
                if not viewer.is_alive():
                    break
                raise
            if on_step is not None:
                # Kinematic real mirrors deliberately disable Genesis' own
                # scene-time pacer. Keep the post-run hold responsive without
                # hammering the renderer as fast as the CPU allows.
                sleep_s = (1.0 / 30.0) - (time.monotonic() - started)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ufactory-grasp-place")
    parser.add_argument("--robot", required=True, choices=("xarm5", "xarm6", "xarm7", "uf850", "lite6"))
    parser.add_argument("--mode", required=True, choices=("sim", "dry-run", "sdk-sim", "real"))
    parser.add_argument("--executor", required=True, choices=("servo_j", "servo_cartesian"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--confirm-real", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Sim: force Genesis viewer; real: open kinematic mirror viewer (async, off servo path)",
    )
    parser.add_argument(
        "--visual-start-hold-s",
        type=float,
        default=0.5,
        help="With --visual on real mode: hold the initial mirror pose before streaming (seconds)",
    )
    args = parser.parse_args(argv)
    if args.visual_start_hold_s < 0.0:
        parser.error("--visual-start-hold-s must be non-negative")
    if args.visual and args.mode in {"dry-run", "sdk-sim"}:
        parser.error("--visual is only supported with --mode sim or --mode real")

    config = load_runtime_config(args.robot, config_path=args.config)
    if args.print_config:
        print(dump_runtime_config(config), end="")
        return 0
    if args.executor == "servo_j" or args.mode == "sim" or args.visual:
        require_genesis_capabilities(
            pbr=True,
            deferred_viewer=bool(args.visual) or bool(config.simulation.show_viewer),
        )
    _print_summary(config)
    ip = args.ip or os.environ.get("XARM_IP")
    arm = None
    serial = None
    if args.mode in {"sdk-sim", "real"}:
        if not ip:
            parser.error("--ip or XARM_IP is required for sdk-sim/real")
        if args.calibration is None:
            parser.error("--calibration is required for sdk-sim/real")
        arm = _connect(ip)
        serial = get_robot_sn(arm)
        _validate_robot_identity(arm, config, serial)
        print(
            "[hardware] identity verified; motion remains disabled until preflight PASS",
            flush=True,
        )
    try:
        urdf, urdf_hash, calibration_hash = _model_and_hashes(config, args.calibration, serial)
        kinematics, collision = _backends(config, urdf)
        ik_ctx = None
        q_home: np.ndarray | None = None
        genesis_for_ik = None
        try:
            if args.executor == "servo_j":
                print(
                    "[ik-compile] building Genesis scene and compiling servo_j trajectory...",
                    flush=True,
                )
                ik_started = time.perf_counter()
                show_viewer_for_ik = args.mode == "sim" and (bool(args.visual) or bool(config.simulation.show_viewer))
                genesis_for_ik = GenesisRuntimeManager(config.simulation)
                genesis_for_ik.__enter__()
                program, ik_ctx, q_home = _compile_servo_j_program(
                    config,
                    calibration=args.calibration,
                    show_viewer=show_viewer_for_ik,
                )
                _print_ik_compile_complete(program, elapsed_s=time.perf_counter() - ik_started)
            else:
                program = _build_program(config, kinematics)
            scene_hash = _json_sha256(
                {"task": dict(config.task.parameters), "contacts": dict(config.task.allowed_contacts)}
            )
            gate = create_safety_gate(
                config,
                kinematics=kinematics,
                collision=collision,
                calibration_sha256=calibration_hash,
                scene_sha256=scene_hash,
                urdf_path=urdf,
            )
            _print_preflight_start(program, config)
            preflight = gate.preflight(program, executor=args.executor)
            _write_json(args.report, preflight.to_dict())
            print(f"program_sha256={preflight.program_sha256}")
            print(f"preflight={'PASS' if preflight.passed else 'FAIL'}")
            _print_preflight_complete(preflight)
            if not preflight.passed:
                _print_preflight_violation_summary(preflight, args.report)
                return 1
            if args.mode == "dry-run":
                return 0
            if args.mode == "sim":
                approved = gate.approve_simulation(program, preflight)
                from ufactory.trajectory.sim_executor import replay_sim

                show_viewer = bool(args.visual) or bool(config.simulation.show_viewer)
                if ik_ctx is not None:
                    sim_report = execute_sim(approved, lambda checked: replay_sim(checked, ik_ctx))
                    if show_viewer:
                        _hold_viewer(ik_ctx)
                else:
                    with GenesisRuntimeManager(config.simulation):
                        ctx = _build_ik_scene(config, calibration=None, show_viewer=show_viewer)
                        sim_report = execute_sim(approved, lambda checked: replay_sim(checked, ctx))
                        if show_viewer:
                            _hold_viewer(ctx)
                print(
                    f"place_error_mm={sim_report.place_error_mm:.3f} "
                    f"home_drift_mm={sim_report.home_drift_mm:.3f} "
                    f"metric_settle_ticks={sim_report.metric_settle_ticks}"
                )
                return 0 if sim_report.place_error_mm < 50.0 and sim_report.home_drift_mm < 10.0 else 1

            assert arm is not None and serial is not None
            home_xyz = _home_xyz_m(config)
            transport = XArmTransport(arm, robot_key=config.robot.key, serial_number=serial)
            shadow = None
            stages = None
            if args.mode == "sdk-sim":
                shadow, stages = gate.shadow_joint_stream(program)
                start_q = shadow[0].copy()
            else:
                print(
                    "[hardware] preflight approved; --confirm-real now permits start prepositioning",
                    flush=True,
                )
                if args.executor == "servo_j":
                    assert q_home is not None
                    start_q = _ensure_program_start(
                        transport,
                        q_home,
                        allow_preposition=bool(args.confirm_real),
                    )
                else:
                    _ensure_cartesian_program_start(
                        transport,
                        home_xyz,
                        allow_preposition=bool(args.confirm_real),
                    )
                    start_q = transport.read_state().q_rad[: config.robot.dof].copy()

            evidence = None
            if args.executor == "servo_cartesian" or args.mode == "sdk-sim":
                if shadow is None or stages is None:
                    shadow, stages = gate.shadow_joint_stream(program)
                firmware = _sdk_sim_feedback(arm, program, config, start_q)
                evidence = validate_sdk_simulation(
                    robot_key=config.robot.key,
                    serial_number=serial,
                    program_sha256=program_sha256(program),
                    config_sha256=config.sha256,
                    shadow_joint_stream_rad=shadow,
                    firmware_joint_stream_rad=firmware,
                    stages=stages,
                    policy=config.safety,
                    kinematics=kinematics,
                    collision=collision,
                    allowed_collision=gate.collision_allowed,
                )
                evidence_path = args.report or Path("reports") / f"sdk_sim_{config.robot.key}.json"
                _write_json(evidence_path, asdict(evidence))
                print(f"sdk_sim={'PASS' if evidence.passed else 'FAIL'}")
                if not evidence.passed:
                    return 1
                if args.mode == "sdk-sim":
                    return 0
                time.sleep(0.2)
                physical_q = _read_q(arm, config.robot.dof)
                if float(np.max(np.abs(physical_q - start_q))) > 0.01:
                    raise RuntimeError("cannot prove SDK simulation and real run share the same physical start")

            if not args.confirm_real:
                raise RuntimeError("real motion requires --confirm-real")
            approved = gate.approve(
                program,
                preflight,
                expected_serial_number=serial,
                sdk_evidence=evidence,
            )

            bridge = None
            tracker = None
            mirror = None
            mirror_ctx = None
            try:
                if args.visual:
                    from ufactory.trajectory.mirror_executor import (
                        AsyncMirrorBridge,
                        KinematicCarryTracker,
                        TrajKinematicMirror,
                    )
                    from ufactory.visualization import start_deferred_viewer

                    if ik_ctx is not None:
                        mirror_ctx = ik_ctx
                    else:
                        if genesis_for_ik is None:
                            genesis_for_ik = GenesisRuntimeManager(config.simulation)
                            genesis_for_ik.__enter__()
                        mirror_ctx = _build_ik_scene(config, calibration=args.calibration, show_viewer=False)
                    base_mm = [v * 1000.0 for v in mirror_ctx.base_pos_world]
                    print(
                        f"\n[mirror] robot={mirror_ctx.robot_key} gripper={mirror_ctx.gripper.family} "
                        f"rate={config.motion.rate_hz}Hz visual_model={mirror_ctx.visual_model} "
                        f"base=[{base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f}]mm "
                        f"kinematic open-loop; cube kinematically carried while gripped "
                        f"(no contact physics; async mirror off servo critical path)"
                    )
                    mirror = TrajKinematicMirror(mirror_ctx, approved.program)
                    mirror.prime_to_home()
                    start_deferred_viewer(mirror_ctx.scene, kinematic_mirror=True)
                    _hold_mirror_at_program_start(mirror, hold_s=args.visual_start_hold_s)
                    tracker = KinematicCarryTracker(
                        mirror,
                        grasp_gap_m=_close_gap_m(config),
                        grasp_segment_label="grip",
                        release_segment_label="release",
                        approach_freeze_labels=("descend",),
                    )
                    _warm_real_visual_tracker(tracker)
                    bridge = AsyncMirrorBridge(tracker)

                transport.authorize_motion(mode=1)
                first_joint = None
                for segment in approved.program.segments:
                    if segment.kind != "movej":
                        continue
                    samples, _ = segment.samples(approved.program.rate)
                    if len(samples) == 0:
                        continue
                    first_joint = np.asarray(samples[0], dtype=np.float64)
                    break
                if first_joint is not None:
                    transport.prime_joint_stream(first_joint)
                elif args.executor == "servo_j":
                    raise RuntimeError("approved servo_j program has no MoveJ sample to prime MODE_SERVO")
                if mirror is not None:
                    mirror.prime_to_first_arm_segment_start(approved.program)
                gripper_adapter = create_gripper_adapter(config.gripper)
                if gripper_adapter.capabilities.real_command:
                    prepare_code = gripper_adapter.prepare_real(arm)
                    if prepare_code != 0:
                        transport.stop()
                        raise RuntimeError(
                            f"gripper preparation failed with SDK code {prepare_code}; controller stopped"
                        )
                current_drive = [config.gripper.open_drive]

                def send_gripper(gap_m: float) -> int:
                    current_drive[0] = gripper_adapter.gap_to_drive(gap_m)
                    return gripper_adapter.send_real_gap(arm, gap_m)

                gripper_sender = send_gripper if gripper_adapter.capabilities.real_command else None
                execution = execute_real(
                    approved,
                    transport=transport,
                    bindings=ExecutionBindings(
                        robot_key=config.robot.key,
                        serial_number=serial,
                        config_sha256=config.sha256,
                        urdf_sha256=urdf_hash,
                        calibration_sha256=calibration_hash,
                        scene_sha256=scene_hash,
                    ),
                    confirm_real=True,
                    runtime_monitor=lambda q, stage: gate.runtime_monitor(
                        q,
                        stage,
                        gripper_drive=current_drive[0],
                        include_collision=False,
                    ),
                    gripper_sender=gripper_sender,
                    on_tick=bridge.on_tick if bridge is not None else None,
                    not_ready_codes=frozenset((9,)),
                )
                print(
                    f"real_execution={'PASS' if execution.completed else 'FAIL'} sent_samples={execution.sent_samples}"
                )
                if execution.fault is not None:
                    print(
                        f"fault={execution.fault.fault_class}: {execution.fault.message} "
                        f"sample_index={execution.fault.sample_index}"
                    )
                print(
                    f"timing_minor_misses={execution.minor_lateness_count} "
                    f"timing_max_lateness_ms={execution.max_lateness_ns / 1_000_000.0:.3f}"
                )
                if args.visual and mirror_ctx is not None and tracker is not None:
                    _hold_viewer(mirror_ctx, on_step=tracker.hold_step)
                return 0 if execution.completed else 1
            finally:
                if bridge is not None:
                    bridge.close()
        finally:
            if genesis_for_ik is not None:
                genesis_for_ik.__exit__(None, None, None)
    finally:
        if arm is not None and getattr(arm, "connected", False):
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
