"""Safe configuration-driven packaging showcase command."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from ufactory.cli.pick_place import (
    _connect,
    _ensure_cartesian_program_start,
    _ensure_program_start,
    _model_and_hashes,
    _print_ik_compile_complete,
    _print_preflight_complete,
    _print_preflight_start,
    _print_preflight_violation_summary,
    _print_summary,
    _read_q,
    _sdk_sim_feedback,
    _validate_robot_identity,
    _write_json,
)
from ufactory.config import dump_runtime_config, load_runtime_config
from ufactory.grippers import create_gripper_adapter
from ufactory.hardware import XArmTransport
from ufactory.kinematics import get_robot_sn
from ufactory.manipulation.packaging.core import (
    OBJECT_RELOCATED_STAGES,
    build_packaging_program,
    packaging_layout,
    packaging_obstacles,
    packaging_scene_sha256,
    validate_payload_box_clearance,
)
from ufactory.robots.registry import robot_cli_choices
from ufactory.safety import validate_sdk_simulation
from ufactory.safety.adapters import PinocchioCollisionBackend, PinocchioKinematicsBackend
from ufactory.safety.adapters.pinocchio import StageAwareObjectCollisionBackend
from ufactory.safety.gate import program_sha256
from ufactory.simulation import GenesisRuntimeManager
from ufactory.simulation.compat import require_genesis_capabilities
from ufactory.trajectory.execution import ExecutionBindings, execute_real
from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.preflight import create_safety_gate


def _build_packaging_context(config: Any, urdf: Path, *, show_viewer: bool = False) -> Any:
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.manipulation.packaging.simulation import (
        _trajectory_context,
        init_showcase_robot,
        stiffen_gripper_mimic_constraints,
    )

    scene, robot, block, display_layout = build_packaging_scene(
        sim_dt=1.0 / float(config.motion.rate_hz),
        show_viewer=show_viewer,
        robot_urdf_path=str(urdf),
        runtime_config=config,
    )
    stiffen_gripper_mimic_constraints(robot)
    standard_offset = (0.0, 0.0)
    showcase_ctx = init_showcase_robot(
        robot,
        display_layout,
        scene,
        runtime_config=config,
        cartesian_xy_offset_m=standard_offset,
        arm_kp_scale=1.0,
    )
    traj_ctx = _trajectory_context(
        scene,
        robot,
        block,
        display_layout,
        showcase_ctx,
        config,
        cartesian_xy_offset_m=standard_offset,
    )
    return traj_ctx


def _backends(config: Any, urdf: Path) -> tuple[Any, Any]:
    if config.gripper is None:
        raise ValueError("packaging requires a configured gripper")
    layout = packaging_layout(config)
    validate_payload_box_clearance(layout, margin_m=float(config.safety.min_collision_distance_m))
    passive = {config.gripper.drive_joint: config.gripper.open_drive}
    kinematics = PinocchioKinematicsBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions=passive,
    )
    collision = StageAwareObjectCollisionBackend(
        PinocchioCollisionBackend(
            urdf,
            joint_names=config.robot.joint_names,
            ee_link=config.robot.ee_link,
            passive_joint_positions=passive,
            adjacent_link_pairs=config.robot.adjacent_collision_pairs,
            obstacles=packaging_obstacles(layout),
        ),
        spawn_center_m=layout.object_position_m,
        place_center_m=layout.target_position_m,
        relocated_stages=OBJECT_RELOCATED_STAGES,
    )
    return kinematics, collision


def _packaging_simulation_main(argv: list[str]) -> int:
    from ufactory.manipulation.packaging.simulation import main as packaging_simulation_main

    return int(packaging_simulation_main(argv))


def _run_sim(args: argparse.Namespace) -> int:
    sim_args = [
        "--robot",
        getattr(args, "robot", "xarm6"),
        "--speed",
        str(args.speed),
        "--executor",
        args.executor,
    ]
    if args.cycles is not None:
        sim_args.extend(("--cycles", str(args.cycles)))
    elif args.loop is not None:
        sim_args.append("--loop" if args.loop else "--no-loop")
    if args.table_height is not None:
        sim_args.extend(("--table-height", str(args.table_height)))
    if args.config is not None:
        sim_args.extend(("--config", str(args.config)))
    if args.capture_keyframes:
        sim_args.append("--capture-keyframes")
    return _packaging_simulation_main(sim_args)


def _positive_cycles(value: str) -> int:
    cycles = int(value)
    if cycles < 1:
        raise argparse.ArgumentTypeError("cycles must be at least 1")
    return cycles


def _sdk_evidence_path(report: Path | None, robot_key: str) -> Path:
    """Resolve the SDK-simulation evidence path without an xArm6 default."""

    return report or Path("reports") / f"sdk_sim_{robot_key}_packaging.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ufactory-packaging-showcase")
    parser.add_argument("--robot", default="xarm6", choices=robot_cli_choices())
    parser.add_argument("--mode", default="sim", choices=("sim", "dry-run", "sdk-sim", "real"))
    parser.add_argument("--executor", default="servo_j", choices=("servo_j", "servo_cartesian"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--confirm-real", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation-only playback multiplier")
    repetition = parser.add_mutually_exclusive_group()
    repetition.add_argument("--cycles", type=_positive_cycles, default=None, help="Simulation cycle count (default: 1)")
    repetition.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Simulation-only infinite loop; --no-loop is one cycle",
    )
    parser.add_argument("--table-height", type=float, default=None)
    parser.add_argument("--capture-keyframes", action="store_true")
    args = parser.parse_args(argv)

    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.mode != "sim" and (
        args.speed != 1.0
        or args.cycles is not None
        or args.loop is True
        or args.table_height is not None
        or args.capture_keyframes
    ):
        parser.error("--speed/--cycles/--loop/--table-height/--capture-keyframes are simulation-only")
    if args.mode == "real" and not args.confirm_real:
        parser.error("real packaging motion requires --confirm-real")
    if args.visual and args.mode in {"dry-run", "sdk-sim"}:
        parser.error("--visual is only supported with --mode sim or --mode real")
    if args.mode == "real":
        args.loop = False
    config = load_runtime_config(args.robot, task="packaging_showcase", config_path=args.config)
    if args.print_config:
        print(dump_runtime_config(config), end="")
        return 0
    if args.mode == "real" and (config.gripper is None or not config.gripper.real_command):
        parser.error(
            f"{config.robot.key} has no enabled real gripper; full real packaging is unsupported for this profile"
        )
    if args.mode == "sim" or args.executor == "servo_j" or args.visual:
        require_genesis_capabilities(
            pbr=True,
            deferred_viewer=args.mode == "sim" or bool(args.visual),
        )
    if args.mode == "sim":
        return _run_sim(args)
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

    genesis = None
    bridge = None
    try:
        urdf, urdf_hash, calibration_hash = _model_and_hashes(config, args.calibration, serial)
        kinematics, collision = _backends(config, urdf)
        ik_ctx = None
        q_home: np.ndarray | None = None
        if args.executor == "servo_j":
            print(
                "[ik-compile] building Genesis scene and compiling servo_j trajectory...",
                flush=True,
            )
            ik_started = time.perf_counter()
            genesis = GenesisRuntimeManager(config.simulation)
            genesis.__enter__()
            ik_ctx = _build_packaging_context(config, urdf, show_viewer=False)
            q_home = np.asarray(ik_ctx.home_qpos, dtype=np.float64)[ik_ctx.arm_dof_idx]
            program = compile_cartesian_program_to_joint_stream(
                build_packaging_program(config, q_home=q_home),
                ik_ctx,
            )
            _print_ik_compile_complete(program, elapsed_s=time.perf_counter() - ik_started)
        else:
            program = build_packaging_program(config, kinematics=kinematics)

        scene_hash = packaging_scene_sha256(config)
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

        assert arm is not None and serial is not None
        layout = packaging_layout(config)
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
                start_q = _ensure_program_start(transport, q_home, allow_preposition=bool(args.confirm_real))
            else:
                _ensure_cartesian_program_start(
                    transport,
                    np.asarray(layout.home_position_m, dtype=np.float64),
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
            evidence_path = _sdk_evidence_path(args.report, config.robot.key)
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

        approved = gate.approve(
            program,
            preflight,
            expected_serial_number=serial,
            sdk_evidence=evidence,
        )
        if args.visual:
            from ufactory.trajectory.mirror_process import PackagingMirrorProcess

            # The servo_j compiler no longer needs its Genesis scene. Destroy
            # it before the isolated viewer starts so GPU memory is not held by
            # two Genesis runtimes at once.
            if genesis is not None:
                genesis.__exit__(None, None, None)
                genesis = None
                ik_ctx = None
            bridge = PackagingMirrorProcess(
                approved.program,
                robot_key=config.robot.key,
                config_path=args.config,
                urdf_path=urdf,
                start_hold_s=0.5,
            )
            bridge.start()

        transport.authorize_motion(mode=1)
        first_joint = None
        for segment in approved.program.segments:
            if segment.kind == "movej":
                samples, _ = segment.samples(approved.program.rate)
                if len(samples):
                    first_joint = np.asarray(samples[0], dtype=np.float64)
                    break
        if first_joint is not None:
            transport.prime_joint_stream(first_joint)
        elif args.executor == "servo_j":
            raise RuntimeError("approved servo_j packaging program has no MoveJ sample")

        gripper_adapter = create_gripper_adapter(config.gripper)
        if gripper_adapter.capabilities.real_command:
            code = gripper_adapter.prepare_real(arm)
            if code != 0:
                transport.stop()
                raise RuntimeError(f"gripper preparation failed with SDK code {code}; controller stopped")
        current_drive = [config.gripper.open_drive]

        def send_gripper(gap_m: float) -> int:
            current_drive[0] = gripper_adapter.gap_to_drive(gap_m)
            return gripper_adapter.send_real_gap(arm, gap_m)

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
            gripper_sender=send_gripper if gripper_adapter.capabilities.real_command else None,
            on_tick=bridge.on_tick if bridge is not None else None,
            not_ready_codes=frozenset((9,)),
        )
        print(f"real_execution={'PASS' if execution.completed else 'FAIL'} sent_samples={execution.sent_samples}")
        if execution.fault is not None:
            print(
                f"fault={execution.fault.fault_class}: {execution.fault.message} "
                f"sample_index={execution.fault.sample_index}"
            )
        print(
            f"timing_minor_misses={execution.minor_lateness_count} "
            f"timing_max_lateness_ms={execution.max_lateness_ns / 1_000_000.0:.3f}"
        )
        if bridge is not None:
            bridge.hold_until_closed()
        return 0 if execution.completed else 1
    finally:
        if bridge is not None:
            bridge.close()
        if genesis is not None:
            genesis.__exit__(None, None, None)
        if arm is not None and getattr(arm, "connected", False):
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
