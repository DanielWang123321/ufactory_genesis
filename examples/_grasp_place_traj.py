"""Shared trajectory-planned grasp-place pipeline, parameterized by robot.

Implements one grasp-place sequence with time-parameterized trajectories
(LSPB / trapezoidal) replayed identically in Genesis (sim) and on the real arm
(MODE_SERVO), for sim-to-real alignment by construction. Each per-robot
``examples/<robot>/<robot>_grasp_place_traj.py`` entry point is a thin CLI
wrapper that calls :func:`main` with its ``robot_key`` -- the scene geometry,
gripper family (Gripper G2 vs. Lite6 parallel gripper), object size, and
workspace coordinates are all resolved from
:func:`ufactory.trajectory.scene.build_scene`'s per-robot defaults.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ufactory.kinematics.calibration import (
    resolve_kinematics_suffix_from_ip,
    validate_kinematics_calibration_request,
)
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.trajectory import (
    CartesianWaypoint,
    EXECUTOR_SERVO_J,
    KinematicCarryTracker,
    RealExecutorConfig,
    ServoLimits,
    TrajKinematicMirror,
    TrajectoryPlannerConfig,
    compile_cartesian_program_to_joint_stream,
    plan_mixed_waypoints,
    replay_real,
    replay_sim,
)
from ufactory.trajectory.mirror_executor import resolve_segment_start_arm_q
from ufactory.trajectory.scene import (
    build_scene,
    default_grasp_gap_m,
    drive_for_gap_m,
    dry_heights,
)
from ufactory.trajectory.sim_executor import (
    DEFAULT_GRIPPER_HOLD_BIAS_GAP_M,
    LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M,
)
from _robot_viewer import start_deferred_viewer

GRIPPER_DURATION_S = 2.0
VISUAL_START_HOLD_S = 0.5
LITE6_PLACE_SETTLE_S = 0.18
SERVO_J_IK_JOINT_ACC_LIMIT_RAD_S2 = 10.0


def _lite6_place_settle_s(ctx) -> float:
    gripper = getattr(ctx, "gripper", None)
    if gripper is not None and gripper.family == "lite6":
        return LITE6_PLACE_SETTLE_S
    return 0.0


def _build_waypoints(ctx, *, grip_open_m: float, grip_close_m: float) -> tuple[list[object], list[float]]:
    """Build the default mixed waypoint list (base-frame link6 poses)."""
    obj_x, obj_y = ctx.obj_xy
    place_x, place_y = ctx.place_xy
    home = list(ctx.home_pos_base)
    pre_grasp = [obj_x, obj_y, ctx.pre_grasp_link6_z]
    grasp = [obj_x, obj_y, ctx.grasp_link6_z]
    lift = [obj_x, obj_y, ctx.lift_link6_z]
    place_top = [place_x, place_y, ctx.lift_link6_z]
    place_grasp = [place_x, place_y, ctx.grasp_link6_z]
    retreat = [place_x, place_y, ctx.lift_link6_z]

    place_tail: list[object] = [
        CartesianWaypoint(place_grasp, label="place-descend"),
    ]
    place_settle_s = _lite6_place_settle_s(ctx)
    if place_settle_s > 0.0:
        place_tail.append(
            {
                "type": "gripper",
                "gap_start": grip_close_m,
                "gap_end": grip_close_m,
                "duration": place_settle_s,
                "label": "place-settle",
            }
        )
    place_tail.extend(
        [
            {"type": "gripper", "gap_start": grip_close_m, "gap_end": grip_open_m, "duration": GRIPPER_DURATION_S, "label": "release"},
            CartesianWaypoint(retreat, label="retreat"),
            CartesianWaypoint(home, label="return-home"),
        ]
    )

    return [
        CartesianWaypoint(pre_grasp, label="home->pregrasp"),
        CartesianWaypoint(grasp, label="descend"),
        {"type": "gripper", "gap_start": grip_open_m, "gap_end": grip_close_m, "duration": GRIPPER_DURATION_S, "label": "grip"},
        CartesianWaypoint(lift, label="lift"),
        CartesianWaypoint(place_top, label="transit"),
        *place_tail,
    ], home


def _build_program(robot_key, ctx, args, *, grip_open_m: float, grip_close_m: float):
    waypoints, start_xyz = _build_waypoints(ctx, grip_open_m=grip_open_m, grip_close_m=grip_close_m)
    z_min_m = None
    if getattr(args, "z_min_mm", None) is not None:
        z_min_m = float(args.z_min_mm) / 1000.0
    config = TrajectoryPlannerConfig(
        robot_key=robot_key,
        rate=args.rate,
        speed_rad_s=args.speed_rad_s,
        mvacc_rad_s2=args.mvacc_rad_s2,
        z_min_m=z_min_m,
    )
    return plan_mixed_waypoints(config, waypoints, start_xyz=start_xyz)


def _print_phase(status, seg) -> None:
    op = status.obj_pos_mm
    lp = status.link6_mm
    print(
        f"  [{status.label:18s}] {seg.kind:7s} dur={status.duration:.3f}s "
        f"N={status.ticks:4d} ESE={status.eside_arm_mm:6.1f}mm "
        f"obj=[{op[0]:5.0f},{op[1]:5.0f},{op[2]:5.0f}] "
        f"L6=[{lp[0]:5.0f},{lp[1]:5.0f},{lp[2]:5.0f}]"
    )


def _prime_scene_to_program_start(ctx, program) -> None:
    """Put the Genesis scene at the first arm segment start before opening viewer."""
    first_arm = next((seg for seg in program.segments if seg.kind in ("movej", "movel")), None)
    if first_arm is None:
        return

    q_arm = resolve_segment_start_arm_q(ctx, first_arm)
    ctx.robot.set_dofs_position(q_arm, ctx.arm_dof_idx, zero_velocity=True)
    ctx.robot.control_dofs_position(q_arm, ctx.arm_dof_idx)

    grip_drive = drive_for_gap_m(ctx.gripper.open_gap_m, ctx.gripper)
    all_grip = np.full(len(ctx.all_gripper_dof_idx), float(grip_drive), dtype=np.float64)
    ctx.robot.set_dofs_position(all_grip, ctx.all_gripper_dof_idx, zero_velocity=True)
    ctx.robot.control_dofs_position(np.asarray([float(grip_drive)], dtype=np.float64), ctx.gripper_dof_idx)


def _open_viewer_at_program_start(ctx, program) -> None:
    _prime_scene_to_program_start(ctx, program)
    start_deferred_viewer(ctx.scene)


def _hold_scene_at_program_start(ctx, program, *, hold_s: float) -> None:
    if hold_s <= 0.0:
        return
    tick_s = 1.0 / float(program.rate)
    steps = max(1, int(np.ceil(float(hold_s) / tick_s)))
    next_deadline = time.monotonic()
    for _ in range(steps):
        _prime_scene_to_program_start(ctx, program)
        ctx.scene.step()
        next_deadline += tick_s
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def _hold_mirror_at_program_start(mirror: TrajKinematicMirror, *, hold_s: float) -> None:
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


def _run_sim(robot_key, args) -> int:
    ctx = build_scene(
        robot_key=robot_key,
        rate=args.rate,
        show_viewer=False,
        substeps=args.substeps,
        visual_model=args.visual_model,
    )
    grip_open_m = ctx.gripper.open_gap_m
    grip_close_m = args.grip_gap_mm / 1000.0
    program = _build_program(robot_key, ctx, args, grip_open_m=grip_open_m, grip_close_m=grip_close_m)
    base_mm = [v * 1000.0 for v in ctx.base_pos_world]
    print(
        f"\n[sim] robot={ctx.robot_key} gripper={ctx.gripper.family} rate={args.rate}Hz "
        f"dt={1.0/args.rate:.4f}s substeps={args.substeps}  "
        f"visual_model={ctx.visual_model} base=[{base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f}]mm  "
        f"finger_z_offset={ctx.finger_z_offset:.4f}m grasp_z={ctx.grasp_link6_z:.4f}m "
        f"grip_gap={args.grip_gap_mm:.1f}mm sim_hold_bias={args.sim_grip_hold_bias_gap_mm:.1f}mm "
        f"sim_grasp_weld={args.sim_grasp_weld}"
    )
    if args.visual:
        _open_viewer_at_program_start(ctx, program)
        _hold_scene_at_program_start(ctx, program, hold_s=args.visual_start_hold_s)
    print("Phases:")
    report = replay_sim(
        program,
        ctx,
        gripper_hold_bias_gap_m=args.sim_grip_hold_bias_gap_mm / 1000.0,
        stabilize_grasp_weld=args.sim_grasp_weld,
        on_phase=_print_phase,
    )
    print("\n" + "=" * 60)
    print(f"  Place error:    {report.place_error_mm:7.1f} mm  "
          f"{'OK' if report.place_error_mm < 50 else 'MISS'} (<50mm)")
    print(f"  Home drift:     {report.home_drift_mm:7.1f} mm  "
          f"{'OK' if report.home_drift_mm < 10 else 'DRIFT'} (<10mm)")
    print(f"  Total ticks:    {report.total_ticks}  duration={report.total_duration:.2f}s")
    print("=" * 60)
    if args.visual:
        _hold_viewer(ctx)
    ok = report.place_error_mm < 50 and report.home_drift_mm < 10
    return 0 if ok else 1


def _hold_viewer(ctx, on_step=None) -> None:
    """Keep the viewer open. ``on_step`` re-asserts a kinematic hold when set.

    The plain sim path (``on_step=None``) relies on real PD: the last
    ``control_dofs_position`` target stays in force across bare
    ``scene.step()`` calls, so the arm holds naturally. The mirror path has PD
    disabled, so it must pass a callback that re-teleports the last pose (and
    any carried object) every iteration or gravity takes over.
    """
    print("Viewer open. Press Ctrl+C to exit...")
    try:
        for _ in range(2000):
            if on_step is not None:
                on_step()
            else:
                ctx.scene.step()
    except KeyboardInterrupt:
        pass


def _resolve_real_kinematics(runtime, args, ip: str | None) -> str | None:
    """Resolve the calibration suffix used by host-side IK and mirror scenes."""
    kinematics_suffix = args.kinematics_suffix
    if ip:
        cli_suffix = kinematics_suffix
        kinematics_suffix, sn = resolve_kinematics_suffix_from_ip(
            ip,
            runtime.model.robot_name,
            kinematics_suffix=cli_suffix,
            kinematics_yaml=args.kinematics_yaml,
        )
        if kinematics_suffix and not cli_suffix and args.kinematics_yaml is None:
            print(f"kinematics_suffix: {kinematics_suffix} (auto from SN {sn})")
        validate_kinematics_calibration_request(
            sn,
            runtime.model.robot_name,
            kinematics_yaml=args.kinematics_yaml,
            kinematics_suffix=kinematics_suffix,
            allow_sn_override=args.force_kinematics,
        )
        return kinematics_suffix

    if args.executor == EXECUTOR_SERVO_J and args.kinematics_yaml is None and not kinematics_suffix:
        print(
            "[WARN] servo_j dry-run is using the nominal URDF for host-side IK. "
            "For real motion, pass --ip so the SN-derived kinematics suffix is used, "
            "or pass --kinematics-yaml/--kinematics-suffix explicitly."
        )
    return kinematics_suffix


def _run_real(robot_key, args) -> int:
    runtime = get_robot_runtime_profile(robot_key)
    mirror = None
    tracker = None
    ctx = None
    ip = args.ip or os.environ.get("XARM_IP")
    kinematics_suffix = _resolve_real_kinematics(runtime, args, ip)
    needs_ik_scene = args.executor == EXECUTOR_SERVO_J
    if args.visual or needs_ik_scene:
        ctx = build_scene(
            robot_key=robot_key,
            rate=args.rate,
            show_viewer=False,
            substeps=args.substeps,
            visual_model=args.visual_model,
            kinematics_yaml=args.kinematics_yaml,
            kinematics_suffix=kinematics_suffix,
            kinematics_yaml_dir=args.kinematics_yaml_dir,
        )
    heights = ctx if ctx is not None else dry_heights(robot_key)
    grip_open_m = ctx.gripper.open_gap_m if ctx is not None else runtime.gripper.open_gap_m
    grip_close_m = args.grip_gap_mm / 1000.0
    program = _build_program(robot_key, heights, args, grip_open_m=grip_open_m, grip_close_m=grip_close_m)
    if args.executor == EXECUTOR_SERVO_J:
        program = compile_cartesian_program_to_joint_stream(program, ctx)
        print(
            f"[ik-compile] executor=servo_j movel_segments={program.metadata['ik_compiled_movel_segments']} "
            f"ticks={program.metadata['ik_compiled_ticks']} "
            f"plateau_collapsed={program.metadata.get('ik_plateau_collapsed_samples', 0)} "
            f"retimed={program.metadata.get('ik_joint_retimed', False)} "
            f"urdf={program.metadata.get('ik_robot_urdf')} "
            f"calib={program.metadata.get('ik_kinematics_yaml') or '(nominal)'}"
        )
    if args.visual and ctx is not None:
        base_mm = [v * 1000.0 for v in ctx.base_pos_world]
        print(
            f"\n[mirror] robot={ctx.robot_key} gripper={ctx.gripper.family} rate={args.rate}Hz "
            f"visual_model={ctx.visual_model} "
            f"base=[{base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f}]mm "
            f"kinematic open-loop; cube kinematically carried while gripped "
            f"(no contact physics)"
        )
        mirror = TrajKinematicMirror(ctx, program)
        mirror.prime_to_home()
        start_deferred_viewer(ctx.scene)
        _hold_mirror_at_program_start(mirror, hold_s=args.visual_start_hold_s)
        tracker = KinematicCarryTracker(mirror, grasp_gap_m=grip_close_m)
    cfg = RealExecutorConfig(
        executor=args.executor,
        robot_key=runtime.model.key,
        rate=args.rate,
        speed_rad_s=args.speed_rad_s,
        mvacc_rad_s2=args.mvacc_rad_s2,
        speed_mm_s=args.speed_mm_s,
        mvacc_mm_s2=args.mvacc_mm_s2,
        z_min_mm=args.z_min_mm,
        dry_run=args.dry_run and not args.sdk_sim_validate,
        ip=ip,
        kinematics_suffix=kinematics_suffix,
        sdk_sim_validate=args.sdk_sim_validate,
        sdk_sim_report_csv=args.sdk_sim_report_csv,
        real_gripper=args.real_gripper,
        servo_limits=(
            ServoLimits(joint_acc_rad_s2=SERVO_J_IK_JOINT_ACC_LIMIT_RAD_S2)
            if args.executor == EXECUTOR_SERVO_J
            else ServoLimits()
        ),
    )
    replay_real(
        program,
        cfg,
        on_tick=tracker.on_tick if tracker is not None else None,
        on_preposition_complete=(
            (lambda: mirror.prime_to_first_arm_segment_start(program)) if mirror is not None else None
        ),
    )
    if args.visual and ctx is not None:
        _hold_viewer(ctx, on_step=tracker.hold_step if tracker is not None else None)
    return 0


def build_arg_parser(robot_key: str, *, robot_label: str) -> argparse.ArgumentParser:
    """Build the shared CLI, seeded with ``robot_key``'s tuned defaults."""
    default_grip_close_m = default_grasp_gap_m(robot_key)
    runtime = get_robot_runtime_profile(robot_key)
    grip_open_mm = runtime.gripper.open_gap_m * 1000.0
    grip_closed_mm = runtime.gripper.closed_gap_m * 1000.0
    default_hold_bias_m = (
        LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M
        if runtime.gripper.family == "lite6"
        else DEFAULT_GRIPPER_HOLD_BIAS_GAP_M
    )

    parser = argparse.ArgumentParser(
        description=f"{robot_label} trajectory-planned grasp-place (RoboDK-style)"
    )
    parser.add_argument("--rate", type=float, default=50.0, choices=[50.0, 100.0])
    # Default is calibrated for sim grasp-place reliability (gentle accel keeps
    # the weakly-actuated grip on the object). Real deploy may raise these.
    parser.add_argument("--speed-rad-s", type=float, default=0.35)
    parser.add_argument("--mvacc-rad-s2", type=float, default=2.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument(
        "--visual-model",
        choices=("glb", "stl"),
        default="glb",
        help="Sim robot visual model: GLB visuals with STL collision by default; 'stl' is xArm6-only legacy visuals.",
    )
    parser.add_argument(
        "--grip-gap-mm",
        type=float,
        default=default_grip_close_m * 1000.0,
        help="Target two-finger gap for the grasp segment; default is tuned for this robot's default object.",
    )
    parser.add_argument(
        "--sim-grip-hold-bias-gap-mm",
        type=float,
        default=default_hold_bias_m * 1000.0,
        help="Sim-only post-contact gripper hold bias, in physical two-finger gap mm.",
    )
    parser.add_argument(
        "--sim-grasp-weld",
        dest="sim_grasp_weld",
        action="store_true",
        default=False,
        help="Sim-only debug: add a contact-gated weld after real bilateral finger/object contact.",
    )
    parser.add_argument(
        "--no-sim-grasp-weld",
        dest="sim_grasp_weld",
        action="store_false",
        help="Compatibility no-op; contact/friction-only grasp is already the default.",
    )
    # Sim-only
    g_sim = parser.add_mutually_exclusive_group()
    g_sim.add_argument("--headless", action="store_true", help="Run sim without viewer (default)")
    g_sim.add_argument("--visual", action="store_true",
                        help="Sim: Genesis viewer; real path: kinematic mirror viewer")
    parser.add_argument(
        "--visual-start-hold-s",
        type=float,
        default=VISUAL_START_HOLD_S,
        help="--visual: hold the initial program pose after the viewer opens before replay starts.",
    )
    # Real path
    parser.add_argument("--executor", default=None, choices=("servo_j", "servo_cartesian"),
                        help="Enable the real path: servo_cartesian uses firmware IK; servo_j uses host-side Genesis IK")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--z-min-mm", type=float, default=0.0)
    parser.add_argument("--kinematics-suffix", default=os.environ.get("XARM_KINEMATICS_SUFFIX"))
    parser.add_argument("--kinematics-yaml", default=None)
    parser.add_argument("--kinematics-yaml-dir", default=None)
    parser.add_argument(
        "--force-kinematics",
        action="store_true",
        help="Use requested kinematics YAML/suffix even when the SN rule says nominal URDF should be used.",
    )
    parser.add_argument("--speed-mm-s", type=float, default=200.0)
    parser.add_argument("--mvacc-mm-s2", type=float, default=1000.0)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                        help="Real path: print digest only (default)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Real path: actually stream to the arm (requires --ip)")
    parser.add_argument(
        "--real-gripper",
        action="store_true",
        help="Real path: also send physical gripper commands. Omit for arm-only safety validation.",
    )
    parser.add_argument(
        "--sdk-sim-validate",
        action="store_true",
        help=(
            "Real path: connect to SDK, enable set_simulation_robot(True), "
            "preposition, and stream servo targets for validation without moving the physical arm."
        ),
    )
    parser.add_argument(
        "--sdk-sim-report-csv",
        default=None,
        help="Optional CSV path for --sdk-sim-validate per-tick target/feedback velocity report.",
    )
    parser.set_defaults(_grip_open_mm=grip_open_mm, _grip_closed_mm=grip_closed_mm)
    return parser


def main(robot_key: str, *, robot_label: str) -> int:
    parser = build_arg_parser(robot_key, robot_label=robot_label)
    args = parser.parse_args()
    grip_open_mm = args._grip_open_mm
    grip_closed_mm = args._grip_closed_mm
    if not grip_closed_mm - 1e-3 <= args.grip_gap_mm <= grip_open_mm + 1e-3:
        parser.error(
            f"--grip-gap-mm must be between {grip_closed_mm:.1f} and {grip_open_mm:.1f}"
        )
    runtime = get_robot_runtime_profile(robot_key)
    if args.visual_start_hold_s < 0.0:
        parser.error("--visual-start-hold-s must be non-negative")
    if args.sdk_sim_report_csv and not args.sdk_sim_validate:
        parser.error("--sdk-sim-report-csv requires --sdk-sim-validate")
    if args.sdk_sim_validate:
        if args.executor is None:
            parser.error("--sdk-sim-validate requires --executor")
        if not (args.ip or os.environ.get("XARM_IP")):
            parser.error("--sdk-sim-validate requires --ip or XARM_IP")
    if args.executor is not None and not args.dry_run and not (args.ip or os.environ.get("XARM_IP")):
        parser.error("--no-dry-run requires --ip or XARM_IP")

    if args.executor is None:
        return _run_sim(robot_key, args)
    return _run_real(robot_key, args)
