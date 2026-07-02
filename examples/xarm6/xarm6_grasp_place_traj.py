"""
xArm6 Grasp-Place — trajectory-planned pipeline.

A structured pick-and-place sequence with **time-parameterized trajectories**
(LSPB / trapezoidal) replayed identically in Genesis (sim) and on the real xArm
(MODE_SERVO), giving sim-to-real alignment by construction: the same absolute
target stream is sampled at the same rate on both sides.

Default (no ``--executor``) runs the **sim** grasp-place sequence and reports
``place_error_mm`` / ``home_drift_mm`` and per-segment duration/profile/ESE.

Real path (``--executor servo-j|servo-cartesian``) default ``--dry-run`` prints a
per-segment/per-tick digest and never moves the arm. To move the real arm:

    python examples/xarm6/xarm6_grasp_place_traj.py \
        --executor servo-j --ip 192.168.1.65 --z-min-mm 0 --no-dry-run

Real path with Genesis kinematic mirror (same planned trajectory, lightweight viewer):

    python examples/xarm6/xarm6_grasp_place_traj.py \
        --executor servo-j --visual --ip 192.168.1.65 --z-min-mm 0 --no-dry-run

Prerequisite for real motion: pass the existing FK/IK alignment gate
(``xarm6_reach_deploy.py --mode align ...``) first.

SDK simulation validation still connects to the controller, but first switches
``set_simulation_robot(True)`` and streams in simulation mode:

    python examples/xarm6/xarm6_grasp_place_traj.py \
        --executor servo-cartesian --sdk-sim-validate --ip 192.168.1.65 \
        --rate 50 --z-min-mm 0 \
        --sdk-sim-report-csv reports/servo_sim.csv

Usage (sim):
    conda activate py313
    python examples/xarm6/xarm6_grasp_place_traj.py --headless --rate 50
    python examples/xarm6/xarm6_grasp_place_traj.py --visual --rate 50
    python examples/xarm6/xarm6_grasp_place_traj.py --visual --visual-model stl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from ufactory.gripper_g2 import GRIPPER_G2_OPEN_GAP_M
from ufactory.kinematics import resolve_kinematics_suffix_from_ip
from ufactory.robot_params import get_robot_runtime_profile
from ufactory.trajectory import (
    KinematicCarryTracker,
    RealExecutorConfig,
    TrajKinematicMirror,
    build_pickplace_program,
    replay_real,
    replay_sim,
)
from ufactory.trajectory.scene import DEFAULT_GRIPPER_GRASP_GAP_M, build_scene

GRIP_OPEN_M = GRIPPER_G2_OPEN_GAP_M  # 0.084 m (84 mm)
GRIP_CLOSE_M = DEFAULT_GRIPPER_GRASP_GAP_M  # 24 mm, tuned for the 40 mm cube in Genesis.
GRIPPER_DURATION_S = 2.0


def _build_waypoints(ctx, *, grip_close_m: float = GRIP_CLOSE_M) -> list[dict]:
    """Build the default grasp-place waypoint list (base-frame link6 poses)."""
    obj_x, obj_y = ctx.obj_xy
    place_x, place_y = ctx.place_xy
    home = list(ctx.home_pos_base)
    pre_grasp = [obj_x, obj_y, ctx.pre_grasp_link6_z]
    grasp = [obj_x, obj_y, ctx.grasp_link6_z]
    lift = [obj_x, obj_y, ctx.lift_link6_z]
    place_top = [place_x, place_y, ctx.lift_link6_z]
    place_grasp = [place_x, place_y, ctx.grasp_link6_z]
    retreat = [place_x, place_y, ctx.lift_link6_z]

    return [
        {"type": "movel", "pose_start": home, "pose_end": pre_grasp, "label": "home->pregrasp"},
        {"type": "movel", "pose_start": pre_grasp, "pose_end": grasp, "label": "descend"},
        {"type": "gripper", "gap_start": GRIP_OPEN_M, "gap_end": grip_close_m, "duration": GRIPPER_DURATION_S, "label": "grip"},
        {"type": "movel", "pose_start": grasp, "pose_end": lift, "label": "lift"},
        {"type": "movel", "pose_start": lift, "pose_end": place_top, "label": "transit"},
        {"type": "movel", "pose_start": place_top, "pose_end": place_grasp, "label": "place-descend"},
        {"type": "gripper", "gap_start": grip_close_m, "gap_end": GRIP_OPEN_M, "duration": GRIPPER_DURATION_S, "label": "release"},
        {"type": "movel", "pose_start": place_grasp, "pose_end": retreat, "label": "retreat"},
        {"type": "movel", "pose_start": retreat, "pose_end": home, "label": "return-home"},
    ]


def _print_phase(status, seg) -> None:
    op = status.obj_pos_mm
    lp = status.link6_mm
    print(
        f"  [{status.label:18s}] {seg.kind:7s} dur={status.duration:.3f}s "
        f"N={status.ticks:4d} ESE={status.eside_arm_mm:6.1f}mm "
        f"obj=[{op[0]:5.0f},{op[1]:5.0f},{op[2]:5.0f}] "
        f"L6=[{lp[0]:5.0f},{lp[1]:5.0f},{lp[2]:5.0f}]"
    )


def run_sim(args) -> int:
    ctx = build_scene(
        rate=args.rate,
        show_viewer=args.visual,
        substeps=args.substeps,
        visual_model=args.visual_model,
    )
    grip_close_m = args.grip_gap_mm / 1000.0
    waypoints = _build_waypoints(ctx, grip_close_m=grip_close_m)
    program = build_pickplace_program(
        rate=args.rate,
        speed_rad_s=args.speed_rad_s,
        mvacc_rad_s2=args.mvacc_rad_s2,
        waypoints=waypoints,
    )
    base_mm = [v * 1000.0 for v in ctx.base_pos_world]
    print(
        f"\n[sim] rate={args.rate}Hz dt={1.0/args.rate:.4f}s substeps={args.substeps}  "
        f"visual_model={ctx.visual_model} base=[{base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f}]mm  "
        f"finger_z_offset={ctx.finger_z_offset:.4f}m grasp_z={ctx.grasp_link6_z:.4f}m "
        f"grip_gap={args.grip_gap_mm:.1f}mm"
    )
    print("Phases:")
    report = replay_sim(program, ctx, on_phase=_print_phase)
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


def run_real(args) -> int:
    mirror = None
    tracker = None
    ctx = None
    if args.visual:
        ctx = build_scene(
            rate=args.rate,
            show_viewer=True,
            substeps=args.substeps,
            visual_model=args.visual_model,
        )
    heights = ctx if ctx is not None else _DryHeights()
    grip_close_m = args.grip_gap_mm / 1000.0
    waypoints = _build_waypoints(heights, grip_close_m=grip_close_m)
    program = build_pickplace_program(
        rate=args.rate,
        speed_rad_s=args.speed_rad_s,
        mvacc_rad_s2=args.mvacc_rad_s2,
        waypoints=waypoints,
    )
    if ctx is not None:
        base_mm = [v * 1000.0 for v in ctx.base_pos_world]
        print(
            f"\n[mirror] rate={args.rate}Hz visual_model={ctx.visual_model} "
            f"base=[{base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f}]mm "
            f"kinematic open-loop; cube kinematically carried while gripped "
            f"(no contact physics)"
        )
        mirror = TrajKinematicMirror(ctx, program)
        mirror.prime_to_home()
        tracker = KinematicCarryTracker(mirror, grasp_gap_m=grip_close_m)
    ip = args.ip or os.environ.get("XARM_IP")
    kinematics_suffix = args.kinematics_suffix
    if ip:
        runtime = get_robot_runtime_profile("xarm6")
        cli_suffix = kinematics_suffix
        kinematics_suffix, sn = resolve_kinematics_suffix_from_ip(
            ip,
            runtime.model.robot_name,
            kinematics_suffix=cli_suffix,
        )
        if kinematics_suffix and not cli_suffix:
            print(f"kinematics_suffix: {kinematics_suffix} (auto from SN {sn})")
    cfg = RealExecutorConfig(
        executor=args.executor,
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
    )
    replay_real(
        program,
        cfg,
        on_tick=tracker.on_tick if tracker is not None else None,
        on_preposition_complete=(
            (lambda: mirror.prime_to_first_arm_segment_start(program)) if mirror is not None else None
        ),
    )
    if ctx is not None:
        _hold_viewer(ctx, on_step=tracker.hold_step if tracker is not None else None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="xArm6 trajectory-planned grasp-place (RoboDK-style)")
    parser.add_argument("--rate", type=float, default=50.0, choices=[50.0, 100.0])
    # Default is calibrated for sim grasp-place reliability (gentle accel keeps
    # the weakly-actuated G2 grip on the cube). Real deploy may raise these.
    parser.add_argument("--speed-rad-s", type=float, default=0.35)
    parser.add_argument("--mvacc-rad-s2", type=float, default=2.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument(
        "--visual-model",
        choices=("glb", "stl"),
        default="glb",
        help="Sim robot visual model: GLB visuals with STL collision by default; use stl for legacy visuals.",
    )
    parser.add_argument(
        "--grip-gap-mm",
        type=float,
        default=GRIP_CLOSE_M * 1000.0,
        help="Target two-finger gap for the grasp segment; default is tuned for the 40 mm cube.",
    )
    # Sim-only
    g_sim = parser.add_mutually_exclusive_group()
    g_sim.add_argument("--headless", action="store_true", help="Run sim without viewer (default)")
    g_sim.add_argument("--visual", action="store_true",
                        help="Sim: Genesis viewer; real path: kinematic mirror viewer")
    # Real path
    parser.add_argument("--executor", default=None, choices=("servo-j", "servo-cartesian"),
                        help="Enable the real path; default (omitted) runs sim")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--z-min-mm", type=float, default=0.0)
    parser.add_argument("--kinematics-suffix", default=os.environ.get("XARM_KINEMATICS_SUFFIX"))
    parser.add_argument("--speed-mm-s", type=float, default=200.0)
    parser.add_argument("--mvacc-mm-s2", type=float, default=1000.0)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                        help="Real path: print digest only (default)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Real path: actually stream to the arm (requires --ip)")
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
    args = parser.parse_args()
    if not 0.0 <= args.grip_gap_mm <= GRIP_OPEN_M * 1000.0:
        parser.error(f"--grip-gap-mm must be between 0 and {GRIP_OPEN_M * 1000.0:.1f}")
    if args.sdk_sim_report_csv and not args.sdk_sim_validate:
        parser.error("--sdk-sim-report-csv requires --sdk-sim-validate")
    if args.sdk_sim_validate:
        if args.executor is None:
            parser.error("--sdk-sim-validate requires --executor")
        if not (args.ip or os.environ.get("XARM_IP")):
            parser.error("--sdk-sim-validate requires --ip or XARM_IP")

    if args.executor is None:
        return run_sim(args)
    return run_real(args)


class _DryHeights:
    """Off-sim height estimate mirrors scene.py constants for the real path.

    The real arm uses per-unit calibrated forward kinematics (SN suffix auto) at deploy time;
    these base-frame heights are the same physically-derived values the sim
    measures, so the dry-run / real stream matches the sim plan up to the
    per-arm kinematics offset. For exact sim parity run the sim first.
    """

    def __init__(self) -> None:
        from ufactory.trajectory import scene as S
        from ufactory.trajectory.scene import FINGER_PAD_BELOW_FC, FINGER_CLOSE_DESCENT, GRASP_TABLE_CLEARANCE

        # URDF-derived link6->finger-center offset with the gripper open at
        # home; kept in sync with the value scene.build_scene measures for the
        # current trajectory URDF so dry-run ticks match sim ticks.
        self.finger_z_offset = 0.1011
        self.home_pos_base = [0.3, 0.0, S.HOME_Z]
        self.obj_xy = tuple(S.OBJ_XY)
        self.place_xy = tuple(S.PLACE_XY)
        self.grasp_link6_z = GRASP_TABLE_CLEARANCE + FINGER_CLOSE_DESCENT + FINGER_PAD_BELOW_FC + self.finger_z_offset
        self.pre_grasp_link6_z = self.grasp_link6_z + 0.10
        self.lift_link6_z = S.LIFT_Z


if __name__ == "__main__":
    raise SystemExit(main())
