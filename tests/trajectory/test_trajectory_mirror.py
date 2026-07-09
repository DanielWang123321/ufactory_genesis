"""Tests for the kinematic Genesis mirror executor."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ufactory.trajectory import (
    AsyncMirrorBridge,
    CartesianWaypoint,
    KinematicCarryTracker,
    RealExecutorConfig,
    TrajKinematicMirror,
    TrajectoryPlannerConfig,
    build_pickplace_program,
    compile_cartesian_program_to_joint_stream,
    plan_mixed_waypoints,
    replay_real,
)
from ufactory.trajectory.mirror_executor import (
    G2_MIRROR_VISUAL_GAP_SHRINK_M,
    LITE6_MIRROR_VISUAL_GAP_SHRINK_M,
    mirror_grip_drive_for_gap_m,
    resolve_segment_start_arm_q,
    resolve_tick_arm_q,
    resolve_tick_grip_drive,
)
from ufactory.trajectory.scene import build_scene, drive_for_gap_m
from ufactory.robots.runtime import G2_GRIPPER_PARAMS, LITE6_GRIPPER_PARAMS

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import PROJECT_ROOT


def _default_waypoints() -> list[dict]:
    obj_x, obj_y = 0.30, 0.00
    place_x, place_y = 0.30, 0.30
    home = [0.30, 0.00, 0.30]
    finger_z_offset = 0.1011
    grasp_z = 0.010 + 0.015 + 0.061 + finger_z_offset
    pre_grasp_z = grasp_z + 0.10
    lift_z = 0.30
    grip_open_m = 0.084
    grip_close_m = 0.024
    grip_duration_s = 2.0

    pre_grasp = [obj_x, obj_y, pre_grasp_z]
    grasp = [obj_x, obj_y, grasp_z]
    lift = [obj_x, obj_y, lift_z]
    place_top = [place_x, place_y, lift_z]
    place_grasp = [place_x, place_y, grasp_z]
    retreat = [place_x, place_y, lift_z]

    return [
        {"type": "movel", "pose_start": home, "pose_end": pre_grasp, "label": "home->pregrasp"},
        {"type": "movel", "pose_start": pre_grasp, "pose_end": grasp, "label": "descend"},
        {"type": "gripper", "gap_start": grip_open_m, "gap_end": grip_close_m, "duration": grip_duration_s, "label": "grip"},
        {"type": "movel", "pose_start": grasp, "pose_end": lift, "label": "lift"},
        {"type": "movel", "pose_start": lift, "pose_end": place_top, "label": "transit"},
        {"type": "movel", "pose_start": place_top, "pose_end": place_grasp, "label": "place-descend"},
        {"type": "gripper", "gap_start": grip_close_m, "gap_end": grip_open_m, "duration": grip_duration_s, "label": "release"},
        {"type": "movel", "pose_start": place_grasp, "pose_end": retreat, "label": "retreat"},
        {"type": "movel", "pose_start": retreat, "pose_end": home, "label": "return-home"},
    ]


def _default_mixed_waypoints() -> tuple[list[object], list[float]]:
    legacy = _default_waypoints()
    start_xyz = list(legacy[0]["pose_start"])
    waypoints: list[object] = []
    for wp in legacy:
        if wp["type"] == "movel":
            waypoints.append(CartesianWaypoint(wp["pose_end"], label=wp["label"]))
        else:
            waypoints.append(wp)
    return waypoints, start_xyz


def _legacy_default_program():
    return build_pickplace_program(
        rate=50.0,
        speed_rad_s=0.35,
        mvacc_rad_s2=2.0,
        waypoints=_default_waypoints(),
    )


def _default_program():
    waypoints, start_xyz = _default_mixed_waypoints()
    config = TrajectoryPlannerConfig(
        robot_key="xarm6",
        rate=50.0,
        speed_rad_s=0.35,
        mvacc_rad_s2=2.0,
        z_min_m=0.0,
    )
    return plan_mixed_waypoints(config, waypoints, start_xyz=start_xyz)


# Genesis only supports one gs.init() per process, so every test that needs a
# real scene must share a single module-scoped build_scene() call rather than
# building its own.
_PRISTINE_OBJ_STATE: dict[str, torch.Tensor] = {}


@pytest.fixture(scope="module")
def shared_ctx():
    ctx = build_scene(rate=50.0, show_viewer=False, substeps=2)
    _PRISTINE_OBJ_STATE["pos"] = ctx.obj.get_pos().clone()
    _PRISTINE_OBJ_STATE["quat"] = ctx.obj.get_quat().clone()
    return ctx


def _fresh_mirror(ctx, program) -> TrajKinematicMirror:
    """Reset the shared object pose and return a freshly primed mirror (test isolation)."""
    ctx.obj.set_pos(_PRISTINE_OBJ_STATE["pos"], zero_velocity=True)
    ctx.obj.set_quat(_PRISTINE_OBJ_STATE["quat"], zero_velocity=True)
    mirror = TrajKinematicMirror(ctx, program)
    mirror.prime_to_home()
    return mirror


def test_resolve_tick_grip_drive_open_and_close():
    samples = np.array([[0.084], [0.024]], dtype=np.float64)
    open_drive = resolve_tick_grip_drive(samples, 0, G2_GRIPPER_PARAMS)
    close_drive = resolve_tick_grip_drive(samples, 1, G2_GRIPPER_PARAMS)
    assert open_drive == pytest.approx(0.0)
    assert close_drive > open_drive


def test_mirror_grip_drive_floors_g2_and_lite6_visual_gap():
    """Kinematic mirror must not teleport pads past the cube faces."""
    obj_w = 0.030
    g2_ctx = SimpleNamespace(gripper=G2_GRIPPER_PARAMS, obj_size=(obj_w, obj_w, obj_w))
    lite_ctx = SimpleNamespace(gripper=LITE6_GRIPPER_PARAMS, obj_size=(obj_w, obj_w, obj_w))

    g2_floor = obj_w - G2_MIRROR_VISUAL_GAP_SHRINK_M
    lite_floor = obj_w - LITE6_MIRROR_VISUAL_GAP_SHRINK_M
    assert mirror_grip_drive_for_gap_m(g2_ctx, 0.022) == pytest.approx(
        drive_for_gap_m(g2_floor, G2_GRIPPER_PARAMS)
    )
    assert mirror_grip_drive_for_gap_m(g2_ctx, g2_floor + 0.001) == pytest.approx(
        drive_for_gap_m(g2_floor + 0.001, G2_GRIPPER_PARAMS)
    )
    assert mirror_grip_drive_for_gap_m(lite_ctx, 0.020) == pytest.approx(
        drive_for_gap_m(lite_floor, LITE6_GRIPPER_PARAMS)
    )


def test_default_grasp_place_program_uses_mixed_waypoint_planner():
    program = _default_program()

    assert program.robot_key == "xarm6_1305"
    assert program.metadata["planner"] == "ufactory.trajectory.planner"
    assert program.metadata["kind"] == "mixed"
    assert [seg.label for seg in program.segments] == [
        "home->pregrasp",
        "descend",
        "grip",
        "lift",
        "transit",
        "place-descend",
        "release",
        "retreat",
        "return-home",
    ]


def test_legacy_pickplace_builder_still_matches_default_segment_shape():
    planned = _default_program()
    legacy = _legacy_default_program()

    assert [seg.kind for seg in legacy.segments] == [seg.kind for seg in planned.segments]
    assert [seg.label for seg in legacy.segments] == [seg.label for seg in planned.segments]
    assert legacy.total_ticks == planned.total_ticks
    assert legacy.robot_key is None


def test_on_tick_count_matches_program_total_ticks(monkeypatch):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    program = _default_program()
    tick_count = 0

    def on_tick(_seg, _tick_idx) -> None:
        nonlocal tick_count
        tick_count += 1

    cfg = RealExecutorConfig(dry_run=True, rate=50.0)
    replay_real(program, cfg, on_tick=on_tick)
    assert tick_count == program.total_ticks


def test_async_mirror_bridge_on_tick_does_not_block_on_tracker():
    """Servo-path on_tick must return without waiting for scene.step / tracker work."""
    started = threading.Event()
    release = threading.Event()
    applied: list[tuple[object, int]] = []

    class SlowTracker:
        def on_tick(self, seg, tick_idx: int) -> None:
            started.set()
            release.wait(timeout=2.0)
            applied.append((seg, tick_idx))

    bridge = AsyncMirrorBridge(SlowTracker())
    seg_a = SimpleNamespace(label="a")
    seg_b = SimpleNamespace(label="b")
    try:
        t0 = time.perf_counter()
        bridge.on_tick(seg_a, 0)
        # Posting must not wait for the slow tracker (budget << tracker hold).
        assert time.perf_counter() - t0 < 0.05
        assert started.wait(timeout=1.0)

        # Latest-wins: while tracker is busy, overwrite pending with seg_b.
        bridge.on_tick(seg_b, 1)
        release.set()
    finally:
        bridge.close()

    assert applied
    assert applied[-1] == (seg_b, 1) or applied == [(seg_a, 0), (seg_b, 1)]
    # Intermediate frames may drop; final applied tick must be the latest posted.
    assert any(item == (seg_b, 1) for item in applied)


def test_async_mirror_bridge_background_consumes_tracker():
    """Background worker must invoke tracker.on_tick for posted samples."""
    calls: list[tuple[object, int]] = []
    done = threading.Event()

    class CountingTracker:
        def on_tick(self, seg, tick_idx: int) -> None:
            calls.append((seg, tick_idx))
            if tick_idx == 2:
                done.set()

    tracker = CountingTracker()
    # Guard: bridge must not touch a scene.step on the posting path.
    scene = MagicMock()
    bridge = AsyncMirrorBridge(tracker)
    seg = SimpleNamespace(label="transit", scene=scene)
    try:
        for i in range(3):
            bridge.on_tick(seg, i)
        assert done.wait(timeout=2.0)
    finally:
        bridge.close()

    assert calls
    assert calls[-1] == (seg, 2)
    scene.step.assert_not_called()


@pytest.mark.gpu
def test_mirror_tick_arm_targets_finite_on_default_program(shared_ctx):
    program = _default_program()
    ctx = shared_ctx
    _fresh_mirror(ctx, program)

    for seg in program.segments:
        samples, n = seg.samples(program.rate)
        for t in range(n):
            if seg.kind == "gripper":
                drive = resolve_tick_grip_drive(samples, t, ctx.gripper)
                assert np.isfinite(drive)
                continue
            q = resolve_tick_arm_q(ctx, seg, samples, t)
            assert q.shape == (6,)
            assert np.all(np.isfinite(q))

    first_arm = next(seg for seg in program.segments if seg.kind in ("movej", "movel"))
    q_start = resolve_segment_start_arm_q(ctx, first_arm)
    assert q_start.shape == (6,)
    assert np.all(np.isfinite(q_start))


@pytest.mark.gpu
def test_hold_step_does_not_drift_arm_pose(shared_ctx):
    """Regression test: hold_step must keep re-teleporting the last pose.

    Before this fix, the CLI's post-replay hold loop called bare
    ``scene.step()`` while the mirror had zeroed PD gains, so the arm sagged
    under gravity, unbounded, tick after tick. ``hold_step`` re-teleports the
    pose every call, so gravity only ever acts for a single physics step
    before being undone -- the true regression signature is *no growth* over
    many holds, not a mathematically exact position match (one step of
    unopposed gravity on a loaded joint is a real, bounded, sub-degree
    artifact of the teleport-then-step approach, not the bug being guarded).
    """
    program = _default_program()
    ctx = shared_ctx
    mirror = _fresh_mirror(ctx, program)

    first_arm_seg = next(seg for seg in program.segments if seg.kind in ("movej", "movel"))
    mirror.prime_to_segment_start(first_arm_seg)

    mirror.hold_step()
    q_after_one_hold = ctx.robot.get_dofs_position()[0, ctx.arm_dof_idx].detach().cpu().numpy().copy()

    for _ in range(200):
        mirror.hold_step()

    q_after_many_holds = ctx.robot.get_dofs_position()[0, ctx.arm_dof_idx].detach().cpu().numpy()
    np.testing.assert_allclose(
        q_after_many_holds, q_after_one_hold, atol=1e-4,
        err_msg="arm pose kept drifting across repeated hold_step() calls instead of staying steady",
    )


def _run_ticks_through_label(tracker, program, *, stop_after_label: str):
    """Drive ``tracker.on_tick`` through segments up to and including ``stop_after_label``."""
    for seg in program.segments:
        samples, n = seg.samples(program.rate)
        for t in range(n):
            tracker.on_tick(seg, t)
        if seg.label == stop_after_label:
            return seg, n
    raise AssertionError(f"segment labelled {stop_after_label!r} not found in program")


@pytest.mark.gpu
def test_grip_segment_obj_xy_drift_bounded_during_close(shared_ctx):
    """Regression: kinematic gripper close must not push the physics cube on the table."""
    program = _default_program()
    ctx = shared_ctx
    mirror = _fresh_mirror(ctx, program)
    tracker = KinematicCarryTracker(mirror, grasp_gap_m=0.024)
    spawn_xy = _PRISTINE_OBJ_STATE["pos"][0, :2].clone()

    max_xy_drift_m = 0.0
    for seg in program.segments:
        if seg.label == "lift":
            break
        samples, n = seg.samples(program.rate)
        for t in range(n):
            tracker.on_tick(seg, t)
            if seg.label == "grip":
                pos_xy = ctx.obj.get_pos()[0, :2]
                max_xy_drift_m = max(
                    max_xy_drift_m,
                    float(torch.norm(pos_xy - spawn_xy).item()),
                )

    assert tracker.attached, "segment-end latch should engage after the grip close segment"
    assert max_xy_drift_m < 0.001, (
        f"grip segment pushed the object {max_xy_drift_m * 1000:.2f} mm in XY "
        "(expected < 1 mm with freeze-until-latch)"
    )


@pytest.mark.gpu
def test_kinematic_carry_tracker_attaches_on_grip_and_releases_on_reopen(shared_ctx):
    program = _default_program()
    ctx = shared_ctx
    mirror = _fresh_mirror(ctx, program)
    tracker = KinematicCarryTracker(mirror, grasp_gap_m=0.024)

    assert not tracker.attached

    _run_ticks_through_label(tracker, program, stop_after_label="grip")
    assert tracker.attached, "segment-end latch should engage after the grip close segment"

    z_at_grasp = float(ctx.obj.get_pos()[0, 2].item())
    _run_ticks_through_label(tracker, program, stop_after_label="lift")
    z_after_lift = float(ctx.obj.get_pos()[0, 2].item())
    assert tracker.attached, "should remain attached through the lift segment"
    assert z_after_lift > z_at_grasp + 0.05, "carried object should rise with the gripper during lift"

    _run_ticks_through_label(tracker, program, stop_after_label="transit")
    _run_ticks_through_label(tracker, program, stop_after_label="place-descend")
    assert tracker.attached, "should still be attached right before the release segment"

    _run_ticks_through_label(tracker, program, stop_after_label="release")
    assert not tracker.attached, "reopening past release_gap_m should unlatch the carry"


@pytest.mark.gpu
def test_kinematic_carry_tracker_accepts_compiled_movej_arm_segments(shared_ctx):
    source = _default_program()
    ctx = shared_ctx
    compiled = compile_cartesian_program_to_joint_stream(source, ctx)
    assert all(seg.kind in ("movej", "gripper") for seg in compiled.segments)

    mirror = _fresh_mirror(ctx, compiled)
    tracker = KinematicCarryTracker(mirror, grasp_gap_m=0.024)

    _run_ticks_through_label(tracker, compiled, stop_after_label="grip")
    assert tracker.attached
    _run_ticks_through_label(tracker, compiled, stop_after_label="place-descend")
    assert tracker.attached
    _run_ticks_through_label(tracker, compiled, stop_after_label="release")
    assert not tracker.attached


@pytest.mark.gpu
def test_kinematic_carry_tracker_does_not_attach_when_object_out_of_reach(shared_ctx):
    program = _default_program()
    ctx = shared_ctx
    mirror = _fresh_mirror(ctx, program)
    tracker = KinematicCarryTracker(mirror, grasp_gap_m=0.024, attach_dist_m=0.08)

    # Move the object far from the grasp point before the gripper closes.
    ctx.obj.set_pos(torch.tensor([[5.0, 5.0, 5.0]], device=ctx.obj.get_pos().device), zero_velocity=True)

    _run_ticks_through_label(tracker, program, stop_after_label="grip")
    assert not tracker.attached, "closing far from any object must not falsely latch"


@pytest.mark.display
@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="No DISPLAY for visual dry-run subprocess")
def test_real_visual_dry_run_subprocess():
    script = PROJECT_ROOT / "examples" / "xarm6" / "xarm6_grasp_place_traj.py"
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--executor",
            "servo_cartesian",
            "--visual",
            "--dry-run",
            "--rate",
            "50",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, (
        f"visual dry-run failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout[-4000:]}\n"
        f"stderr:\n{proc.stderr[-4000:]}"
    )
    assert "[mirror]" in proc.stdout
