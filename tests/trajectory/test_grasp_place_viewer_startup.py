"""Regression tests for grasp-place visual startup pose."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.robots.runtime import G2_GRIPPER_PARAMS
from ufactory.trajectory.segments import Program, Segment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))


@pytest.fixture()
def grasp_traj():
    return importlib.import_module("_grasp_place_traj")


class _FakeRobot:
    def __init__(self) -> None:
        self.set_calls = []
        self.control_calls = []

    def set_dofs_position(self, values, dofs_idx, zero_velocity=False) -> None:
        self.set_calls.append(
            (np.asarray(values, dtype=np.float64).copy(), list(dofs_idx), bool(zero_velocity))
        )

    def control_dofs_position(self, values, dofs_idx) -> None:
        self.control_calls.append((np.asarray(values, dtype=np.float64).copy(), list(dofs_idx)))


def _fake_ctx() -> SimpleNamespace:
    scene = SimpleNamespace(name="scene")
    scene.step_calls = 0
    scene.step = lambda: setattr(scene, "step_calls", scene.step_calls + 1)
    return SimpleNamespace(
        scene=scene,
        robot=_FakeRobot(),
        robot_key="xarm6_1305",
        gripper=G2_GRIPPER_PARAMS,
        arm_dof_idx=[0, 1, 2, 3, 4, 5],
        gripper_dof_idx=[6],
        all_gripper_dof_idx=[6, 7, 8, 9, 10, 11],
        obj_xy=(0.30, 0.00),
        place_xy=(0.30, 0.30),
        home_pos_base=[0.30, 0.00, 0.30],
        pre_grasp_link6_z=0.287,
        grasp_link6_z=0.187,
        lift_link6_z=0.30,
        base_pos_world=(0.30, 0.00, 0.40),
        visual_model="glb",
        finger_z_offset=0.1011,
    )


def _args(*, visual: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rate=50.0,
        speed_rad_s=0.35,
        mvacc_rad_s2=2.0,
        z_min_mm=0.0,
        substeps=2,
        visual_model="glb",
        grip_gap_mm=22.0,
        sim_grip_hold_bias_gap_mm=6.9,
        sim_grasp_weld=False,
        visual=visual,
        visual_start_hold_s=0.5,
    )


def test_prime_scene_to_program_start_sets_arm_and_open_gripper(grasp_traj, monkeypatch):
    ctx = _fake_ctx()
    q_start = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64)
    program = Program(
        segments=[
            Segment(
                kind="movej",
                duration=0.02,
                v_max=1.0,
                a_max=1.0,
                q_start=q_start,
                q_end=q_start,
            )
        ],
        rate=50.0,
    )

    monkeypatch.setattr(grasp_traj, "resolve_segment_start_arm_q", lambda _ctx, _seg: q_start)

    grasp_traj._prime_scene_to_program_start(ctx, program)

    arm_set, grip_set = ctx.robot.set_calls
    np.testing.assert_allclose(arm_set[0], q_start)
    assert arm_set[1] == ctx.arm_dof_idx
    assert arm_set[2] is True
    np.testing.assert_allclose(grip_set[0], np.zeros(len(ctx.all_gripper_dof_idx)))
    assert grip_set[1] == ctx.all_gripper_dof_idx
    assert grip_set[2] is True

    arm_control, grip_control = ctx.robot.control_calls
    np.testing.assert_allclose(arm_control[0], q_start)
    assert arm_control[1] == ctx.arm_dof_idx
    np.testing.assert_allclose(grip_control[0], np.array([G2_GRIPPER_PARAMS.open_pos]))
    assert grip_control[1] == ctx.gripper_dof_idx


def test_run_sim_defers_viewer_until_after_program_start(grasp_traj, monkeypatch):
    events: list[str] = []
    build_kwargs = {}

    def fake_build_scene(**kwargs):
        build_kwargs.update(kwargs)
        return _fake_ctx()

    def fake_open_viewer(ctx, program) -> None:
        assert ctx.scene.name == "scene"
        assert program.segments[0].label == "home->pregrasp"
        events.append("open-viewer")

    def fake_replay_sim(program, ctx, **kwargs):
        events.append("replay")
        return SimpleNamespace(
            place_error_mm=0.0,
            home_drift_mm=0.0,
            total_ticks=program.total_ticks,
            total_duration=program.total_duration,
        )

    monkeypatch.setattr(grasp_traj, "build_scene", fake_build_scene)
    monkeypatch.setattr(grasp_traj, "_open_viewer_at_program_start", fake_open_viewer)
    monkeypatch.setattr(
        grasp_traj,
        "_hold_scene_at_program_start",
        lambda _ctx, _program, *, hold_s: events.append(f"startup-hold:{hold_s}"),
    )
    monkeypatch.setattr(grasp_traj, "replay_sim", fake_replay_sim)
    monkeypatch.setattr(grasp_traj, "_hold_viewer", lambda _ctx: events.append("hold"))

    assert grasp_traj._run_sim("xarm6", _args(visual=True)) == 0

    assert build_kwargs["show_viewer"] is False
    assert events == ["open-viewer", "startup-hold:0.5", "replay", "hold"]


def test_run_sim_headless_does_not_open_deferred_viewer(grasp_traj, monkeypatch):
    events: list[str] = []

    monkeypatch.setattr(grasp_traj, "build_scene", lambda **_kwargs: _fake_ctx())
    monkeypatch.setattr(grasp_traj, "_open_viewer_at_program_start", lambda *_args: events.append("open-viewer"))
    monkeypatch.setattr(
        grasp_traj,
        "replay_sim",
        lambda program, _ctx, **_kwargs: SimpleNamespace(
            place_error_mm=0.0,
            home_drift_mm=0.0,
            total_ticks=program.total_ticks,
            total_duration=program.total_duration,
        ),
    )

    assert grasp_traj._run_sim("xarm6", _args(visual=False)) == 0

    assert events == []


def test_startup_hold_keeps_program_start_for_configured_duration(grasp_traj, monkeypatch):
    ctx = _fake_ctx()
    program = Program(
        segments=[
            Segment(
                kind="movej",
                duration=0.02,
                v_max=1.0,
                a_max=1.0,
                q_start=np.zeros(6),
                q_end=np.zeros(6),
            )
        ],
        rate=50.0,
    )
    prime_calls = []

    monkeypatch.setattr(grasp_traj, "_prime_scene_to_program_start", lambda _ctx, _program: prime_calls.append(1))
    monkeypatch.setattr(grasp_traj.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(grasp_traj.time, "sleep", lambda _s: None)

    grasp_traj._hold_scene_at_program_start(ctx, program, hold_s=0.5)

    assert len(prime_calls) == 25
    assert ctx.scene.step_calls == 25


def test_mirror_startup_hold_steps_at_program_rate(grasp_traj, monkeypatch):
    hold_calls = []
    mirror = SimpleNamespace(rate=100.0, hold_step=lambda: hold_calls.append(1))

    monkeypatch.setattr(grasp_traj.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(grasp_traj.time, "sleep", lambda _s: None)

    grasp_traj._hold_mirror_at_program_start(mirror, hold_s=0.5)

    assert len(hold_calls) == 50
