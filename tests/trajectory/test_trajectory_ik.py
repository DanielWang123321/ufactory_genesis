"""Tests for host-side IK compilation of Cartesian trajectory programs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.segments import JointLimits, Program, make_gripper, make_movel


def test_compile_cartesian_program_to_joint_stream_preserves_ticks_labels_and_uses_continuous_seed(monkeypatch):
    calls: list[np.ndarray | None] = []

    def fake_solve(_ctx, xyz_base, *, init_qpos):
        init_copy = None if init_qpos is None else np.asarray(init_qpos, dtype=np.float64).copy()
        calls.append(init_copy)
        xyz = np.asarray(xyz_base, dtype=np.float64).reshape(3)
        return np.array([xyz[0], xyz[1], xyz[2], len(calls) * 0.01, 0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr("ufactory.trajectory.ik._solve_base_xyz", fake_solve)
    ctx = SimpleNamespace(
        robot_key="xarm6_1305",
        robot_urdf_path="/tmp/xarm6_g2.urdf",
        kinematics_yaml_path="/tmp/xarm6_kinematics_F56A14.yaml",
        kinematics_suffix="F56A14",
    )
    limits = JointLimits(0.35, 2.0, 0.14, 0.8)
    move_seg = make_movel(
        np.array([0.30, 0.0, 0.30]),
        np.array([0.30, 0.0, 0.20]),
        rate=50.0,
        limits=limits,
        label="descend",
    )
    grip_seg = make_gripper(
        0.084,
        0.022,
        rate=50.0,
        duration_s=0.02,
        label="grip",
    )
    source = Program(
        rate=50.0,
        robot_key="xarm6_1305",
        limits=limits,
        metadata={"kind": "mixed"},
        segments=[move_seg, grip_seg],
    )
    _move_samples, move_ticks = move_seg.samples(source.rate)

    compiled = compile_cartesian_program_to_joint_stream(source, ctx)

    assert compiled.metadata["kind"] == "joint-from-cartesian-ik"
    assert compiled.metadata["ik_compiled_movel_segments"] == 1
    assert compiled.metadata["ik_joint_retimed"] is True
    # Tick count may grow when joint LSPB needs longer than the Cartesian window.
    assert compiled.metadata["ik_compiled_ticks"] >= move_ticks
    assert compiled.metadata["ik_kinematics_suffix"] == "F56A14"
    assert [seg.kind for seg in compiled.segments] == ["movej", "gripper"]
    assert [seg.label for seg in compiled.segments] == ["descend", "grip"]
    assert compiled.total_ticks >= source.total_ticks

    move = compiled.segments[0]
    q_samples, n = move.samples(compiled.rate)
    assert n == compiled.metadata["ik_compiled_ticks"]
    assert move.q_start.shape == (6,)
    assert move.q_end.shape == (6,)
    np.testing.assert_allclose(move.q_end, q_samples[-1])
    assert calls[0] is None
    np.testing.assert_allclose(calls[1], move.q_start)
    # IK still runs once per Cartesian sample (+ start); retiming happens after.
    assert len(calls) == move_ticks + 1

def test_collapse_joint_keypoints_drops_plateau():
    from ufactory.trajectory.ik import collapse_joint_keypoints

    q0 = np.zeros(6)
    rows = np.vstack(
        [
            np.zeros(6),
            np.zeros(6),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    keys = collapse_joint_keypoints(q0, rows)
    assert keys.shape == (3, 6)
    np.testing.assert_allclose(keys[0], 0.0)
    np.testing.assert_allclose(keys[1, 0], 0.1)
    np.testing.assert_allclose(keys[2, 0], 0.2)


def test_retime_ik_joint_stream_bounds_finite_diff_acc():
    from ufactory.trajectory.ik import retime_ik_joint_stream
    from ufactory.trajectory.real_executor import compute_servo_stream_stats

    q0 = np.zeros(6)
    # Leading plateau then a jump — the failure mode seen at 100 Hz.
    rows = np.vstack([np.zeros((4, 6)), np.linspace(0.0, 0.05, 20)[:, None] * np.array([0, 1, 0, 0, 1, 0])])
    q_out, n, dur, n_keys = retime_ik_joint_stream(
        q0, rows, rate=100.0, duration_s=0.25, v_max=0.35, a_max=2.0
    )
    assert n_keys < rows.shape[0] + 1
    # Short Cartesian duration is extended to the joint LSPB bound.
    assert dur >= 0.25 - 1e-9
    stats = compute_servo_stream_stats("j", q_out, q0, rate=100.0, label="synthetic")
    assert stats.max_acc <= 2.0 + 1e-3
    assert stats.max_speed <= 0.35 + 1e-3