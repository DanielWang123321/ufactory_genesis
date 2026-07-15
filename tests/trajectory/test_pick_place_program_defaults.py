"""Default shared pick-place program shape tests."""

from __future__ import annotations

import pytest

from ufactory.cli.pick_place import _build_program
from ufactory.config import load_runtime_config
from ufactory.trajectory.real_executor import _segment_servo_targets, compute_servo_stream_stats
from ufactory.trajectory.scene import default_grasp_gap_m


def _default_program(robot_key: str):
    return _build_program(load_runtime_config(robot_key))


@pytest.mark.parametrize("robot_key", ["xarm5", "xarm6", "xarm7", "uf850"])
def test_g2_pick_place_program_settles_after_grip_before_lift(robot_key: str):
    program = _default_program(robot_key)
    labels = [seg.label for seg in program.segments]

    assert labels == [
        "home->pregrasp",
        "descend",
        "grip",
        "grip-settle",
        "lift",
        "transit",
        "place-descend",
        "release",
        "retreat",
        "return-home",
    ]
    grip = program.segments[labels.index("grip")]
    settle = program.segments[labels.index("grip-settle")]
    release = program.segments[labels.index("release")]
    grip_close_m = default_grasp_gap_m(robot_key)

    assert grip.duration == pytest.approx(2.0)
    assert release.duration == pytest.approx(2.0)
    assert settle.kind == "gripper"
    assert settle.duration == pytest.approx(0.5)
    assert settle.gap_start == pytest.approx(grip_close_m)
    assert settle.gap_end == pytest.approx(grip_close_m)
    assert labels.index("grip") < labels.index("grip-settle") < labels.index("lift")


def test_lite6_pick_place_program_uses_fast_gripper_without_grip_settle():
    program = _default_program("lite6")
    labels = [seg.label for seg in program.segments]

    assert labels == [
        "home->pregrasp",
        "descend",
        "grip",
        "lift",
        "transit",
        "place-descend",
        "place-settle",
        "release",
        "retreat",
        "return-home",
    ]
    assert "grip-settle" not in labels
    grip = program.segments[labels.index("grip")]
    place_settle = program.segments[labels.index("place-settle")]
    release = program.segments[labels.index("release")]

    assert grip.duration == pytest.approx(0.5)
    assert release.duration == pytest.approx(0.5)
    assert place_settle.duration == pytest.approx(0.18)


@pytest.mark.parametrize("robot_key", ["xarm6", "uf850", "lite6"])
def test_default_pick_place_arm_segments_use_150mm_s_source_timing(robot_key: str):
    config = load_runtime_config(robot_key)
    program = _default_program(robot_key)
    arm_stats = []
    for seg in program.segments:
        if seg.kind == "gripper":
            continue
        kind, samples, start = _segment_servo_targets(seg, program.rate)
        assert kind == "c"
        arm_stats.append(compute_servo_stream_stats(kind, samples, start, rate=program.rate, label=seg.label))

    max_xyz_speed = max(stat.max_speed for stat in arm_stats)
    assert max_xyz_speed == pytest.approx(config.motion.cartesian_speed_m_s * 1000.0, abs=2.0)
