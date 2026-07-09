"""Default shared grasp-place program shape tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import _grasp_place_traj as grasp_traj  # noqa: E402
from ufactory.trajectory.real_executor import _segment_servo_targets, compute_servo_stream_stats  # noqa: E402
from ufactory.trajectory.scene import default_grasp_gap_m, dry_heights  # noqa: E402


def _default_program(robot_key: str):
    ctx = dry_heights(robot_key)
    args = SimpleNamespace(
        rate=50.0,
        speed_rad_s=grasp_traj.DEFAULT_GRASP_PLACE_JOINT_SPEED_RAD_S,
        mvacc_rad_s2=grasp_traj.DEFAULT_GRASP_PLACE_JOINT_ACC_RAD_S2,
        speed_mm_s=grasp_traj.DEFAULT_GRASP_PLACE_LINEAR_SPEED_MM_S,
        mvacc_mm_s2=grasp_traj.DEFAULT_GRASP_PLACE_LINEAR_ACC_MM_S2,
        z_min_mm=0.0,
    )
    return grasp_traj._build_program(
        robot_key,
        ctx,
        args,
        grip_open_m=ctx.gripper.open_gap_m,
        grip_close_m=default_grasp_gap_m(robot_key),
    )


@pytest.mark.parametrize("robot_key", ["xarm5", "xarm6", "xarm7", "uf850"])
def test_g2_grasp_place_program_settles_after_grip_before_lift(robot_key: str):
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


def test_lite6_grasp_place_program_uses_fast_gripper_without_grip_settle():
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
def test_default_grasp_place_arm_segments_use_150mm_s_source_timing(robot_key: str):
    program = _default_program(robot_key)
    arm_stats = []
    for seg in program.segments:
        if seg.kind == "gripper":
            continue
        kind, samples, start = _segment_servo_targets(seg, program.rate)
        assert kind == "c"
        arm_stats.append(compute_servo_stream_stats(kind, samples, start, rate=program.rate, label=seg.label))

    max_xyz_speed = max(stat.max_speed for stat in arm_stats)
    assert max_xyz_speed == pytest.approx(grasp_traj.DEFAULT_GRASP_PLACE_LINEAR_SPEED_MM_S, abs=2.0)
