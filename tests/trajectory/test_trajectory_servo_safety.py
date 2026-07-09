"""Servo-mode safety tests for the xArm6 trajectory real executor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ufactory.trajectory import (
    CartesianWaypoint,
    JointWaypoint,
    RealExecutorConfig,
    ServoLimits,
    TrajectorySafetyError,
    TrajectoryPlannerConfig,
    build_pickplace_program,
    check_segment_safety,
    plan_cartesian_waypoints,
    plan_joint_waypoints,
    replay_real,
    validate_servo_stream,
)
from ufactory.trajectory.real_executor import _segment_servo_targets
from ufactory.trajectory.real_executor import _read_reported_target


def test_joint_jump_at_100hz_fails_with_reported_degrees_per_second():
    start = np.zeros(6)
    samples = np.zeros((1, 6))
    samples[0, 0] = math.radians(10.0)

    with pytest.raises(TrajectorySafetyError, match=r"1000\.0 deg/s"):
        validate_servo_stream("j", samples, start, rate=100.0, limits=ServoLimits(), label="bad-joint")


def test_cartesian_jump_at_100hz_fails_with_reported_mm_per_second():
    start = np.zeros(6)
    samples = np.zeros((1, 6))
    samples[0, 0] = 100.0

    with pytest.raises(TrajectorySafetyError, match=r"10000\.0 mm/s"):
        validate_servo_stream("c", samples, start, rate=100.0, limits=ServoLimits(), label="bad-cart")


def test_default_grasp_place_program_passes_servo_stream_limits_at_50hz():
    program = build_pickplace_program(
        rate=50.0,
        speed_rad_s=0.35,
        mvacc_rad_s2=2.0,
        waypoints=_default_waypoints(),
    )

    stats = []
    for seg in program.segments:
        if seg.kind == "gripper":
            continue
        kind, samples, start = _segment_servo_targets(seg, program.rate)
        stats.append(
            validate_servo_stream(kind, samples, start, rate=program.rate, limits=ServoLimits(), label=seg.label)
        )

    max_xyz_speed = max(s.max_speed for s in stats if s.kind == "c")
    max_xyz_acc = max(s.max_acc for s in stats if s.kind == "c")
    assert max_xyz_speed == pytest.approx(142.47, abs=0.2)
    assert max_xyz_acc == pytest.approx(800.0, abs=0.5)


def test_replay_real_rejects_movel_program_with_servo_j():
    program = plan_cartesian_waypoints(
        TrajectoryPlannerConfig(robot_key="xarm6", rate=50.0),
        [0.30, 0.00, 0.30],
        [CartesianWaypoint([0.30, 0.10, 0.30], label="line")],
    )

    with pytest.raises(TrajectorySafetyError, match="cannot replay MoveL"):
        replay_real(program, RealExecutorConfig(executor="servo_j", dry_run=True, rate=50.0))


def test_replay_real_rejects_movej_program_with_servo_cartesian():
    program = plan_joint_waypoints(
        TrajectoryPlannerConfig(robot_key="xarm6", rate=50.0),
        np.zeros(6),
        [JointWaypoint(np.zeros(6), label="joint")],
    )

    with pytest.raises(TrajectorySafetyError, match="cannot replay MoveJ"):
        replay_real(program, RealExecutorConfig(executor="servo_cartesian", dry_run=True, rate=50.0))


@pytest.mark.parametrize("dof", [5, 6, 7])
def test_servo_j_reported_target_uses_program_dof(dof: int):
    class Arm:
        def get_servo_angle(self, *, is_radian):
            assert is_radian is True
            return 0, list(range(10))

    reported = _read_reported_target(Arm(), "j", dof)

    assert reported.shape == (dof,)
    np.testing.assert_allclose(reported, np.arange(dof, dtype=np.float64))


def test_full_sample_joint_limit_check_catches_middle_sample_violation():
    class FakeMoveJ:
        kind = "movej"
        label = "middle-joint"
        q_start = np.zeros(6)

        def samples(self, rate):
            samples = np.zeros((3, 6))
            samples[1, 2] = 0.2
            return samples, samples.shape[0]

    with pytest.raises(TrajectorySafetyError, match=r"sample 2 joint3"):
        check_segment_safety(
            FakeMoveJ(),
            rate=50.0,
            lower=np.full(6, -0.1),
            upper=np.full(6, 0.1),
            z_min_mm=0.0,
            margin=0.0,
        )


def test_full_sample_z_min_check_catches_middle_sample_violation():
    class FakeMoveL:
        kind = "movel"
        label = "middle-z"
        pose_start = np.array([0.3, 0.0, 0.1])

        def samples(self, rate):
            samples = np.array(
                [
                    [0.3, 0.0, 0.1],
                    [0.3, 0.0, -0.01],
                    [0.3, 0.0, 0.1],
                ],
                dtype=np.float64,
            )
            return samples, samples.shape[0]

    with pytest.raises(TrajectorySafetyError, match=r"sample 2 .*below minimum 0\.0 mm"):
        check_segment_safety(
            FakeMoveL(),
            rate=50.0,
            lower=np.full(6, -math.pi),
            upper=np.full(6, math.pi),
            z_min_mm=0.0,
            margin=0.0,
        )


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
