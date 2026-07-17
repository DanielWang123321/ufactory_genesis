from __future__ import annotations

import numpy as np
import pytest

from ufactory.trajectory.compile import compile_program_for_servo_j
from ufactory.trajectory.segments import Program, Segment


class IK:
    def __init__(self, invalid=False):
        self.invalid = invalid
        self.poses = []

    def forward(self, q):
        return np.asarray((0.3, 0.0, 0.3))

    def inverse(self, pose, seed):
        self.poses.append(np.asarray(pose).copy())
        if self.invalid:
            return np.full_like(seed, np.nan)
        result = np.asarray(seed).copy()
        result[0] += 0.001
        return result


def _line_only():
    line = Segment(
        "movel",
        0.04,
        0.1,
        0.2,
        "line",
        pose_start=np.array([0.3, 0.0, 0.3]),
        pose_end=np.array([0.31, 0.0, 0.3]),
        samples_count=2,
    )
    return Program([line], rate=50.0, robot_key="xarm6_1305")


def test_compile_mixed_program_preserves_non_cartesian_segments():
    q = np.zeros(6)
    move = Segment("movej", 0.02, 1.0, 2.0, "start", q_start=q, q_end=q, q_samples=q[None, :], samples_count=1)
    grip = Segment("gripper", 0.02, 1.0, 1.0, "grip", gap_start=0.08, gap_end=0.02, samples_count=1)
    program = Program([move, *_line_only().segments, grip], rate=50.0, robot_key="xarm6_1305")

    ik = IK()
    compiled = compile_program_for_servo_j(program, ik)

    assert [segment.kind for segment in compiled.segments] == ["movej", "movej", "gripper"]
    assert compiled.metadata["servo_j_compiled"] is True
    ticks = compiled.metadata["servo_j_compiled_ticks"]
    assert ticks == len(compiled.segments[1].q_samples)
    np.testing.assert_allclose(compiled.segments[1].q_end[0], ticks * 0.001)
    assert all(pose.shape == (7,) for pose in ik.poses)
    assert all(np.allclose(pose[3:], (1.0, 0.0, 0.0, 0.0)) for pose in ik.poses)


def test_compile_requires_joint_start_and_finite_ik():
    with pytest.raises(ValueError, match="preceding MoveJ"):
        compile_program_for_servo_j(_line_only(), IK())

    q = np.zeros(6)
    move = Segment("movej", 0.02, 1.0, 2.0, "start", q_start=q, q_end=q, q_samples=q[None, :], samples_count=1)
    program = Program([move, *_line_only().segments], rate=50.0, robot_key="xarm6_1305")
    with pytest.raises(ValueError, match="non-finite"):
        compile_program_for_servo_j(program, IK(invalid=True))
