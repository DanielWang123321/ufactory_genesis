"""Pure logic tests for the xArm6 trajectory-planning kernel."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from ufactory.trajectory.profile import (
    gap_lspb_samples,
    joint_lspb_samples,
    linear_cartesian_samples,
    lspb_duration,
)
from ufactory.trajectory.segments import build_pickplace_program


def _finite_difference_velocity(samples: np.ndarray, start: np.ndarray, rate: float) -> np.ndarray:
    points = np.vstack([np.asarray(start, dtype=np.float64).reshape(1, -1), samples])
    return np.diff(points, axis=0) * float(rate)


def test_profile_import_does_not_load_genesis():
    code = (
        "import sys; "
        "import ufactory.trajectory.profile; "
        "from ufactory.trajectory import build_pickplace_program; "
        "print('genesis' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_lspb_duration_matches_triangular_and_trapezoidal_cases():
    assert lspb_duration(0.0, v_max=0.5, a_max=1.0) == pytest.approx(0.0)
    assert lspb_duration(1.0, v_max=0.0, a_max=1.0) == pytest.approx(0.0)

    # Triangular: distance is too short to reach v_max.
    assert lspb_duration(0.01, v_max=0.5, a_max=1.0) == pytest.approx(0.2)

    # Trapezoidal: accel 0.5 s + cruise 1.5 s + decel 0.5 s.
    assert lspb_duration(1.0, v_max=0.5, a_max=1.0) == pytest.approx(2.5)


def test_joint_lspb_samples_synchronize_axes_and_respect_limits():
    rate = 50.0
    v_max = 0.5
    a_max = 1.0
    q0 = np.array([0.0, 0.0, 0.0])
    q1 = np.array([1.0, 0.2, -0.4])

    q, n, duration = joint_lspb_samples(q0, q1, rate=rate, v_max=v_max, a_max=a_max)

    assert q.shape == (n, 3)
    assert n == round(duration * rate)
    assert duration == pytest.approx(n / rate)
    np.testing.assert_allclose(q[-1], q1, atol=1e-12)

    signed_steps = np.diff(np.vstack([q0, q]), axis=0) * np.sign(q1 - q0)
    assert np.all(signed_steps >= -1e-12)

    vel = _finite_difference_velocity(q, q0, rate)
    acc = np.diff(vel, axis=0) * rate
    assert np.max(np.abs(vel)) <= v_max + 1e-10
    assert np.max(np.abs(acc)) <= a_max + 1e-10


def test_joint_lspb_samples_identical_target_returns_single_endpoint():
    q0 = np.array([1.0, -1.0, 0.25])

    q, n, duration = joint_lspb_samples(q0, q0, rate=50.0, v_max=0.5, a_max=1.0)

    assert n == 1
    assert duration == pytest.approx(0.0)
    np.testing.assert_allclose(q, q0.reshape(1, -1))


def test_linear_cartesian_samples_stay_on_line_and_respect_limits():
    rate = 50.0
    v_max = 0.2
    a_max = 0.5
    p0 = np.array([0.10, -0.20, 0.05])
    p1 = np.array([0.40, 0.20, 0.05])
    path = p1 - p0
    direction = path / np.linalg.norm(path)

    p, n, duration = linear_cartesian_samples(p0, p1, rate=rate, v_max=v_max, a_max=a_max)

    assert p.shape == (n, 3)
    assert n == round(duration * rate)
    np.testing.assert_allclose(p[-1], p1, atol=1e-12)

    rel = p - p0
    np.testing.assert_allclose(np.cross(rel, direction), 0.0, atol=1e-12)
    progress = rel @ direction
    assert np.all(np.diff(np.r_[0.0, progress]) >= -1e-12)

    vel = _finite_difference_velocity(p, p0, rate)
    speed = np.linalg.norm(vel, axis=1)
    accel = np.diff(speed) * rate
    assert np.max(speed) <= v_max + 1e-10
    assert np.max(np.abs(accel)) <= a_max + 1e-9


def test_gap_lspb_samples_use_requested_duration_without_overshoot():
    rate = 50.0
    gap_start = 0.084
    gap_end = 0.024

    gaps, n, duration = gap_lspb_samples(gap_start, gap_end, rate=rate, duration_s=2.0)

    assert gaps.shape == (100, 1)
    assert n == 100
    assert duration == pytest.approx(2.0)
    assert gaps[-1, 0] == pytest.approx(gap_end)
    assert gaps[:, 0].max() <= gap_start
    assert gaps[:, 0].min() >= gap_end
    assert np.all(np.diff(gaps[:, 0]) <= 1e-12)


def test_program_segments_share_the_same_sampling_contract():
    rate = 20.0
    q0 = np.zeros(6)
    q1 = np.array([0.2, -0.1, 0.05, 0.0, 0.1, -0.05])
    waypoints = [
        {"type": "movej", "q_start": q0, "q_end": q1, "label": "joint"},
        {
            "type": "movel",
            "pose_start": [0.30, 0.00, 0.30],
            "pose_end": [0.30, 0.10, 0.30],
            "label": "line",
        },
        {"type": "gripper", "gap_start": 0.084, "gap_end": 0.024, "duration": 1.0, "label": "grip"},
    ]

    program = build_pickplace_program(
        rate=rate,
        speed_rad_s=0.5,
        mvacc_rad_s2=1.0,
        waypoints=waypoints,
    )

    assert len(program.segments) == 3
    assert program.total_ticks == sum(seg.samples_count for seg in program.segments)
    assert program.total_duration == pytest.approx(sum(seg.duration for seg in program.segments))

    resolved = program.iter_samples()
    assert [kind for kind, _arr in resolved] == ["movej", "movel", "gripper"]
    np.testing.assert_allclose(resolved[0][1][-1], q1)
    np.testing.assert_allclose(resolved[1][1][-1], [0.30, 0.10, 0.30])
    assert resolved[2][1][-1, 0] == pytest.approx(0.024)
    for seg, (_kind, arr) in zip(program.segments, resolved, strict=True):
        assert arr.shape[0] == seg.samples_count
