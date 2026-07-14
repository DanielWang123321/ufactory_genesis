from __future__ import annotations

import numpy as np
import pytest

from ufactory.safety.statistics import compute_motion_statistics, signed_velocity_and_acceleration


def test_signed_reversal_produces_acceleration_not_zero():
    velocity, acceleration = signed_velocity_and_acceleration(np.asarray([[0.0], [0.1], [0.0]]), rate_hz=10.0)
    assert velocity[:, 0].tolist() == pytest.approx([1.0, -1.0])
    assert np.max(np.abs(acceleration)) == pytest.approx(20.0)


def test_cartesian_acceleration_uses_vector_difference():
    joints = np.asarray([[0.0], [0.1], [0.0]])
    xyz = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    stats = compute_motion_statistics(joints, xyz, rate_hz=10.0)
    assert stats.max_cartesian_speed_m_s == pytest.approx(1.0)
    assert stats.max_cartesian_acceleration_m_s2 == pytest.approx(20.0)


@pytest.mark.parametrize(
    "values",
    [np.asarray([[np.nan]]), np.asarray([[np.inf]]), np.asarray([[-np.inf]])],
)
def test_non_finite_timeline_is_rejected(values: np.ndarray):
    with pytest.raises(ValueError, match="NaN or infinity"):
        signed_velocity_and_acceleration(values, rate_hz=50.0)


def test_single_sample_and_hold_have_zero_motion():
    velocity, acceleration = signed_velocity_and_acceleration(np.asarray([[1.0, 2.0]]), rate_hz=50.0)
    assert velocity.size == 0
    assert acceleration.size == 0
    stats = compute_motion_statistics(np.ones((4, 2)), np.ones((4, 3)), rate_hz=50.0)
    assert stats.max_joint_speed_rad_s == 0.0
    assert stats.max_joint_acceleration_rad_s2 == 0.0
