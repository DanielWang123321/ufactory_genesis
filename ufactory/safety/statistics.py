"""Signed whole-program motion statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ufactory.types import FloatArray


@dataclass(frozen=True)
class MotionStatistics:
    samples: int
    max_joint_speed_rad_s: float = 0.0
    max_joint_acceleration_rad_s2: float = 0.0
    max_cartesian_speed_m_s: float = 0.0
    max_cartesian_acceleration_m_s2: float = 0.0


def _finite_2d(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def signed_velocity_and_acceleration(values: FloatArray, *, rate_hz: float) -> tuple[FloatArray, FloatArray]:
    """Compute signed vector velocity and acceleration with static endpoints.

    Acceleration is calculated from vector velocity differences, so a constant
    speed reversal is correctly reported as a large acceleration.  A zero
    velocity is injected before the first and after the last program interval.
    """

    points = _finite_2d(values, name="timeline")
    rate = float(rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_hz must be finite and positive")
    if len(points) < 2:
        empty = np.empty((0, points.shape[1]), dtype=np.float64)
        return empty, empty
    velocity = np.diff(points, axis=0) * rate
    stationary = np.zeros((1, points.shape[1]), dtype=np.float64)
    velocity_with_boundaries = np.concatenate((stationary, velocity, stationary), axis=0)
    acceleration = np.diff(velocity_with_boundaries, axis=0) * rate
    return velocity, acceleration


def compute_motion_statistics(
    joint_positions_rad: FloatArray,
    cartesian_positions_m: FloatArray,
    *,
    rate_hz: float,
) -> MotionStatistics:
    joints = _finite_2d(joint_positions_rad, name="joint timeline")
    cartesian = _finite_2d(cartesian_positions_m, name="Cartesian timeline")
    if len(joints) != len(cartesian):
        raise ValueError("joint and Cartesian timelines must have the same length")
    joint_v, joint_a = signed_velocity_and_acceleration(joints, rate_hz=rate_hz)
    cart_v, cart_a = signed_velocity_and_acceleration(cartesian, rate_hz=rate_hz)
    return MotionStatistics(
        samples=len(joints),
        max_joint_speed_rad_s=float(np.max(np.abs(joint_v), initial=0.0)),
        max_joint_acceleration_rad_s2=float(np.max(np.abs(joint_a), initial=0.0)),
        max_cartesian_speed_m_s=float(np.max(np.linalg.norm(cart_v, axis=1), initial=0.0)),
        max_cartesian_acceleration_m_s2=float(np.max(np.linalg.norm(cart_a, axis=1), initial=0.0)),
    )
