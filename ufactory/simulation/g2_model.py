"""G2 SDK units, actual pad geometry and steady-force calibration.

This module has no Genesis or hardware connection. Simulations load the project URDF directly.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

from ufactory.grippers.g2 import G2LinkageGeometry
from ufactory.robots.paths import gripper_g2_movable_visual_urdf


G2_KP, G2_KV, G2_KI = 40.0, 5.0, 40.0
G2_ARMATURE = 0.01
G2_RATE_HZ, G2_SUBSTEPS = 50, 32
G2_CONSTRAINT_TIMECONST = 0.00125

REFERENCE_RAW_POSITION = 230

FORCE_CURVE_CURRENTS_PERCENT: tuple[float, ...] = tuple(float(c) for c in range(0, 101, 5))
FORCE_CURVE_TABLE_N: dict[float, tuple[float, ...]] = {
    0.0: (
        15.04,
        15.23,
        15.44,
        16.38,
        20.38,
        24.24,
        25.18,
        31.43,
        32.73,
        34.0,
        35.13,
        43.56,
        44.61,
        45.79,
        46.78,
        47.59,
        48.49,
        58.49,
        59.71,
        60.55,
        63.79,
    ),
    250.0: (
        15.04,
        15.23,
        15.44,
        15.89,
        19.73,
        23.84,
        24.59,
        31.43,
        32.73,
        34.28,
        36.25,
        44.45,
        44.61,
        45.79,
        46.78,
        47.78,
        48.68,
        58.49,
        59.71,
        60.55,
        63.79,
    ),
    500.0: (
        14.93,
        15.04,
        15.14,
        16.06,
        20.28,
        23.76,
        24.59,
        31.43,
        32.73,
        33.89,
        35.33,
        43.56,
        44.61,
        45.79,
        46.78,
        47.78,
        48.68,
        58.72,
        59.71,
        60.55,
        62.95,
    ),
    750.0: (
        14.83,
        14.82,
        15.03,
        15.98,
        22.42,
        24.06,
        25.25,
        31.43,
        32.73,
        33.89,
        35.33,
        43.72,
        44.61,
        45.79,
        46.88,
        48.14,
        49.04,
        58.72,
        59.71,
        60.55,
        62.95,
    ),
    1000.0: (
        17.67,
        17.32,
        17.75,
        17.75,
        23.14,
        23.84,
        25.22,
        31.59,
        32.81,
        33.89,
        35.53,
        43.66,
        44.61,
        45.79,
        46.88,
        48.01,
        48.83,
        59.04,
        59.71,
        60.55,
        62.95,
    ),
    1250.0: (
        17.67,
        17.32,
        17.75,
        17.75,
        23.14,
        23.59,
        24.2,
        31.59,
        32.81,
        33.89,
        35.73,
        43.76,
        44.61,
        45.79,
        46.88,
        48.01,
        48.83,
        59.04,
        59.71,
        60.95,
        61.53,
    ),
    1500.0: (
        16.2,
        16.3,
        16.31,
        16.25,
        22.62,
        23.68,
        24.98,
        32.45,
        33.22,
        34.03,
        36.04,
        44.06,
        44.61,
        45.79,
        46.88,
        48.01,
        48.83,
        59.04,
        59.71,
        60.95,
        61.53,
    ),
    1750.0: (
        16.41,
        16.41,
        16.41,
        16.56,
        22.73,
        23.85,
        24.55,
        31.92,
        33.22,
        34.03,
        36.04,
        44.06,
        45.1,
        46.22,
        47.21,
        48.61,
        58.36,
        59.23,
        60.16,
        60.6,
        61.41,
    ),
    2000.0: (
        16.62,
        16.82,
        16.62,
        16.88,
        22.99,
        23.76,
        23.9,
        31.92,
        33.22,
        34.13,
        36.04,
        44.06,
        45.1,
        46.22,
        47.21,
        48.61,
        58.36,
        59.23,
        60.16,
        61.82,
        61.58,
    ),
    2250.0: (
        17.22,
        17.62,
        17.42,
        17.49,
        22.91,
        23.66,
        28.65,
        31.94,
        33.09,
        34.13,
        43.38,
        44.34,
        45.1,
        46.22,
        47.21,
        48.61,
        58.36,
        59.23,
        60.16,
        61.01,
        61.01,
    ),
    2500.0: (
        18.02,
        18.02,
        18.02,
        18.24,
        23.11,
        23.63,
        24.41,
        32.35,
        33.65,
        34.56,
        43.38,
        44.34,
        45.44,
        46.54,
        47.21,
        48.61,
        58.36,
        59.13,
        60.23,
        61.66,
        62.8,
    ),
    2750.0: (
        22.64,
        22.64,
        22.64,
        22.54,
        23.24,
        23.43,
        23.71,
        32.39,
        33.54,
        34.39,
        43.15,
        44.23,
        45.22,
        46.41,
        47.38,
        48.46,
        58.16,
        59.14,
        60.14,
        62.18,
        63.21,
    ),
    3000.0: (
        22.42,
        22.42,
        22.42,
        22.39,
        23.01,
        23.02,
        23.1,
        32.08,
        33.36,
        34.18,
        42.91,
        44.11,
        45.22,
        46.41,
        47.38,
        48.46,
        58.16,
        59.09,
        60.15,
        60.49,
        60.6,
    ),
    3250.0: (
        23.04,
        23.03,
        23.04,
        22.81,
        22.81,
        22.82,
        22.95,
        31.73,
        32.7,
        33.76,
        43.22,
        43.6,
        45.14,
        46.14,
        47.14,
        48.37,
        57.93,
        58.68,
        60.12,
        61.78,
        63.08,
    ),
    3500.0: (
        22.82,
        23.02,
        23.42,
        23.22,
        23.22,
        23.22,
        23.25,
        32.05,
        32.82,
        34.24,
        42.64,
        44.01,
        44.81,
        46.01,
        47.01,
        48.03,
        57.83,
        59.01,
        59.8,
        60.81,
        63.21,
    ),
}


def checked(value: float, low: float, high: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and within {low:g}–{high:g}")
    return value


def sdk_position_counts(position_mm: float) -> int:
    position = checked(position_mm, 0, 84, "position_mm")
    return int((math.degrees(math.asin((position - 16) / 110)) + 8.33) * 18.28)


def sdk_position_mm(raw_position: float) -> float:
    raw_position = float(raw_position)
    if not math.isfinite(raw_position):
        raise ValueError("raw_position must be finite")
    return math.sin(math.radians(raw_position / 18.28 - 8.33)) * 110 + 16


REFERENCE_GAP_MM = sdk_position_mm(REFERENCE_RAW_POSITION)


def speed_mm_s(speed_percent: float) -> float:
    speed = checked(speed_percent, 0, 100, "speed_percent")
    return max(15.0, 225.0 * speed / 100.0) if speed else 0.0


def sdk_speed_rpm(speed_mm_s: float) -> int:
    speed = checked(speed_mm_s, 0, 225, "speed_mm_s")
    return int(((speed * 60) / 9.88235 + 140) / 0.4) if speed else 0


@dataclass(frozen=True)
class ForceLookup:
    force_n: float
    requested_rpm: float
    calibration_rpm: float
    outside_calibration: bool


class ForceCurve:
    """Piecewise linear in current, then speed; no fitted or smoothed curve."""

    def __init__(self, points: dict[float, dict[float, float]] | None = None):
        if points is None:
            points = {
                float(rpm): dict(zip(FORCE_CURVE_CURRENTS_PERCENT, forces, strict=True))
                for rpm, forces in FORCE_CURVE_TABLE_N.items()
            }
        self.points: dict[float, dict[float, float]] = {}
        for rpm, curve in points.items():
            rpm = float(rpm)
            if not math.isfinite(rpm):
                raise ValueError("invalid force calibration speed")
            row: dict[float, float] = {}
            for current, force in curve.items():
                current, force = float(current), float(force)
                if not math.isfinite(current) or not math.isfinite(force) or force <= 0:
                    raise ValueError("invalid force calibration point")
                row[current] = force
            self.points[rpm] = row
        self.speeds = sorted(self.points)
        if len(self.speeds) < 2 or self.speeds[0] != 0 or self.speeds[-1] != 3500:
            raise ValueError("force curves must cover 0–3500 rpm")
        for curve in self.points.values():
            if len(curve) < 2 or min(curve) != 0 or max(curve) != 100:
                raise ValueError("each force curve must cover 0–100% current")

    @property
    def maximum_force_n(self) -> float:
        return max(force for curve in self.points.values() for force in curve.values())

    @staticmethod
    def _bracket(knots: list[float], value: float) -> tuple[float, float, float]:
        if value in knots:
            return value, value, 0.0
        upper = bisect_right(knots, value)
        lo, hi = knots[upper - 1], knots[upper]
        return lo, hi, (value - lo) / (hi - lo)

    def _at_speed(self, rpm: float, current: float) -> float:
        points = self.points[rpm]
        lo, hi, ratio = self._bracket(sorted(points), current)
        a, b = points[lo], points[hi]
        return a + ratio * (b - a)

    def lookup(self, rpm: float, force_percent: float) -> ForceLookup:
        current = checked(force_percent, 1, 100, "force_percent")
        rpm = checked(rpm, 0, 1e9, "rpm")
        calibration_rpm = min(rpm, self.speeds[-1])
        lo, hi, ratio = self._bracket(self.speeds, calibration_rpm)
        a, b = self._at_speed(lo, current), self._at_speed(hi, current)
        return ForceLookup(
            a + ratio * (b - a),
            rpm,
            calibration_rpm,
            rpm > self.speeds[-1],
        )


@dataclass(frozen=True)
class G2Model:
    """Calibrated G2 linkage and force curve, referred to the drive joint.

    A force sensor between two symmetric pads reads the load at either pad, so
    virtual work gives tau_drive = F_sensor * abs(dgap/dq) without an extra
    factor of two. The calibrated torque gain combines reduction and efficiency;
    it is not an identified mechanical gear ratio.
    """

    linkage: G2LinkageGeometry
    curve: ForceCurve
    rated_motor_torque_nm: float = 0.22
    peak_motor_torque_nm: float = 0.5
    reference_raw_position: int = REFERENCE_RAW_POSITION

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> G2Model:
        return cls(G2LinkageGeometry.from_urdf(gripper_g2_movable_visual_urdf()), ForceCurve())

    @property
    def reference_gap_mm(self) -> float:
        return sdk_position_mm(self.reference_raw_position)

    @property
    def reference_jacobian_m_rad(self) -> float:
        q = self.linkage.drive_for_gap_m(self.reference_gap_mm / 1000)
        return self.linkage.gap_jacobian_m_per_rad(q)

    @property
    def effective_torque_gain(self) -> float:
        return self.curve.maximum_force_n * self.reference_jacobian_m_rad / self.peak_motor_torque_nm

    def torque_limit_nm(self, lookup: ForceLookup) -> float:
        return lookup.force_n * self.reference_jacobian_m_rad

    def static_force_n(self, lookup: ForceLookup, gap_mm: float) -> float:
        q = self.linkage.drive_for_gap_m(checked(gap_mm, 0, 84, "gap_mm") / 1000)
        return self.torque_limit_nm(lookup) / self.linkage.gap_jacobian_m_per_rad(q)

    def motor_torque_estimate_nm(self, drive_torque_nm: float) -> float:
        return float(drive_torque_nm) / self.effective_torque_gain


def g2_physics_summary(*, simulation=None, rate_hz: float = G2_RATE_HZ) -> dict:
    """Effective controller constants and calibrated G2 model parameters."""
    from ufactory.config.models import G2_PHYSICS_PROFILE

    model = G2Model.load()
    summary = {
        "physics_profile": G2_PHYSICS_PROFILE,
        "kp": G2_KP,
        "kv": G2_KV,
        "ki": G2_KI,
        "armature_kg_m2": G2_ARMATURE,
        "rate_hz": rate_hz,
        "substeps": G2_SUBSTEPS if simulation is None else simulation.substeps,
        "constraint_time_constant_s": G2_CONSTRAINT_TIMECONST
        if simulation is None
        else max(simulation.constraint_time_constant_s, 2 / rate_hz / simulation.substeps),
        "reference_raw_position": REFERENCE_RAW_POSITION,
        "reference_gap_mm": REFERENCE_GAP_MM,
        "reference_jacobian_m_rad": model.reference_jacobian_m_rad,
        "effective_torque_gain": model.effective_torque_gain,
        "rated_motor_torque_nm": model.rated_motor_torque_nm,
        "peak_motor_torque_nm": model.peak_motor_torque_nm,
        "maximum_force_n": model.curve.maximum_force_n,
        "drive_limits_rad": [model.linkage.drive_lower_rad, model.linkage.drive_upper_rad],
    }
    if simulation is not None:
        summary["simulation"] = asdict(simulation)
    return summary
