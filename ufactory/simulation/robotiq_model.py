"""Robotiq 2F-85 and 2F-140 units, actual pad geometry and steady-force calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

from ufactory.grippers.robotiq import RobotiqLinkageGeometry, RobotiqSpec, get_robotiq_spec
from ufactory.robots.paths import robotiq_2f_85_movable_visual_urdf, robotiq_2f_140_movable_visual_urdf
from ufactory.simulation.g2_model import checked

ROBOTIQ_KP, ROBOTIQ_KV, ROBOTIQ_KI = 120.0, 10.0, 160.0
ROBOTIQ_ARMATURE = 0.02
ROBOTIQ_RATE_HZ, ROBOTIQ_SUBSTEPS = 50, 32
ROBOTIQ_CONSTRAINT_TIMECONST = 0.00125


@dataclass(frozen=True)
class RobotiqForceLookup:
    force_n: float
    requested_speed_mm_s: float
    force_percent: float
    outside_calibration: bool = False


class RobotiqForceCurve:
    """Linear datasheet steady-force curve across ``force_percent`` [1..100]."""

    def __init__(self, force_min_n: float, force_max_n: float) -> None:
        self.force_min_n = checked(force_min_n, 1e-6, 1e4, "force_min_n")
        self.force_max_n = checked(force_max_n, self.force_min_n, 1e4, "force_max_n")

    @property
    def maximum_force_n(self) -> float:
        return self.force_max_n

    def lookup(self, speed_mm_s: float, force_percent: float) -> RobotiqForceLookup:
        requested_speed = checked(speed_mm_s, 0, 1e4, "speed_mm_s")
        pct = checked(force_percent, 1, 100, "force_percent")
        ratio = (pct - 1.0) / 99.0
        force_n = self.force_min_n + ratio * (self.force_max_n - self.force_min_n)
        return RobotiqForceLookup(
            force_n=force_n,
            requested_speed_mm_s=requested_speed,
            force_percent=pct,
            outside_calibration=False,
        )


def robotiq_speed_mm_s(spec: RobotiqSpec, speed_percent: float) -> float:
    speed = checked(speed_percent, 0, 100, "speed_percent")
    return spec.speed_percent_to_mm_s(speed) if speed > 0.0 else 0.0


@dataclass(frozen=True)
class RobotiqModel:
    """Calibrated Robotiq 2F linkage and force curve, referred to the drive joint."""

    spec: RobotiqSpec
    linkage: RobotiqLinkageGeometry
    curve: RobotiqForceCurve
    reference_gap_mm: float

    @classmethod
    @lru_cache(maxsize=2)
    def load(cls, variant: str = "robotiq_2f_85") -> RobotiqModel:
        spec = get_robotiq_spec(variant)
        urdf = (
            robotiq_2f_85_movable_visual_urdf()
            if spec.adapter == "robotiq_2f_85"
            else robotiq_2f_140_movable_visual_urdf()
        )
        linkage = RobotiqLinkageGeometry.from_urdf(urdf, prefix=spec.prefix)
        return cls(
            spec=spec,
            linkage=linkage,
            curve=RobotiqForceCurve(spec.force_min_n, spec.force_max_n),
            reference_gap_mm=spec.open_gap_mm * 0.5,
        )

    @property
    def reference_jacobian_m_rad(self) -> float:
        q = self.linkage.drive_for_gap_m(self.reference_gap_mm / 1000.0)
        return self.linkage.gap_jacobian_m_per_rad(q)

    def torque_limit_nm(self, lookup: RobotiqForceLookup) -> float:
        return lookup.force_n * self.reference_jacobian_m_rad

    def static_force_n(self, lookup: RobotiqForceLookup, gap_mm: float) -> float:
        q = self.linkage.drive_for_gap_m(checked(gap_mm, 0, self.spec.open_gap_mm, "gap_mm") / 1000.0)
        return self.torque_limit_nm(lookup) / self.linkage.gap_jacobian_m_per_rad(q)

    def motor_torque_estimate_nm(self, drive_torque_nm: float) -> float:
        # Report normalised fraction of peak actuator torque scaled cleanly to drive torque.
        peak_nm = self.curve.maximum_force_n * self.reference_jacobian_m_rad
        if peak_nm <= 0.0 or not math.isfinite(peak_nm):
            return float(drive_torque_nm)
        return float(drive_torque_nm) / peak_nm


def robotiq_physics_summary(
    variant: str = "robotiq_2f_85",
    *,
    simulation=None,
    rate_hz: float = ROBOTIQ_RATE_HZ,
) -> dict:
    """Effective controller constants and calibrated Robotiq model parameters."""
    from ufactory.config.models import ROBOTIQ_2F_85_PHYSICS_PROFILE, ROBOTIQ_2F_140_PHYSICS_PROFILE

    model = RobotiqModel.load(variant)
    profile_name = (
        ROBOTIQ_2F_85_PHYSICS_PROFILE if model.spec.adapter == "robotiq_2f_85" else ROBOTIQ_2F_140_PHYSICS_PROFILE
    )
    summary = {
        "physics_profile": profile_name,
        "variant": model.spec.adapter,
        "kp": ROBOTIQ_KP,
        "kv": ROBOTIQ_KV,
        "ki": ROBOTIQ_KI,
        "armature_kg_m2": ROBOTIQ_ARMATURE,
        "rate_hz": rate_hz,
        "substeps": ROBOTIQ_SUBSTEPS if simulation is None else simulation.substeps,
        "constraint_time_constant_s": ROBOTIQ_CONSTRAINT_TIMECONST
        if simulation is None
        else max(simulation.constraint_time_constant_s, 2 / rate_hz / simulation.substeps),
        "open_gap_mm": model.spec.open_gap_mm,
        "speed_range_mm_s": [model.spec.speed_min_mm_s, model.spec.speed_max_mm_s],
        "force_range_n": [model.spec.force_min_n, model.spec.force_max_n],
        "reference_gap_mm": model.reference_gap_mm,
        "reference_jacobian_m_rad": model.reference_jacobian_m_rad,
        "maximum_force_n": model.curve.maximum_force_n,
        "drive_limits_rad": [model.linkage.drive_lower_rad, model.linkage.drive_upper_rad],
    }
    if simulation is not None:
        summary["simulation"] = asdict(simulation)
    return summary
