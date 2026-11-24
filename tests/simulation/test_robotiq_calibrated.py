"""Unit tests for Robotiq 2F-85 and 2F-140 calibrated linkage models and force curves."""

from __future__ import annotations

from dataclasses import asdict
import math

import pytest

from ufactory.config.models import ROBOTIQ_2F_85_PHYSICS_PROFILE, ROBOTIQ_2F_140_PHYSICS_PROFILE
from ufactory.grippers.robotiq import RobotiqLinkageGeometry, get_robotiq_spec
from ufactory.robots.paths import robotiq_2f_85_movable_visual_urdf, robotiq_2f_140_movable_visual_urdf
from ufactory.simulation.robotiq_model import RobotiqForceCurve, RobotiqModel, robotiq_physics_summary


@pytest.mark.parametrize(
    ("variant", "open_mm", "f_min", "f_max", "profile_name"),
    [
        ("robotiq_2f_85", 85.0, 20.0, 235.0, ROBOTIQ_2F_85_PHYSICS_PROFILE),
        ("robotiq_2f_140", 140.0, 10.0, 125.0, ROBOTIQ_2F_140_PHYSICS_PROFILE),
    ],
)
def test_robotiq_linkage_and_model_roundtrip(
    variant: str,
    open_mm: float,
    f_min: float,
    f_max: float,
    profile_name: str,
) -> None:
    spec = get_robotiq_spec(variant)
    urdf = robotiq_2f_85_movable_visual_urdf() if variant == "robotiq_2f_85" else robotiq_2f_140_movable_visual_urdf()
    from_urdf = RobotiqLinkageGeometry.from_urdf(urdf, prefix=spec.prefix)
    assert asdict(from_urdf) == pytest.approx(asdict(spec.linkage), abs=1e-6)

    for gap_mm in (0.0, 22.0, open_mm * 0.5, open_mm):
        q = spec.gap_m_to_sim_drive(gap_mm / 1000.0)
        restored_mm = spec.sim_drive_to_gap_m(q) * 1000.0
        assert math.isclose(restored_mm, gap_mm, abs_tol=1e-2)

    assert spec.gap_m_to_register(open_mm / 1000.0) == 0
    assert spec.gap_m_to_register(0.0) == 255
    assert math.isclose(spec.register_to_gap_m(0) * 1000.0, open_mm, abs_tol=1e-6)
    assert math.isclose(spec.register_to_gap_m(255) * 1000.0, 0.0, abs_tol=1e-6)

    model = RobotiqModel.load(variant)
    assert model.curve.lookup(50.0, 1.0).force_n == pytest.approx(f_min)
    assert model.curve.lookup(50.0, 100.0).force_n == pytest.approx(f_max)
    lookup_mid = model.curve.lookup(50.0, 50.0)
    assert model.torque_limit_nm(lookup_mid) > 0.0
    assert model.static_force_n(lookup_mid, 30.0) > 0.0
    assert math.isfinite(model.motor_torque_estimate_nm(1.0))

    summary = robotiq_physics_summary(variant)
    assert summary["physics_profile"] == profile_name
    assert summary["open_gap_mm"] == open_mm
    assert summary["force_range_n"] == [f_min, f_max]


def test_robotiq_force_curve_validation() -> None:
    curve = RobotiqForceCurve(20.0, 235.0)
    with pytest.raises(ValueError):
        curve.lookup(50.0, 0.0)
    with pytest.raises(ValueError):
        curve.lookup(-1.0, 50.0)
