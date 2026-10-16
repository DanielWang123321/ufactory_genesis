from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from ufactory.grippers.g2 import (
    G2_LINKAGE_GEOMETRY,
    GRIPPER_G2_SIM_OPEN_DRIVE,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    gripper_g2_gap_m_to_sim_drive,
    gripper_g2_sim_drive_to_gap_m,
)
from ufactory.simulation.g2_model import (
    G2Model,
    REFERENCE_GAP_MM,
    REFERENCE_RAW_POSITION,
    sdk_position_counts,
    sdk_position_mm,
    sdk_speed_rpm,
    speed_mm_s,
)


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("trimesh")
    return G2Model.load()


@pytest.mark.parametrize(
    "percent,mm_s,rpm",
    [(0, 0, 0), (1, 15, 577), (5, 15, 577), (10, 22.5, 691), (50, 112.5, 2057), (90, 202.5, 3423), (100, 225, 3765)],
)
def test_percent_speed_and_sdk_register_truncation(percent, mm_s, rpm):
    assert speed_mm_s(percent) == mm_s
    assert sdk_speed_rpm(mm_s) == rpm


def test_reference_retains_sdk_raw_count_and_inverse():
    assert REFERENCE_RAW_POSITION == 230
    assert REFERENCE_GAP_MM == pytest.approx(24.15587200800565, abs=1e-12)
    assert sdk_position_counts(0) == 0
    assert sdk_position_counts(84) == 850


@pytest.mark.parametrize("position", [0, 10, 24.155872, 30, 60, 84])
def test_sdk_position_quantization_is_at_most_one_count(position):
    raw = sdk_position_counts(position)
    restored = sdk_position_mm(raw)
    assert abs(restored - position) < 0.106


@pytest.mark.parametrize(
    "function,value",
    [
        (speed_mm_s, -1),
        (speed_mm_s, 101),
        (speed_mm_s, math.nan),
        (sdk_speed_rpm, 226),
        (sdk_position_counts, -0.1),
        (sdk_position_counts, 84.1),
        (sdk_position_counts, math.inf),
    ],
)
def test_invalid_units_are_rejected(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_curve_retains_plateau_and_sharp_force_change(model):
    assert model.curve.lookup(3000, 1).force_n == model.curve.lookup(3000, 10).force_n
    assert model.curve.lookup(0, 55).force_n - model.curve.lookup(0, 50).force_n > 8
    assert model.curve.maximum_force_n == 63.79


def test_piecewise_current_and_speed_interpolation(model):
    a = model.curve.lookup(3000, 45).force_n
    b = model.curve.lookup(3000, 50).force_n
    assert model.curve.lookup(3000, 47.5).force_n == pytest.approx((a + b) / 2)
    at_3250 = model.curve.lookup(3250, 47.5).force_n
    assert model.curve.lookup(3125, 47.5).force_n == pytest.approx((a + b + 2 * at_3250) / 4)


@pytest.mark.parametrize("gap_mm", [0, 10, REFERENCE_GAP_MM, 30, 60, 84])
def test_physical_endpoints_inverse_and_gap_jacobian(model, gap_mm):
    linkage = model.linkage
    q = linkage.drive_for_gap_m(gap_mm / 1000)
    assert linkage.pad_gap_m(q) == pytest.approx(gap_mm / 1000, abs=1e-12)
    step = 1e-6
    numerical = (linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)) / (2 * step)
    assert linkage.gap_jacobian_m_per_rad(q) == pytest.approx(numerical, rel=1e-8)


def test_public_geometry_matches_the_urdf_collision_pads(model):
    assert asdict(model.linkage) == asdict(G2_LINKAGE_GEOMETRY)
    assert model.linkage.pad_gap_m(GRIPPER_G2_SIM_OPEN_DRIVE) * 1000 == pytest.approx(84, abs=1e-6)
    assert model.linkage.pad_gap_m(GRIPPER_G2_SIM_CLOSE_DRIVE) * 1000 == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("gap_mm", [0, 10, REFERENCE_GAP_MM, 30, 60, 84])
def test_public_scalar_conversion_matches_pad_geometry(gap_mm):
    q = gripper_g2_gap_m_to_sim_drive(gap_mm / 1000)
    assert G2_LINKAGE_GEOMETRY.pad_gap_m(q) * 1000 == pytest.approx(gap_mm, abs=1e-6)
    assert gripper_g2_sim_drive_to_gap_m(q) * 1000 == pytest.approx(gap_mm, abs=1e-6)


def test_scalar_conversion_is_monotonic_and_clamps_to_nominal_stroke():
    gaps = (0, 10, REFERENCE_GAP_MM, 30, 60, 84)
    angles = [gripper_g2_gap_m_to_sim_drive(gap / 1000) for gap in gaps]
    assert all(a > b for a, b in zip(angles, angles[1:]))
    assert gripper_g2_gap_m_to_sim_drive(-0.01) == pytest.approx(GRIPPER_G2_SIM_CLOSE_DRIVE)
    assert gripper_g2_gap_m_to_sim_drive(0.1) == pytest.approx(GRIPPER_G2_SIM_OPEN_DRIVE)
    assert gripper_g2_sim_drive_to_gap_m(0) == pytest.approx(0.084)
    assert gripper_g2_sim_drive_to_gap_m(1) == pytest.approx(0)


@pytest.mark.parametrize("speed,force", [(10, 10), (50, 30), (50, 50), (90, 100)])
def test_two_pad_virtual_work_at_reference_gap(model, speed, force):
    lookup = model.curve.lookup(sdk_speed_rpm(speed_mm_s(speed)), force)
    linkage = model.linkage
    q = linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    step = 1e-6
    full_gap_travel = linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)
    # Each pad travels half of the full gap change; the sensor reads the load at either pad.
    two_pad_work = 2 * lookup.force_n * full_gap_travel / 2
    joint_work = model.torque_limit_nm(lookup) * 2 * step
    assert joint_work == pytest.approx(two_pad_work, rel=1e-8)
    assert model.torque_limit_nm(lookup) == pytest.approx(lookup.force_n * model.reference_jacobian_m_rad, rel=1e-12)


@pytest.mark.parametrize("rpm,force,wide_gap_mm,narrow_gap_mm", [(2057, 50, 80, 20), (3500, 100, 70, 10)])
def test_fixed_torque_changes_force_with_gap(model, rpm, force, wide_gap_mm, narrow_gap_mm):
    lookup = model.curve.lookup(rpm, force)
    assert model.static_force_n(lookup, wide_gap_mm) > model.static_force_n(lookup, narrow_gap_mm)


def test_rated_torque_does_not_clip_curve(model):
    lookup = model.curve.lookup(3500, 100)
    tau = model.torque_limit_nm(lookup)
    assert model.rated_motor_torque_nm == 0.22
    assert model.motor_torque_estimate_nm(tau) > model.rated_motor_torque_nm
    assert model.motor_torque_estimate_nm(tau) <= model.peak_motor_torque_nm
    maximum = model.curve.lookup(0, 100)
    assert model.motor_torque_estimate_nm(model.torque_limit_nm(maximum)) == pytest.approx(0.5)
