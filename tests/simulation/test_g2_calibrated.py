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
from ufactory.simulation.g2_controller import G2Controller
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


def test_speed_above_chart_clamps_force_only(model):
    controller = G2Controller(model)
    controller.command(0, 100, 80)
    assert controller.requested_speed_mm_s == 225
    assert controller.requested_speed_rpm == 3765
    lookup = controller.force_lookup
    assert lookup.calibration_rpm == 3500
    assert lookup.outside_calibration
    assert lookup.force_n == model.curve.lookup(3500, 80).force_n


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


def test_reference_advances_at_command_speed_without_overshooting_target(model):
    controller = G2Controller(model)
    controller.command(80, 10, 50)
    q = model.linkage.drive_for_gap_m(0.084)
    for i in range(20):
        previous = controller.reference_gap_mm
        output = controller.update(q, 0)
        assert 0 <= previous - output.reference_gap_mm <= 0.450000001
        assert output.reference_gap_mm >= 80
    assert controller.reference_gap_mm == 80


def test_pause_holds_preload_and_last_nonzero_calibration_then_resumes(model):
    c = G2Controller(model, 0)
    c.command(0, 90, 80)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.update(q, 0)
    before = c.force_lookup
    c.command(0, 0, 80)
    paused = c.update(q, 0)
    assert paused.reference_gap_mm == pytest.approx(REFERENCE_GAP_MM)
    assert paused.velocity_rad_s == 0
    assert paused.integral_torque_nm == pytest.approx(paused.torque_limit_nm)
    assert c.force_lookup == before
    assert c.requested_speed_rpm == 0
    for _ in range(20):
        assert c.update(q, 0).reference_gap_mm == paused.reference_gap_mm
    c.command(0, 90, 80)
    resumed = c.update(q, 0)
    assert resumed.reference_gap_mm < paused.reference_gap_mm
    assert resumed.velocity_rad_s > 0


def test_paused_force_change_updates_saturated_holding_effort(model):
    c = G2Controller(model, 0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 30)
    c.update(q, 0)
    c.command(0, 0, 30)
    before = c.update(q, 0)
    c.command(0, 0, 80)
    after = c.update(q, 0)
    assert after.torque_limit_nm > before.torque_limit_nm
    assert after.integral_torque_nm == pytest.approx(after.torque_limit_nm)
    # More compression changes the joint posture. The old position reference
    # must not unwind the new holding effort while feed is paused.
    compressed_q = q + 0.005
    for _ in range(500):
        held = c.update(compressed_q, 0)
    assert held.integral_torque_nm == pytest.approx(held.torque_limit_nm)
    assert c.kp * (held.position_rad - compressed_q) == pytest.approx(held.torque_limit_nm)


def test_paused_new_position_keeps_grasp_until_nonzero_speed(model):
    c = G2Controller(model, 0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(84, 0, 80)
    paused = c.update(q, 0)
    assert paused.reference_gap_mm == pytest.approx(REFERENCE_GAP_MM)
    assert paused.integral_torque_nm == pytest.approx(paused.torque_limit_nm)
    c.command(84, 50, 80)
    opened = c.update(q, 0)
    assert opened.reference_gap_mm > REFERENCE_GAP_MM
    assert opened.velocity_rad_s < 0
    assert opened.integral_torque_nm <= 0


def test_moving_pause_uses_actual_pad_position_but_stopped_pause_keeps_joint_posture(model):
    c = G2Controller(model, 50)
    q = model.linkage.drive_for_gap_m(0.050)
    c.command(0, 0, 80)
    assert c.update(q, 1.0, measured_gap_mm=47.0).reference_gap_mm == 47.0
    c = G2Controller(model, 0)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(0, 0, 80)
    assert c.update(q, 0, measured_gap_mm=47.0).reference_gap_mm == pytest.approx(50.0)


def test_conditional_integration_stops_windup_and_allows_unwinding(model):
    c = G2Controller(model, 0)
    c.command(0, 50, 10)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    for _ in range(500):
        c.update(q, 0)
    assert c.integral_torque_nm == 0
    c = G2Controller(model, 30)
    limit = model.torque_limit_nm(c.force_lookup)
    c.integral_torque_nm = limit
    narrower_q = model.linkage.drive_for_gap_m(0.029)
    c.update(narrower_q, -2)
    assert 0 < c.integral_torque_nm < limit


def test_reversal_clears_integral_and_rebases_to_observed_position(model):
    c = G2Controller(model, 0)
    c.command(0, 50, 80)
    c.integral_torque_nm = model.torque_limit_nm(c.force_lookup)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(84, 50, 80)
    output = c.update(q, 0)
    assert output.reference_gap_mm == pytest.approx(REFERENCE_GAP_MM + 2.25)
    assert output.integral_torque_nm <= 0
    assert output.velocity_rad_s < 0


@pytest.mark.parametrize(
    "command", [(-1, 50, 50), (85, 50, 50), (0, -1, 50), (0, 101, 50), (0, 50, 0), (0, 50, 101), (0, math.nan, 50)]
)
def test_invalid_command_is_atomic(model, command):
    c = G2Controller(model)
    before = vars(c).copy()
    with pytest.raises(ValueError):
        c.command(*command)
    assert vars(c) == before
