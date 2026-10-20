import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ufactory.simulation.g2_controller import G2Controller
from ufactory.simulation.g2_model import G2Model
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive, gripper_g2_sim_drive_to_gap_m
from ufactory.simulation.g2_model import REFERENCE_GAP_MM
from ufactory.grippers.g2_torch import drive_for_gap_m, pad_gap_m

# Trajectories recorded from the v0.3.1 scalar controller before it was merged
# into the Torch implementation (inputs: joint state and measured gap).
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "g2_controller_reference.json").read_text())
FIELDS = REFERENCE["step_fields"]
OUTPUTS = ("position_rad", "velocity_rad_s", "reference_gap_mm", "integral_torque_nm", "torque_limit_nm")


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("trimesh")
    return G2Model.load()


@pytest.mark.parametrize("dtype,tolerance_mm", [(torch.float64, 1e-9), (torch.float32, 2e-5)])
def test_tensor_geometry_matches_scalar_conversions(model, dtype, tolerance_mm):
    gaps = torch.tensor([0, 10, REFERENCE_GAP_MM, 30, 60, 84], dtype=dtype)
    controller = G2Controller(model, num_envs=len(gaps), dtype=dtype)
    expected = torch.tensor([gripper_g2_gap_m_to_sim_drive(g / 1000) for g in gaps.tolist()], dtype=dtype)
    q = controller.drive_for_gap_mm(gaps)
    torch.testing.assert_close(q, expected, rtol=0, atol=tolerance_mm / 80)
    torch.testing.assert_close(controller.gap_mm(q), gaps, rtol=0, atol=tolerance_mm)
    torch.testing.assert_close(drive_for_gap_m(gaps / 1000), q, rtol=0, atol=0)
    restored = pad_gap_m(q)
    scalar_gaps = torch.tensor([gripper_g2_sim_drive_to_gap_m(float(a)) for a in q], dtype=dtype)
    torch.testing.assert_close(restored, scalar_gaps, rtol=0, atol=tolerance_mm / 1000)
    assert bool((q[:-1] > q[1:]).all())


def test_contact_preflight_preserves_actual_angles_and_object_envelope():
    from unittest.mock import MagicMock

    from ufactory.simulation.g2_adapter import G2ContactCollisionBackend

    backend = MagicMock()
    collision = G2ContactCollisionBackend(backend, object_width_m=0.030)
    for gap in (0, 0.010, 0.02415587200800565, 0.030, 0.060, 0.084):
        drive = gripper_g2_gap_m_to_sim_drive(gap)
        expected = gripper_g2_gap_m_to_sim_drive(max(gap, 0.030))
        for method in ("check", "check_all", "check_all_within_margin"):
            kwargs = {"stage": "grip", "gripper_drive": drive}
            if method == "check_all_within_margin":
                kwargs["margin_m"] = 0.005
            getattr(collision, method)([0] * 6, **kwargs)
            assert getattr(backend, method).call_args.kwargs["gripper_drive"] == pytest.approx(expected, abs=1e-14)
    collision.check([0] * 6, stage="approach")
    assert backend.check.call_args.kwargs["gripper_drive"] == pytest.approx(gripper_g2_gap_m_to_sim_drive(0.084))


def _replay_reference(model, case, dtype, tolerance):
    controller = G2Controller(model, initial_position_mm=case["initial_position_mm"], dtype=dtype)
    if case["reset"] is not None:
        controller.reset(**case["reset"])
    for event in case["commands"]:
        controller.command(*event["command"])
        assert float(controller.requested_speed_mm_s[0]) == pytest.approx(event["requested_speed_mm_s"], abs=1e-9)
        assert int(controller.requested_speed_rpm[0]) == event["requested_speed_rpm"]
        for step in event["steps"]:
            row = dict(zip(FIELDS, step, strict=True))
            output = controller.update(
                row["q"], row["velocity"], REFERENCE["dt"], measured_gap_mm=row["measured_gap_mm"]
            )
            for field in OUTPUTS:
                assert float(getattr(output, field)[0]) == pytest.approx(row[field], abs=tolerance), field
            lookup = controller.force_lookup()
            assert lookup.force_n == pytest.approx(row["force_n"], abs=1e-12)
            assert lookup.calibration_rpm == row["calibration_rpm"]
            assert lookup.outside_calibration == bool(row["outside_calibration"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=[case["name"] for case in REFERENCE["cases"]])
def test_controller_matches_recorded_scalar_reference(model, case):
    _replay_reference(model, case, torch.float64, 1e-10)


@pytest.mark.parametrize("name", ["speed_force_grid", "reverse_mid_stroke"])
def test_float32_controller_tracks_recorded_scalar_reference(model, name):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    _replay_reference(model, case, torch.float32, 2e-4)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 0.0), (torch.float32, 0.0)])
def test_batch_rows_match_independent_single_environment_controllers(model, dtype, tolerance):
    batch = G2Controller(model, num_envs=4, dtype=dtype)
    single = [G2Controller(model, dtype=dtype) for _ in range(4)]
    q = torch.tensor([model.linkage.drive_for_gap_m(g / 1000) for g in (24, 30, 50, 70)], dtype=dtype)
    velocity = torch.tensor([0, 1, -0.5, 0.01], dtype=dtype)
    gaps = [23, 29, 49, 69]
    for position, speeds, forces in [
        ([0, 10, 20, 30], [10, 50, 90, 100], [10, 30, 50, 100]),
        ([0, 10, 20, 30], [0, 0, 0, 0], [10, 30, 80, 100]),
        ([84, 84, 0, 0], [0, 50, 90, 100], [80, 30, 50, 10]),
        ([84, 84, 84, 84], [50, 90, 10, 50], [80, 50, 100, 30]),
    ]:
        batch.command(position, speeds, forces)
        for i, c in enumerate(single):
            c.command(position[i], speeds[i], forces[i])
        for _ in range(12):
            actual = batch.update(q, velocity, measured_gap_mm=gaps)
            expected = [
                c.update(q[i : i + 1], velocity[i : i + 1], measured_gap_mm=gaps[i]) for i, c in enumerate(single)
            ]
            for field in OUTPUTS:
                torch.testing.assert_close(
                    getattr(actual, field), torch.cat([getattr(out, field) for out in expected]), rtol=0, atol=tolerance
                )


def test_reset_is_independent_and_preheld_reset_seeds_torque(model):
    batch = G2Controller(model, num_envs=4)
    batch.command(0, [10, 50, 90, 100], [10, 30, 50, 100])
    batch.update(torch.zeros(4), torch.zeros(4))
    untouched = {
        key: value[[1, 3]].clone()
        for key, value in vars(batch).items()
        if isinstance(value, torch.Tensor) and value.shape == (4,)
    }
    batch.reset([0, 2], position_mm=[30, 40], target_mm=[0, 10], prime=True)
    for key, value in untouched.items():
        torch.testing.assert_close(getattr(batch, key)[[1, 3]], value, rtol=0, atol=0)
    assert torch.equal(batch.integral_torque_nm[[0, 2]], batch.torque_limit_nm[[0, 2]])
    batch.reset([2], position_mm=84)
    assert batch.integral_torque_nm[2] == 0
    assert batch.reference_gap_mm[2] == batch.position_mm[2] == 84


def test_independent_torque_limits_and_atomic_invalid_batch_command(model):
    batch = G2Controller(model, num_envs=4, dtype=torch.float32)
    batch.command(0, [10, 50, 90, 100], [10, 30, 80, 100])
    assert batch.torque_limit_nm.tolist() == pytest.approx(
        [model.torque_limit_nm(batch.force_lookup(i)) for i in range(4)], rel=1e-6
    )
    before = batch.position_mm.clone()
    with pytest.raises(ValueError):
        batch.command([84, 84, 85, 84], 50, 50)
    assert torch.equal(before, batch.position_mm)


def test_num_envs_is_keyword_only(model):
    # A positional second argument used to be the initial position.
    with pytest.raises(TypeError):
        G2Controller(model, 50)


# Servo behaviour on one environment (float64), moved from test_g2_calibrated.py.


def _value(tensor) -> float:
    return float(tensor[0])


def test_speed_above_chart_clamps_force_only(model):
    controller = G2Controller(model)
    controller.command(0, 100, 80)
    assert _value(controller.requested_speed_mm_s) == 225
    assert _value(controller.requested_speed_rpm) == 3765
    lookup = controller.force_lookup()
    assert lookup.calibration_rpm == 3500
    assert lookup.outside_calibration
    assert lookup.force_n == model.curve.lookup(3500, 80).force_n


def test_reference_advances_at_command_speed_without_overshooting_target(model):
    controller = G2Controller(model)
    controller.command(80, 10, 50)
    q = model.linkage.drive_for_gap_m(0.084)
    for _ in range(20):
        previous = _value(controller.reference_gap_mm)
        output = controller.update(q, 0)
        assert 0 <= previous - _value(output.reference_gap_mm) <= 0.450000001
        assert _value(output.reference_gap_mm) >= 80
    assert _value(controller.reference_gap_mm) == 80


def test_pause_holds_preload_and_last_nonzero_calibration_then_resumes(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 90, 80)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.update(q, 0)
    before = c.force_lookup()
    c.command(0, 0, 80)
    paused = c.update(q, 0)
    assert _value(paused.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM)
    assert _value(paused.velocity_rad_s) == 0
    assert _value(paused.integral_torque_nm) == pytest.approx(_value(paused.torque_limit_nm))
    assert c.force_lookup() == before
    assert _value(c.requested_speed_rpm) == 0
    for _ in range(20):
        assert _value(c.update(q, 0).reference_gap_mm) == _value(paused.reference_gap_mm)
    c.command(0, 90, 80)
    resumed = c.update(q, 0)
    assert _value(resumed.reference_gap_mm) < _value(paused.reference_gap_mm)
    assert _value(resumed.velocity_rad_s) > 0


def test_paused_force_change_updates_saturated_holding_effort(model):
    c = G2Controller(model, initial_position_mm=0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 30)
    c.update(q, 0)
    c.command(0, 0, 30)
    before = c.update(q, 0)
    c.command(0, 0, 80)
    after = c.update(q, 0)
    assert _value(after.torque_limit_nm) > _value(before.torque_limit_nm)
    assert _value(after.integral_torque_nm) == pytest.approx(_value(after.torque_limit_nm))
    # More compression changes the joint posture. The old position reference
    # must not unwind the new holding effort while feed is paused.
    compressed_q = q + 0.005
    for _ in range(500):
        held = c.update(compressed_q, 0)
    assert _value(held.integral_torque_nm) == pytest.approx(_value(held.torque_limit_nm))
    assert c.kp * (_value(held.position_rad) - compressed_q) == pytest.approx(_value(held.torque_limit_nm))


def test_paused_new_position_keeps_grasp_until_nonzero_speed(model):
    c = G2Controller(model, initial_position_mm=0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(84, 0, 80)
    paused = c.update(q, 0)
    assert _value(paused.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM)
    assert _value(paused.integral_torque_nm) == pytest.approx(_value(paused.torque_limit_nm))
    c.command(84, 50, 80)
    opened = c.update(q, 0)
    assert _value(opened.reference_gap_mm) > REFERENCE_GAP_MM
    assert _value(opened.velocity_rad_s) < 0
    assert _value(opened.integral_torque_nm) <= 0


def test_moving_pause_uses_actual_pad_position_but_stopped_pause_keeps_joint_posture(model):
    c = G2Controller(model, initial_position_mm=50)
    q = model.linkage.drive_for_gap_m(0.050)
    c.command(0, 0, 80)
    assert _value(c.update(q, 1.0, measured_gap_mm=47.0).reference_gap_mm) == 47.0
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(0, 0, 80)
    assert _value(c.update(q, 0, measured_gap_mm=47.0).reference_gap_mm) == pytest.approx(50.0)


def test_conditional_integration_stops_windup_and_allows_unwinding(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 10)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    for _ in range(500):
        c.update(q, 0)
    assert _value(c.integral_torque_nm) == 0
    c = G2Controller(model, initial_position_mm=30)
    limit = model.torque_limit_nm(c.force_lookup())
    c.integral_torque_nm.fill_(limit)
    narrower_q = model.linkage.drive_for_gap_m(0.029)
    c.update(narrower_q, -2)
    assert 0 < _value(c.integral_torque_nm) < limit


def test_reversal_clears_integral_and_rebases_to_observed_position(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 80)
    c.integral_torque_nm.fill_(model.torque_limit_nm(c.force_lookup()))
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(84, 50, 80)
    output = c.update(q, 0)
    assert _value(output.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM + 2.25)
    assert _value(output.integral_torque_nm) <= 0
    assert _value(output.velocity_rad_s) < 0


@pytest.mark.parametrize(
    "command", [(-1, 50, 50), (85, 50, 50), (0, -1, 50), (0, 101, 50), (0, 50, 0), (0, 50, 101), (0, math.nan, 50)]
)
def test_invalid_command_is_atomic(model, command):
    c = G2Controller(model)
    before = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in vars(c).items()}
    with pytest.raises(ValueError):
        c.command(*command)
    for key, value in vars(c).items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key
        else:
            assert value == before[key], key
