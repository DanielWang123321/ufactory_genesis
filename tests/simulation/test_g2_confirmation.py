from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from ufactory.simulation.g2_model import G2Model
from ufactory.trajectory.g2_confirmation import (
    GripForceGate,
    force_at_target,
    grip_timeout_s,
    prepare_g2_program,
    summarize,
)
from ufactory.config.models import G2SimulationConfig


def build_program(config, speed_percent=50, position_mm=0):
    pytest.importorskip("genesis")
    from ufactory.cli.pick_place import _build_program

    return prepare_g2_program(_build_program(config), G2SimulationConfig(position_mm, speed_percent, 50))


from ufactory.trajectory.g2_confirmation import ReleaseGate, release_at_target, release_timeout_s
from ufactory.config import load_runtime_config


@pytest.fixture(scope="module")
def config():
    return load_runtime_config("xarm6")


def test_arm_path_matches_source_example(config):
    pytest.importorskip("genesis")
    from ufactory.cli.pick_place import _build_program

    source = _build_program(config)
    replica = build_program(config)
    assert [seg.label for seg in replica.segments] == [
        seg.label for seg in source.segments if seg.label != "grip-settle"
    ]
    source_by_label = {seg.label: seg for seg in source.segments}
    for copied in replica.segments:
        original = source_by_label[copied.label]
        if original.kind != "gripper":
            np.testing.assert_array_equal(copied.samples(50)[0], original.samples(50)[0])
            assert copied.duration == original.duration
    assert next(seg for seg in replica.segments if seg.label == "grip").gap_end == 0
    assert next(seg for seg in replica.segments if seg.label == "release").gap_end == 0.084


def test_gripper_phases_use_deadlines_and_keep_arm_path(config):
    slow = build_program(config, speed_percent=10)
    fast = build_program(config, speed_percent=90)
    close = next(seg for seg in slow.segments if seg.label == "grip")
    assert close.duration == pytest.approx(84 / 22.5 + 1, abs=0.02)
    assert next(seg for seg in fast.segments if seg.label == "grip").duration == 2
    assert next(seg for seg in slow.segments if seg.label == "release").duration == pytest.approx(
        84 / 22.5 + 1, abs=0.02
    )
    assert next(seg for seg in fast.segments if seg.label == "release").duration == 2
    assert "grip-settle" not in [seg.label for seg in slow.segments]
    for original, copied in zip(fast.segments, slow.segments, strict=True):
        if original.kind != "gripper":
            np.testing.assert_array_equal(copied.samples(50)[0], original.samples(50)[0])


def successful_observations():
    rows = []
    for phase, length in (
        ("grip", 2),
        ("lift", 20),
        ("transit", 20),
        ("release", 2),
        ("retreat", 20),
        ("return-home", 20),
    ):
        held = phase in {"grip", "lift", "transit"}
        for _ in range(length):
            rows.append(
                {
                    "time_s": (len(rows) + 1) / 50,
                    "phase": phase,
                    "object_base_x_mm": 300.0,
                    "object_base_y_mm": 300.0 if phase != "grip" else 0.0,
                    "object_base_z_mm": 115.0 if phase in {"lift", "transit"} else 15.0,
                    "tool_base_x_mm": 300.0,
                    "tool_base_y_mm": 0.0,
                    "tool_base_z_mm": 250.0,
                    "object_in_tool_x_mm": 0.0,
                    "object_in_tool_y_mm": 0.0,
                    "object_in_tool_z_mm": -180.0,
                    "object_in_pad_x_mm": 0.0,
                    "object_in_pad_y_mm": 0.0,
                    "object_in_pad_z_mm": 0.0,
                    "sensor_force_n": 37.8 if held else 0.0,
                    "left_normal_n": 37.8 if held else 0.0,
                    "right_normal_n": 37.8 if held else 0.0,
                    "expected_static_force_n": 37.8,
                    "drive_torque_nm": 4.0,
                    "torque_limit_nm": 4.1,
                    "motor_torque_estimate_nm": 0.29,
                    "gap_mm": 30.0 if held else 84.0,
                    "home_error_mm": 0.5,
                    "object_speed_mm_s": 0.0,
                }
            )
    return rows


def test_success_requires_observed_lift_transport_and_release(config):
    result = summarize(successful_observations(), config)
    assert result["passed"]
    assert result["place_error_mm"] == 0
    assert result["carry_max_slip_mm"] == 0


@pytest.mark.parametrize(
    "phase,key,value,failed",
    [
        ("lift", "object_base_z_mm", 15, "lifted_50mm"),
        ("transit", "object_base_z_mm", 15, "transit_clearance_20mm"),
        ("transit", "right_normal_n", 0, "transit_bilateral_95percent"),
        ("transit", "object_in_pad_z_mm", -6, "carry_slip_within_5mm"),
        ("return-home", "object_base_x_mm", 330, "placed_within_10mm"),
        ("return-home", "left_normal_n", 1, "released"),
        ("transit", "drive_torque_nm", 4.2, "torque_within_limit"),
    ],
)
def test_metrics_reject_false_success(config, phase, key, value, failed):
    rows = deepcopy(successful_observations())
    for row in rows:
        if row["phase"] == phase:
            row[key] = value
    result = summarize(rows, config)
    assert not result["passed"]
    assert failed in result["failed_checks"]


def test_slip_is_relative_to_actual_pads_rather_than_flange(config):
    rows = successful_observations()
    for row in rows:
        if row["phase"] == "transit":
            row["object_in_tool_z_mm"] -= 10
    result = summarize(rows, config)
    assert result["passed"]
    assert result["carry_max_slip_mm"] == 0
    assert result["carry_max_tool_displacement_mm"] == 10


@pytest.mark.parametrize("offset,original_passed", [(30, True), (60, False)])
def test_original_position_thresholds_are_reported_alongside_dev_quality(config, offset, original_passed):
    rows = successful_observations()
    rows[-1]["object_base_x_mm"] += offset
    result = summarize(rows, config)
    assert not result["passed"]
    assert result["source_example_thresholds_passed"] is original_passed


def test_packaging_omits_post_release_placement_distance_gate(config):
    rows = [
        {**row, "phase": "post-release-settle" if row["phase"] == "retreat" else row["phase"]}
        for row in successful_observations()
    ]
    rows[-1]["object_base_x_mm"] += 40
    result = summarize(rows, config, packaging=True)
    assert "placed_within_10mm" not in result["transport_checks"]
    assert result["transport_passed"]
    assert result["passed"]
    assert result["source_example_thresholds_passed"]


def force_sample(tick, left=37.8, right=37.8, target=37.8):
    return {"time_s": tick / 50, "left_normal_n": left, "right_normal_n": right, "expected_static_force_n": target}


def test_force_confirmation_requires_two_frames():
    gate = GripForceGate()
    assert not gate.observe(force_sample(1))
    assert gate.consecutive_frames == 1
    assert gate.observe(force_sample(2))
    assert gate.confirmed_time_s == 0.04


def test_single_frame_impact_does_not_trigger():
    gate = GripForceGate()
    assert not gate.observe(force_sample(1))
    assert not gate.observe(force_sample(2, left=60, right=60))
    assert gate.consecutive_frames == 0
    assert gate.confirmed_time_s is None


def test_interruption_restarts_confirmation_count():
    gate = GripForceGate()
    assert not gate.observe(force_sample(1))
    assert not gate.observe(force_sample(2, left=0, right=0))
    assert not gate.observe(force_sample(3))
    assert gate.consecutive_frames == 1
    assert gate.observe(force_sample(4))
    assert gate.confirmed_time_s == 0.08


def test_missing_control_sample_restarts_confirmation_count():
    gate = GripForceGate()
    assert not gate.observe(force_sample(1))
    assert not gate.observe(force_sample(3))
    assert gate.observe(force_sample(4))


@pytest.mark.parametrize("left,right", [(0, 75.6), (75.6, 0), (0.2, 75.4), (75.4, 0.2)])
def test_single_sided_contact_cannot_confirm_even_if_average_is_on_target(left, right):
    gate = GripForceGate()
    for tick in range(1, 5):
        assert not gate.observe(force_sample(tick, left, right))


@pytest.mark.parametrize("ratio,expected", [(0.949, False), (0.95, True), (1, True), (1.05, True), (1.051, False)])
def test_force_band_is_relative_five_percent(ratio, expected):
    assert force_at_target(force_sample(1, left=2 * ratio, right=2 * ratio, target=2)) is expected


@pytest.mark.parametrize("speed,force,gap", [(10, 10, 10), (50, 50, 30), (90, 100, 70)])
def test_confirmation_uses_speed_force_and_actual_gap_model(speed, force, gap):
    pytest.importorskip("trimesh")
    pytest.importorskip("torch")
    from ufactory.simulation.g2_controller import G2Controller

    model = G2Model.load()
    controller = G2Controller(model)
    controller.command(0, speed, force)
    target = model.static_force_n(controller.force_lookup(), gap)
    gate = GripForceGate()
    assert not gate.observe(force_sample(1, target, target, target))
    assert gate.observe(force_sample(2, target, target, target))
    assert not force_at_target(force_sample(3, target * 1.06, target * 1.06, target))


def test_timeout_uses_current_reference_travel():
    assert grip_timeout_s(84, 0, 10) == pytest.approx(84 / 22.5 + 1)
    assert grip_timeout_s(40, 20, 10) == 2
    assert grip_timeout_s(84, 0, 50) == 2


def test_metrics_reject_delayed_lift_and_fixed_wait(config):
    rows = successful_observations()
    rows.insert(2, dict(rows[1]))
    for tick, row in enumerate(rows, start=1):
        row["time_s"] = tick / 50
    result = summarize(rows, config)
    assert not result["timing_passed"]
    assert not result["checks"]["lift_next_control_tick"]
    assert not result["checks"]["no_fixed_grip_wait"]
    assert result["transport_passed"]


def test_metrics_use_both_confirmation_frames_for_slip_baseline(config):
    rows = successful_observations()
    rows[0]["object_in_pad_z_mm"] = 2
    rows[1]["object_in_pad_z_mm"] = 4
    result = summarize(rows, config)
    assert result["carry_max_slip_mm"] == 3
    assert result["grip_force_confirmation_time_s"] == 0.04
    assert result["lift_start_time_s"] == 0.06
    assert result["force_to_lift_delay_s"] == pytest.approx(0.02)


def test_metrics_reject_lift_without_force_confirmation(config):
    rows = successful_observations()
    rows[1]["left_normal_n"] = 0
    result = summarize(rows, config)
    assert not result["timing_passed"]
    assert not result["checks"]["unconfirmed_grip_prevents_lift"]
    assert result["grip_confirmation_samples"] == []


def release_sample(tick, gap=84, left=0, right=0):
    return {"time_s": tick / 50, "gap_mm": gap, "left_normal_n": left, "right_normal_n": right}


def test_release_confirmation_requires_two_frames():
    gate = ReleaseGate()
    assert not gate.observe(release_sample(1))
    assert gate.consecutive_frames == 1
    assert gate.observe(release_sample(2))
    assert gate.confirmed_time_s == 0.04


@pytest.mark.parametrize(
    "gap,left,right,expected",
    [
        (83, 0, 0, True),
        (85, 0, 0, True),
        (82.99, 0, 0, False),
        (85.01, 0, 0, False),
        (84, 0.2, 0.2, True),
        (84, 0.21, 0, False),
        (84, 0, 0.21, False),
        (30, 0, 0, False),
        (84, -1, 0, False),
        (float("nan"), 0, 0, False),
        (84, float("nan"), 0, False),
        (84, 0, float("inf"), False),
    ],
)
def test_release_requires_opening_and_both_contacts_clear(gap, left, right, expected):
    assert release_at_target(release_sample(1, gap, left, right)) is expected


@pytest.mark.parametrize("gap,left,right", [(82, 0, 0), (84, 0.3, 0), (84, 0, 0.3)])
def test_single_open_frame_and_interruption_do_not_confirm_release(gap, left, right):
    gate = ReleaseGate()
    assert not gate.observe(release_sample(1))
    assert not gate.observe(release_sample(2, gap, left, right))
    assert gate.consecutive_frames == 0
    assert not gate.observe(release_sample(3))
    assert gate.observe(release_sample(4))
    assert gate.confirmed_time_s == 0.08


def test_release_missing_sample_restarts_count():
    gate = ReleaseGate()
    assert not gate.observe(release_sample(1))
    assert not gate.observe(release_sample(3))
    assert gate.observe(release_sample(4))


def test_release_timeout_allows_position_reference_to_finish_feed():
    assert release_timeout_s(0, 10) == pytest.approx(84 / 22.5 + 1)
    assert release_timeout_s(30, 10) == pytest.approx(54 / 22.5 + 1)
    assert release_timeout_s(0, 50) == 2


def test_metrics_reject_extra_release_wait(config):
    rows = successful_observations()
    index = next(i for i, row in enumerate(rows) if row["phase"] == "retreat")
    rows.insert(index, dict(rows[index - 1]))
    for tick, row in enumerate(rows, start=1):
        row["time_s"] = tick / 50
    result = summarize(rows, config)
    assert not result["timing_passed"]
    assert not result["checks"]["retreat_next_control_tick"]
    assert not result["checks"]["no_fixed_release_wait"]
    assert result["transport_passed"]


@pytest.mark.parametrize("key,value", [("gap_mm", 82), ("left_normal_n", 0.3), ("right_normal_n", 0.3)])
def test_metrics_reject_retreat_without_release_confirmation(config, key, value):
    rows = successful_observations()
    for row in rows:
        if row["phase"] == "release":
            row[key] = value
            row["release_confirmed"] = True
    result = summarize(rows, config)
    assert not result["passed"]
    assert not result["checks"]["release_confirmed"]
    assert not result["checks"]["unconfirmed_release_prevents_retreat"]
    assert result["release_confirmation_samples"] == []


@pytest.mark.parametrize("fault", [None, "grip", "release"])
def test_confirmation_controls_next_tick_and_timeout_saves_failure(fault, monkeypatch, tmp_path):
    from types import SimpleNamespace
    import json

    torch = pytest.importorskip("torch")
    import ufactory.trajectory.g2_replay as replay
    from ufactory.trajectory.g2_replay import replay_g2
    from ufactory.trajectory.segments import Program, Segment

    state = SimpleNamespace(command=84, phase_ticks={})
    position = torch.zeros(1, 3)
    robot = SimpleNamespace(
        get_dofs_position=lambda idx: torch.zeros(1, len(idx)), control_dofs_position=lambda *_: None
    )
    obj = SimpleNamespace(get_pos=lambda: position)
    adapter = SimpleNamespace(
        controller=SimpleNamespace(reference_gap_mm=torch.tensor([84.0], dtype=torch.float64)),
        reset=lambda *_: None,
        gap_mm=lambda: 84,
        apply=lambda *_: None,
        command=lambda p, *_: setattr(state, "command", p),
    )
    ctx = SimpleNamespace(
        robot=robot,
        scene=SimpleNamespace(step=lambda: None),
        obj=obj,
        ik_link=SimpleNamespace(get_pos=lambda: position),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
        home_pos_base=(0, 0, 0),
        obj_pos_base=(0, 0, 0),
        place_pos_base=(0, 0, 0),
        obj_size=(0.03, 0.03, 0.03),
        base_to_world=lambda p: p,
        g2_settings=G2SimulationConfig(),
        g2_controller=adapter,
    )

    def record(ctx, adapter, output, phase, time_s, previous, q_arm, dt):
        state.phase_ticks[phase] = state.phase_ticks.get(phase, 0) + 1
        closed = state.command == 0
        gap = 30 if closed else 84
        force = 10 if closed else 0
        if fault == "grip":
            force = 0
        if fault == "release" and phase == "release":
            gap, force = 30, 10
        row = dict(
            time_s=time_s,
            phase=phase,
            gap_mm=gap,
            left_normal_n=force,
            right_normal_n=force,
            expected_static_force_n=10,
            sensor_force_n=force,
            drive_torque_nm=1,
            torque_limit_nm=2,
            motor_torque_estimate_nm=0.1,
            object_speed_mm_s=0,
            home_error_mm=0,
        )
        for prefix in ("object_base", "object_in_pad", "object_in_tool"):
            row.update({f"{prefix}_{axis}_mm": 0 for axis in "xyz"})
        return row

    monkeypatch.setattr(replay, "_record", record)
    monkeypatch.setattr("ufactory.trajectory.sim_executor._assert_finite_replay_state", lambda *_a, **_kw: None)
    segments = [
        Segment("gripper", 2, 0, 0, label="grip", gap_start=0.084, gap_end=0),
        Segment("movej", 0.02, 1, 1, label="lift", q_start=np.zeros(1), q_end=np.zeros(1)),
        Segment("gripper", 2, 0, 0, label="release", gap_start=0, gap_end=0.084),
        Segment("movej", 0.02, 1, 1, label="retreat", q_start=np.zeros(1), q_end=np.zeros(1)),
    ]
    path = tmp_path / "result.json"
    report = replay_g2(
        Program(rate=50, segments=segments),
        ctx,
        final_metric_settle_s=0,
        csv_path=path.with_suffix(".csv"),
        sim_report_path=path,
    )
    saved = json.loads(path.read_text())
    if fault:
        assert not report.completed and report.timeout_phase == fault
        assert report.records[-1]["phase"] == fault
        assert state.phase_ticks[fault] == 100
        assert "retreat" not in state.phase_ticks
        if fault == "grip":
            assert "lift" not in state.phase_ticks
        assert not saved["completed"]
    else:
        assert report.completed
        assert state.phase_ticks["grip"] == state.phase_ticks["release"] == 2
        assert report.confirmations["grip"]["next_action_delay_s"] == 0.02
        assert report.confirmations["release"]["next_action_delay_s"] == 0.02
    assert path.with_suffix(".csv").read_text().count("\n") == len(report.records) + 1
