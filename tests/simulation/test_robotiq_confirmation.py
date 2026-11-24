"""Unit tests for Robotiq 2F-85 and 2F-140 state confirmation, deadlines and acceptance summary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ufactory.config.models import RobotiqSimulationConfig
from ufactory.trajectory.g2_confirmation import (
    GripForceGate,
    ReleaseGate,
    prepare_robotiq_program,
    robotiq_grip_timeout_s,
    robotiq_release_timeout_s,
    summarize,
)
from ufactory.trajectory.segments import Program, Segment


@pytest.mark.parametrize(("variant", "open_mm"), [("robotiq_2f_85", 85.0), ("robotiq_2f_140", 140.0)])
def test_prepare_robotiq_program_and_deadlines(variant: str, open_mm: float) -> None:
    program = Program(
        rate=50.0,
        segments=[
            Segment("gripper", 1.0, 0.0, 0.0, gap_start=open_mm / 1000.0, gap_end=0.022, label="grip"),
            Segment("gripper", 0.5, 0.0, 0.0, gap_start=0.022, gap_end=0.022, label="grip-settle"),
            Segment("gripper", 0.2, 0.0, 0.0, gap_start=0.022, gap_end=0.029, label="pre-release-relax"),
            Segment("gripper", 0.5, 0.0, 0.0, gap_start=0.022, gap_end=0.022, label="pre-release-settle"),
            Segment("gripper", 1.0, 0.0, 0.0, gap_start=0.022, gap_end=open_mm / 1000.0, label="release"),
        ],
    )
    settings = RobotiqSimulationConfig(position_mm=18.0, speed_percent=40.0, force_percent=60.0)
    prepared = prepare_robotiq_program(program, settings, variant)
    labels = [s.label for s in prepared.segments]
    assert labels == ["grip", "pre-release-settle", "release"]
    assert prepared.segments[0].gap_start == pytest.approx(open_mm / 1000.0)
    assert prepared.segments[0].gap_end == pytest.approx(0.018)
    assert prepared.segments[0].duration == pytest.approx(robotiq_grip_timeout_s(variant, open_mm, 18.0, 40.0))
    assert prepared.segments[2].duration == pytest.approx(robotiq_release_timeout_s(variant, 18.0, 40.0))

    with pytest.raises(ValueError, match="speed_percent > 0"):
        prepare_robotiq_program(program, replace(settings, speed_percent=0.0), variant)


@pytest.mark.parametrize(("variant", "open_mm"), [("robotiq_2f_85", 85.0), ("robotiq_2f_140", 140.0)])
def test_release_gate_and_summarize_accepts_variant_open_gap(variant: str, open_mm: float) -> None:
    del variant
    gate = ReleaseGate(50.0, open_position_mm=open_mm)
    assert not gate.observe({"time_s": 0.02, "gap_mm": open_mm - 0.3, "left_normal_n": 0.0, "right_normal_n": 0.0})
    assert gate.observe({"time_s": 0.04, "gap_mm": open_mm, "left_normal_n": 0.0, "right_normal_n": 0.0})

    force_gate = GripForceGate(50.0)
    assert not force_gate.observe(
        {"time_s": 0.02, "left_normal_n": 100.0, "right_normal_n": 100.0, "expected_static_force_n": 100.0}
    )
    assert force_gate.observe(
        {"time_s": 0.04, "left_normal_n": 100.0, "right_normal_n": 100.0, "expected_static_force_n": 100.0}
    )

    def _row(t: float, phase: str, *, gap: float, left: float, right: float, z: float, obj_xy: tuple[float, float]):
        return {
            "time_s": t,
            "phase": phase,
            "gap_mm": gap,
            "left_normal_n": left,
            "right_normal_n": right,
            "expected_static_force_n": 100.0,
            "drive_torque_nm": 2.0,
            "torque_limit_nm": 5.0,
            "motor_torque_estimate_nm": 0.4,
            "sensor_force_n": (left + right) / 2.0,
            "object_base_x_mm": obj_xy[0],
            "object_base_y_mm": obj_xy[1],
            "object_base_z_mm": z,
            "object_in_pad_x_mm": 0.0,
            "object_in_pad_y_mm": 0.0,
            "object_in_pad_z_mm": 0.0,
            "object_in_tool_x_mm": 0.0,
            "object_in_tool_y_mm": 0.0,
            "object_in_tool_z_mm": 150.0,
            "object_speed_mm_s": 0.0,
            "home_error_mm": 0.5,
        }

    rows = [
        _row(0.02, "grip", gap=30.0, left=100.0, right=100.0, z=15.0, obj_xy=(300.0, 0.0)),
        _row(0.04, "grip", gap=30.0, left=100.0, right=100.0, z=15.0, obj_xy=(300.0, 0.0)),
        _row(0.06, "lift", gap=30.0, left=100.0, right=100.0, z=80.0, obj_xy=(300.0, 0.0)),
        _row(0.08, "transit", gap=30.0, left=100.0, right=100.0, z=80.0, obj_xy=(300.0, 300.0)),
        _row(0.10, "release", gap=open_mm, left=0.0, right=0.0, z=15.0, obj_xy=(300.0, 300.0)),
        _row(0.12, "release", gap=open_mm, left=0.0, right=0.0, z=15.0, obj_xy=(300.0, 300.0)),
        _row(0.14, "retreat", gap=open_mm, left=0.0, right=0.0, z=15.0, obj_xy=(300.0, 300.0)),
        *[
            _row(0.16 + 0.02 * i, "return-home", gap=open_mm, left=0.0, right=0.0, z=15.0, obj_xy=(300.0, 300.0))
            for i in range(10)
        ],
    ]
    config = SimpleNamespace(
        motion=SimpleNamespace(rate_hz=50.0),
        task=SimpleNamespace(
            parameters={
                "fixed_object_position_m": (0.3, 0.0, 0.015),
                "fixed_target_position_m": (0.3, 0.3, 0.015),
                "object_size_m": (0.03, 0.03, 0.03),
            }
        ),
    )
    summary = summarize(rows, config, open_position_mm=open_mm)
    assert summary["passed"] is True, summary["failed_checks"]
    assert summary["timing_passed"] is True
    assert summary["transport_passed"] is True
