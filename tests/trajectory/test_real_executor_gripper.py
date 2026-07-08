"""Unit tests for real gripper SDK integration (mocked XArmAPI)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ufactory.trajectory.real_executor import (
    GRIPPER_G2_MAX_SPEED_MM_S,
    GRIPPER_G2_MIN_SPEED_MM_S,
    RealExecutorConfig,
    _gripper_g2_target_speed_mm_s,
    _gripper_motion_enabled,
    _run_gripper_segment,
)
from ufactory.trajectory.segments import Segment


def _grip_segment(gap_start=0.084, gap_end=0.024, duration=2.0, label="grip") -> Segment:
    return Segment(kind="gripper", duration=duration, v_max=0.0, a_max=0.0, label=label, gap_start=gap_start, gap_end=gap_end)


def _make_gripper_arm():
    arm = MagicMock()
    arm.set_gripper_g2_position.return_value = 0
    arm.get_gripper_g2_position.return_value = (0, 24.0)
    return arm


@pytest.mark.parametrize(
    "dry_run,sdk_sim_validate,real_gripper,expected",
    [
        (True, False, True, False),
        (False, True, True, False),
        (True, True, True, False),
        (False, False, False, False),
        (False, False, True, True),
    ],
)
def test_gripper_motion_enabled_gating(dry_run, sdk_sim_validate, real_gripper, expected):
    cfg = RealExecutorConfig(
        dry_run=dry_run,
        sdk_sim_validate=sdk_sim_validate,
        real_gripper=real_gripper,
    )
    assert _gripper_motion_enabled(cfg) is expected


def test_gripper_g2_target_speed_within_range():
    seg = _grip_segment(gap_start=0.084, gap_end=0.024, duration=2.0)
    # (84-24)mm / 2s = 30 mm/s, within [15, 225].
    assert _gripper_g2_target_speed_mm_s(seg) == pytest.approx(30.0)


def test_gripper_g2_target_speed_clamped_to_minimum():
    seg = _grip_segment(gap_start=0.030, gap_end=0.024, duration=2.0)
    # (30-24)mm / 2s = 3 mm/s, below the SDK's 15 mm/s floor.
    assert _gripper_g2_target_speed_mm_s(seg) == pytest.approx(GRIPPER_G2_MIN_SPEED_MM_S)


def test_gripper_g2_target_speed_clamped_to_maximum():
    seg = _grip_segment(gap_start=0.084, gap_end=0.000, duration=0.1)
    # 84mm / 0.1s = 840 mm/s, above the SDK's 225 mm/s ceiling.
    assert _gripper_g2_target_speed_mm_s(seg) == pytest.approx(GRIPPER_G2_MAX_SPEED_MM_S)


def test_run_gripper_segment_sends_sdk_command_for_real_motion(monkeypatch):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = _make_gripper_arm()
    cfg = RealExecutorConfig(dry_run=False, sdk_sim_validate=False, real_gripper=True, rate=50.0)
    seg = _grip_segment()

    _run_gripper_segment(seg, cfg, arm)

    arm.set_gripper_g2_position.assert_called_once()
    args, kwargs = arm.set_gripper_g2_position.call_args
    assert args[0] == pytest.approx(24.0)  # gap_end=0.024m -> 24mm
    assert kwargs["speed"] == pytest.approx(30.0)
    assert kwargs["wait"] is False
    arm.get_gripper_g2_position.assert_called_once()


def test_run_gripper_segment_skips_sdk_command_during_dry_run(monkeypatch):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = _make_gripper_arm()
    cfg = RealExecutorConfig(dry_run=True, sdk_sim_validate=False, rate=50.0)
    seg = _grip_segment()

    _run_gripper_segment(seg, cfg, arm)

    arm.set_gripper_g2_position.assert_not_called()


def test_run_gripper_segment_skips_sdk_command_when_real_gripper_not_enabled(monkeypatch):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = _make_gripper_arm()
    cfg = RealExecutorConfig(dry_run=False, sdk_sim_validate=False, real_gripper=False, rate=50.0)
    seg = _grip_segment()

    _run_gripper_segment(seg, cfg, arm)

    arm.set_gripper_g2_position.assert_not_called()


def test_run_gripper_segment_skips_sdk_command_during_sdk_sim_validate(monkeypatch):
    """--sdk-sim-validate keeps cfg.dry_run False but must not move the physical gripper."""
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = _make_gripper_arm()
    cfg = RealExecutorConfig(dry_run=False, sdk_sim_validate=True, rate=50.0)
    seg = _grip_segment()

    _run_gripper_segment(seg, cfg, arm)

    arm.set_gripper_g2_position.assert_not_called()


def test_run_gripper_segment_raises_on_sdk_failure_code(monkeypatch):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = _make_gripper_arm()
    arm.set_gripper_g2_position.return_value = 1
    cfg = RealExecutorConfig(dry_run=False, sdk_sim_validate=False, real_gripper=True, rate=50.0)
    seg = _grip_segment()

    with pytest.raises(RuntimeError, match="set_gripper_g2_position failed"):
        _run_gripper_segment(seg, cfg, arm)


def test_run_gripper_segment_warns_but_does_not_raise_on_position_mismatch(monkeypatch, capsys):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = _make_gripper_arm()
    arm.get_gripper_g2_position.return_value = (0, 40.0)  # target was 24mm; 16mm off, over tolerance
    cfg = RealExecutorConfig(dry_run=False, sdk_sim_validate=False, real_gripper=True, rate=50.0)
    seg = _grip_segment()

    _run_gripper_segment(seg, cfg, arm)  # must not raise

    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_run_lite6_gripper_segment_close_and_open_commands(monkeypatch):
    monkeypatch.setattr("ufactory.trajectory.real_executor.time.sleep", lambda _s: None)
    arm = MagicMock()
    arm.close_lite6_gripper.return_value = 0
    arm.open_lite6_gripper.return_value = 0
    cfg = RealExecutorConfig(
        robot_key="lite6",
        dry_run=False,
        sdk_sim_validate=False,
        real_gripper=True,
        rate=50.0,
    )

    _run_gripper_segment(_grip_segment(gap_start=0.038, gap_end=0.028), cfg, arm)
    _run_gripper_segment(_grip_segment(gap_start=0.028, gap_end=0.038, label="release"), cfg, arm)

    arm.close_lite6_gripper.assert_called_once_with(sync=False)
    arm.open_lite6_gripper.assert_called_once_with(sync=False)
