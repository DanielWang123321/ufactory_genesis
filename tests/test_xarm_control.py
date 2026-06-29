"""Unit tests for ufactory.xarm_control (mocked XArmAPI)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ufactory.xarm_control import (
    MODE_POSITION,
    MODE_SERVO,
    REPORT_STATE_STOPPING,
    STATE_MOTION,
    assert_motion_ready,
    prepare_arm_for_motion,
)


def _make_arm(
    *,
    state: int = 0,
    error_code: int = 0,
    warn_code: int = 0,
    mode: int = 0,
):
    arm = MagicMock()
    arm.state = state
    arm.mode = mode
    arm.error_code = error_code
    arm.warn_code = warn_code
    arm.clean_warn.return_value = 0
    arm.clean_error.return_value = 0
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.get_state.return_value = (0, 0)
    arm.emergency_stop = MagicMock()
    return arm


def test_prepare_arm_for_motion_happy_path():
    arm = _make_arm(state=0)
    prepare_arm_for_motion(arm, mode=MODE_POSITION)

    arm.clean_warn.assert_called()
    arm.motion_enable.assert_called_with(enable=True)
    arm.set_mode.assert_called_with(MODE_POSITION)
    arm.set_state.assert_called_with(STATE_MOTION)


def test_prepare_arm_for_motion_waits_out_of_stop_state():
    arm = _make_arm(state=REPORT_STATE_STOPPING)
    states = [REPORT_STATE_STOPPING, REPORT_STATE_STOPPING, 0]

    def read_state():
        return states.pop(0) if states else 0

    type(arm).state = property(lambda self: read_state())

    prepare_arm_for_motion(arm, mode=MODE_SERVO, retries=3, poll_timeout_s=1.0)
    arm.set_mode.assert_called_with(MODE_SERVO)


def test_prepare_arm_for_motion_cleans_error_first():
    arm = _make_arm(state=0, error_code=1)

    def clear_error():
        arm.error_code = 0
        return 0

    arm.clean_error.side_effect = clear_error
    prepare_arm_for_motion(arm)
    arm.clean_error.assert_called()


def test_prepare_arm_for_motion_fails_when_stuck_in_stop():
    arm = _make_arm(state=REPORT_STATE_STOPPING)
    with pytest.raises(RuntimeError, match="state=4"):
        prepare_arm_for_motion(arm, retries=1, poll_timeout_s=0.05)


def test_prepare_arm_for_motion_retries_on_set_mode_failure():
    arm = _make_arm(state=0)
    arm.set_mode.side_effect = [1, 0]
    prepare_arm_for_motion(arm, retries=2, poll_timeout_s=0.05)
    assert arm.set_mode.call_count == 2


def test_assert_motion_ready_rejects_stop_state():
    arm = _make_arm(state=REPORT_STATE_STOPPING)
    with pytest.raises(RuntimeError, match="not ready"):
        assert_motion_ready(arm)


def test_assert_motion_ready_rejects_error():
    arm = _make_arm(state=0, error_code=5)
    with pytest.raises(RuntimeError, match="active error"):
        assert_motion_ready(arm)
