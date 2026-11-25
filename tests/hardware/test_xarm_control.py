"""Unit tests for ufactory.hardware.xarm (mocked XArmAPI).

The helpers only use the xArm Python SDK shared by every UFACTORY arm, so the motion
cases are model-agnostic; per-model behaviour is limited to controller identification.
"""

from __future__ import annotations

import pytest

from ufactory.hardware.xarm import (
    MODE_POSITION,
    MODE_SERVO,
    REPORT_STATE_SLEEPING,
    REPORT_STATE_STOPPING,
    STATE_MOTION,
    assert_motion_ready,
    controller_robot_name,
    prepare_arm_for_motion,
    prime_servo_angle_j,
    verify_controller_robot,
    wait_for_servo_motion_ready,
)
from ufactory.robots.registry import get_robot_profile

# (axis, device_type) read from the five supported controllers (firmware v2.8.x).
CONTROLLER_IDENTITIES = {
    "xarm5_1305": (5, 5),
    "xarm6_1305": (6, 6),
    "xarm7_1305": (7, 7),
    "lite6": (6, 9),
    "uf850": (6, 12),
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _s: None)


def test_prepare_arm_for_motion_happy_path(make_arm):
    arm = make_arm(state=0)
    prepare_arm_for_motion(arm, mode=MODE_POSITION)

    arm.clean_warn.assert_not_called()
    arm.motion_enable.assert_called_with(enable=True)
    arm.set_mode.assert_called_with(MODE_POSITION)
    arm.set_state.assert_called_with(STATE_MOTION)


def test_prepare_arm_for_motion_rejects_stop_state_without_recovery(make_arm):
    arm = make_arm(state=REPORT_STATE_STOPPING)
    with pytest.raises(RuntimeError, match="not ready"):
        prepare_arm_for_motion(arm, mode=MODE_SERVO)
    arm.motion_enable.assert_not_called()
    arm.emergency_stop.assert_not_called()


def test_prepare_arm_for_motion_rejects_error_without_cleaning(make_arm):
    arm = make_arm(state=0, error_code=1)

    with pytest.raises(RuntimeError, match="active error"):
        prepare_arm_for_motion(arm)
    arm.clean_error.assert_not_called()


def test_prepare_arm_for_motion_fails_when_controller_stays_in_stop(make_arm):
    arm = make_arm(state=0)

    def drop_to_stop(_state):
        arm.state = REPORT_STATE_STOPPING
        return 0

    arm.set_state.side_effect = drop_to_stop
    with pytest.raises(RuntimeError, match="did not confirm motion state"):
        prepare_arm_for_motion(arm, poll_timeout_s=0.01)
    arm.clean_error.assert_not_called()


def test_prepare_arm_for_motion_stops_on_set_mode_failure(make_arm):
    arm = make_arm(state=0)
    arm.set_mode.return_value = 1
    with pytest.raises(RuntimeError, match="without automatic recovery"):
        prepare_arm_for_motion(arm, poll_timeout_s=0.01)
    assert arm.set_mode.call_count == 1
    arm.set_state.assert_called_once_with(REPORT_STATE_STOPPING)


def test_assert_motion_ready_rejects_stop_state(make_arm):
    arm = make_arm(state=REPORT_STATE_STOPPING)
    with pytest.raises(RuntimeError, match="not ready"):
        assert_motion_ready(arm)


def test_assert_motion_ready_rejects_error(make_arm):
    arm = make_arm(state=0, error_code=5)
    with pytest.raises(RuntimeError, match="active error"):
        assert_motion_ready(arm)


def test_wait_for_servo_motion_ready_accepts_sleeping_state(make_arm):
    arm = make_arm(state=REPORT_STATE_SLEEPING, mode=MODE_SERVO)
    arm.get_state.return_value = (0, REPORT_STATE_SLEEPING)
    wait_for_servo_motion_ready(arm, timeout_s=0.1, poll_s=0.01)


def test_wait_for_servo_motion_ready_rejects_stopping(make_arm):
    arm = make_arm(state=REPORT_STATE_STOPPING, mode=MODE_SERVO)
    arm.get_state.return_value = (0, REPORT_STATE_STOPPING)
    with pytest.raises(RuntimeError, match="not ready"):
        wait_for_servo_motion_ready(arm, timeout_s=0.05, poll_s=0.01)


def test_wait_for_servo_motion_ready_times_out_outside_servo_mode(make_arm):
    arm = make_arm(state=REPORT_STATE_SLEEPING, mode=MODE_POSITION)
    arm.get_state.return_value = (0, REPORT_STATE_SLEEPING)
    with pytest.raises(RuntimeError, match="Servo mode not ready"):
        wait_for_servo_motion_ready(arm, timeout_s=0.01, poll_s=0.0)


def test_prime_servo_angle_j_retries_until_accepted(make_arm):
    arm = make_arm(mode=MODE_SERVO)
    arm.set_servo_angle_j.side_effect = [9, 9, 0]

    prime_servo_angle_j(arm, [0.1] * 6, speed=0.0, mvacc=0.0, retries=5, retry_s=0.0)

    assert arm.set_servo_angle_j.call_count == 3
    assert arm.set_servo_angle_j.call_args.kwargs == {
        "angles": [0.1] * 6,
        "speed": 0.0,
        "mvacc": 0.0,
        "mvtime": 0,
        "is_radian": True,
    }


def test_prime_servo_angle_j_raises_after_retries(make_arm):
    arm = make_arm(mode=MODE_SERVO)
    arm.set_servo_angle_j.return_value = 9

    with pytest.raises(RuntimeError, match="after 3 attempts, last code=9"):
        prime_servo_angle_j(arm, [0.0] * 6, speed=0.0, mvacc=0.0, retries=3, retry_s=0.0)
    assert arm.set_servo_angle_j.call_count == 3


@pytest.mark.parametrize("robot_key", sorted(CONTROLLER_IDENTITIES))
def test_verify_controller_robot_accepts_matching_model(make_arm, robot_key):
    axis, device_type = CONTROLLER_IDENTITIES[robot_key]
    arm = make_arm(axis=axis, device_type=device_type)
    expected = get_robot_profile(robot_key).robot_name

    assert controller_robot_name(arm) == expected
    assert verify_controller_robot(arm, robot_key) == expected


@pytest.mark.parametrize(
    ("robot_key", "connected"),
    [("xarm6_1305", "lite6"), ("xarm6_1305", "uf850"), ("lite6", "xarm6_1305"), ("uf850", "lite6")],
)
def test_verify_controller_robot_refuses_another_six_axis_model(make_arm, robot_key, connected):
    axis, device_type = CONTROLLER_IDENTITIES[connected]
    arm = make_arm(axis=axis, device_type=device_type)

    with pytest.raises(RuntimeError, match=f"controller reports {get_robot_profile(connected).robot_name} "):
        verify_controller_robot(arm, robot_key)


def test_verify_controller_robot_refuses_axis_mismatch_without_device_type(make_arm):
    arm = make_arm(axis=5)

    with pytest.raises(RuntimeError, match="controller reports 5 axes"):
        verify_controller_robot(arm, "xarm6_1305")


@pytest.mark.parametrize(("axis", "device_type"), [(None, None), (6, 99)])
def test_verify_controller_robot_trusts_declared_model_when_unidentified(make_arm, capsys, axis, device_type):
    arm = make_arm(axis=axis, device_type=device_type)

    assert controller_robot_name(arm) is None
    assert verify_controller_robot(arm, "xarm6_1305") is None
    assert "trusting the declared robot xarm6" in capsys.readouterr().out
