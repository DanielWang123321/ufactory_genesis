"""Unit tests for real robot movement helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import math
import numpy as np
import pytest

from ufactory.real_robot_session import (
    MOVE_STRATEGY_AXIS_SEQUENTIAL,
    MOVE_STRATEGY_DIRECT,
    RealRobotSession,
    RobotMotionError,
    build_axis_sequential_waypoints,
)
from ufactory.robot_params import DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
from ufactory.xarm_control import MODE_POSITION


def _make_mock_arm(initial_q):
    current_q = np.asarray(initial_q, dtype=np.float64)
    arm = MagicMock()
    arm.state = 0
    arm.mode = MODE_POSITION
    arm.error_code = 0
    arm.warn_code = 0
    arm.clean_warn.return_value = 0
    arm.clean_error.return_value = 0
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.get_state.return_value = (0, 0)

    def get_joint_states(*, is_radian=True, num=3):
        del is_radian, num
        return 0, [current_q.tolist(), np.zeros(current_q.size).tolist(), np.zeros(current_q.size).tolist()]

    def set_servo_angle(*, angle, speed, mvtime, wait, is_radian, radius):
        del speed, mvtime, wait, is_radian, radius
        current_q[:] = np.asarray(angle, dtype=np.float64)
        return 0

    arm.get_joint_states.side_effect = get_joint_states
    arm.set_servo_angle.side_effect = set_servo_angle
    arm.set_servo_angle_j = MagicMock()
    return arm


def _make_session(arm):
    session = object.__new__(RealRobotSession)
    session.ip = "mock"
    session.dof = 6
    session.home_qpos = np.zeros(6)
    session._motion_mode = MODE_POSITION
    session.arm = arm
    return session


def test_axis_sequential_waypoints_move_j1_to_j6_in_order():
    waypoints = build_axis_sequential_waypoints(
        np.zeros(6),
        np.array([0.1, 0.2, 0.0, 0.4, 0.0, 0.6]),
    )

    expected = [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.0, 0.4, 0.0, 0.0],
        [0.1, 0.2, 0.0, 0.4, 0.0, 0.6],
    ]
    assert [w.tolist() for w in waypoints] == expected
    assert build_axis_sequential_waypoints(np.zeros(6), np.zeros(6)) == []


def test_axis_sequential_move_uses_mode0_set_servo_angle(monkeypatch):
    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = _make_mock_arm(np.zeros(6))
    session = _make_session(arm)

    session.move_to(
        [0.1, 0.0, 0.2, 0.0, 0.0, 0.3],
        speed_rad_s=math.radians(8.0),
        move_strategy=MOVE_STRATEGY_AXIS_SEQUENTIAL,
    )

    assert arm.set_mode.call_count == 3
    arm.set_mode.assert_called_with(MODE_POSITION)
    assert arm.set_servo_angle.call_count == 3
    assert [c.kwargs["angle"] for c in arm.set_servo_angle.call_args_list] == [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.2, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.2, 0.0, 0.0, 0.3],
    ]
    assert all(c.kwargs["radius"] is None for c in arm.set_servo_angle.call_args_list)
    arm.set_servo_angle_j.assert_not_called()


def test_direct_move_keeps_single_set_servo_angle_command(monkeypatch):
    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = _make_mock_arm(np.zeros(6))
    session = _make_session(arm)

    session.move_to([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], move_strategy=MOVE_STRATEGY_DIRECT)

    assert arm.set_servo_angle.call_count == 1
    assert arm.set_servo_angle.call_args.kwargs["angle"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_session_slices_states_by_configured_dof():
    arm = _make_mock_arm(np.arange(7, dtype=np.float64))
    session = _make_session(arm)
    session.dof = 7
    session.home_qpos = np.zeros(7)

    q, qvel, tau = session.get_joint_states()

    assert q.tolist() == list(np.arange(7, dtype=np.float64))
    assert len(qvel) == 7
    assert len(tau) == 7


def test_move_error_reports_waypoint_target_and_status(monkeypatch):
    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = _make_mock_arm(np.zeros(6))
    arm.set_servo_angle.side_effect = lambda **kwargs: 9
    session = _make_session(arm)

    with pytest.raises(RobotMotionError, match="waypoint 1/1") as exc_info:
        session.move_to([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], move_strategy=MOVE_STRATEGY_DIRECT)

    message = str(exc_info.value)
    assert "target_q=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]" in message
    assert "state=0" in message
    assert exc_info.value.code == 9


def test_configure_for_simulation_collision_check(monkeypatch):
    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = _make_mock_arm(np.zeros(6))
    arm.set_gravity_direction = MagicMock(return_value=0)
    arm.set_tcp_load = MagicMock(return_value=0)
    arm.set_report_tau_or_i = MagicMock(return_value=0)
    arm.set_simulation_robot = MagicMock(return_value=0)
    arm.set_self_collision_detection = MagicMock(return_value=0)
    session = _make_session(arm)

    session.configure_for_simulation_collision_check()

    arm.set_simulation_robot.assert_called_once_with(on_off=True)
    arm.set_self_collision_detection.assert_called_once_with(True)


def test_direct_move_default_speed_is_40_deg_s_equivalent(monkeypatch):
    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = _make_mock_arm(np.zeros(6))
    session = _make_session(arm)

    session.move_to([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert arm.set_servo_angle.call_args.kwargs["speed"] == DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
