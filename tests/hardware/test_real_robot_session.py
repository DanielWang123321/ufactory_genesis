"""Unit tests for real robot session helpers (mocked SDK, any arm model)."""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufactory.hardware.session import (
    MOVE_STRATEGY_AXIS_SEQUENTIAL,
    MOVE_STRATEGY_DIRECT,
    HoldTimeSeriesSample,
    RealRobotSession,
    RobotMotionError,
    build_axis_sequential_waypoints,
)
from ufactory.hardware.xarm import MODE_POSITION
from ufactory.robots.runtime import DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S


@pytest.fixture
def make_config_arm(make_joint_arm):
    def _make(*, simulation_switches: bool = True):
        arm = make_joint_arm(np.zeros(6))
        arm.is_simulation_robot = False
        arm.set_gravity_direction = MagicMock(return_value=0)
        arm.set_tcp_load = MagicMock(return_value=0)
        arm.set_report_tau_or_i = MagicMock(return_value=0)
        arm.set_self_collision_detection = MagicMock(return_value=0)

        def set_simulation_robot(*, on_off):
            if simulation_switches:
                arm.is_simulation_robot = bool(on_off)
            return 0

        arm.set_simulation_robot = MagicMock(side_effect=set_simulation_robot)
        return arm

    return _make


def _install_fake_sdk(monkeypatch, *, axis: int, device_type: int) -> MagicMock:
    """Make ``xarm.wrapper.XArmAPI`` return a connected mock controller reporting ``(axis, device_type)``."""
    arm = MagicMock()
    arm.connected = True
    arm.axis = axis
    arm.device_type = device_type
    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = MagicMock(return_value=arm)  # type: ignore[attr-defined]
    package = types.ModuleType("xarm")
    package.wrapper = wrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarm", package)
    monkeypatch.setitem(sys.modules, "xarm.wrapper", wrapper)
    monkeypatch.setattr("ufactory.hardware.session.time.sleep", lambda _s: None)
    return arm


def test_session_connect_refuses_controller_of_another_model(monkeypatch):
    arm = _install_fake_sdk(monkeypatch, axis=6, device_type=9)  # Lite6 controller

    with pytest.raises(RuntimeError, match="controller reports lite6"):
        RealRobotSession("192.0.2.1", robot_key="xarm6_1305")

    arm.disconnect.assert_called_once()
    arm.motion_enable.assert_not_called()


def test_session_connect_derives_dof_from_verified_model(monkeypatch):
    arm = _install_fake_sdk(monkeypatch, axis=5, device_type=5)

    session = RealRobotSession("192.0.2.1", robot_key="xarm5")

    assert (session.robot_key, session.dof, session.home_qpos.shape) == ("xarm5_1305", 5, (5,))
    arm.disconnect.assert_not_called()


def test_axis_sequential_waypoints_move_one_joint_at_a_time():
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


def test_axis_sequential_move_uses_mode0_set_servo_angle(monkeypatch, make_joint_arm, make_session):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = make_joint_arm(np.zeros(6))
    session = make_session(arm)

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


def test_direct_move_keeps_single_set_servo_angle_command(monkeypatch, make_joint_arm, make_session):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = make_joint_arm(np.zeros(6))
    session = make_session(arm)

    session.move_to([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], move_strategy=MOVE_STRATEGY_DIRECT)

    assert arm.set_servo_angle.call_count == 1
    assert arm.set_servo_angle.call_args.kwargs["angle"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


@pytest.mark.parametrize("dof", [5, 6, 7])
def test_session_slices_padded_sdk_states_to_robot_dof(make_joint_arm, make_session, dof):
    sdk_q = np.arange(1.0, 8.0)  # the SDK reports 7 joint slots on every arm
    session = make_session(make_joint_arm(sdk_q), dof=dof)

    q, qvel, tau = session.get_joint_states()

    assert q.tolist() == sdk_q[:dof].tolist()
    assert len(qvel) == len(tau) == dof


def test_move_error_reports_waypoint_target_and_status(monkeypatch, make_joint_arm, make_session):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = make_joint_arm(np.zeros(6))
    arm.set_servo_angle.side_effect = lambda **kwargs: 9
    session = make_session(arm)

    with pytest.raises(RobotMotionError, match="waypoint 1/1") as exc_info:
        session.move_to([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], move_strategy=MOVE_STRATEGY_DIRECT)

    message = str(exc_info.value)
    assert "target_q=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]" in message
    assert "state=0" in message
    assert exc_info.value.code == 9


def test_configure_for_simulation_collision_check(monkeypatch, make_config_arm, make_session):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = make_config_arm()
    session = make_session(arm)

    session.configure_for_simulation_collision_check()

    arm.set_simulation_robot.assert_called_once_with(on_off=True)
    arm.set_self_collision_detection.assert_called_once_with(True)


def test_configure_for_dynamics_reenables_before_set_tcp_load(make_config_arm, make_session):
    arm = make_config_arm()
    session = make_session(arm)

    session.configure_for_dynamics()

    names = [c[0] for c in arm.method_calls]
    i_gravity = names.index("set_gravity_direction")
    i_tcp_load = names.index("set_tcp_load")
    assert "set_state" in names[i_gravity:i_tcp_load]
    arm.set_tcp_load.assert_called_once_with(0, [0, 0, 0])


def test_configure_for_dynamics_raises_when_set_tcp_load_rejected(make_config_arm, make_session):
    arm = make_config_arm()
    arm.set_tcp_load.return_value = 9
    session = make_session(arm)

    with pytest.raises(RuntimeError, match=r"set_tcp_load\(0\) failed with code 9"):
        session.configure_for_dynamics()


def test_simulation_switch_waits_for_report_before_next_command(monkeypatch, make_config_arm, make_session):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = make_config_arm(simulation_switches=False)
    polls = {"n": 0}

    def fake_sleep(_dt):
        polls["n"] += 1
        if polls["n"] == 3:
            arm.is_simulation_robot = True

    monkeypatch.setattr("ufactory.hardware.session.time.sleep", fake_sleep)
    session = make_session(arm)

    session.configure_for_simulation_collision_check()

    assert polls["n"] == 3
    arm.set_self_collision_detection.assert_called_once_with(True)


def test_simulation_switch_timeout_stops_without_motion(monkeypatch, make_config_arm, make_session):
    clock = {"t": 0.0}
    monkeypatch.setattr("ufactory.hardware.session.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("ufactory.hardware.session.time.sleep", lambda dt: clock.__setitem__("t", clock["t"] + dt))
    arm = make_config_arm(simulation_switches=False)
    session = make_session(arm)

    with pytest.raises(RuntimeError, match="simulation robot did not switch"):
        session.configure_for_simulation_collision_check(simulation_timeout_s=1.0)

    arm.set_state.assert_called_with(4)
    arm.set_self_collision_detection.assert_not_called()
    arm.set_servo_angle.assert_not_called()


def test_direct_move_default_speed_is_40_deg_s_equivalent(monkeypatch, make_joint_arm, make_session):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = make_joint_arm(np.zeros(6))
    session = make_session(arm)

    session.move_to([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert arm.set_servo_angle.call_args.kwargs["speed"] == DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S


def test_collect_hold_timeseries_records_current_and_tau(monkeypatch, make_joint_arm, make_session):
    clock = {"t": 0.0}
    monkeypatch.setattr("ufactory.hardware.session.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("ufactory.hardware.session.time.sleep", lambda dt: clock.__setitem__("t", clock["t"] + dt))
    tick = {"n": 0}
    arm = make_joint_arm(np.zeros(6))

    def get_joint_states(*, is_radian=True, num=3):
        del is_radian, num
        n = tick["n"]
        tick["n"] += 1
        tau = np.full(6, -10.0 - float(n), dtype=np.float64)
        return 0, [np.zeros(6).tolist(), np.zeros(6).tolist(), tau.tolist()]

    arm.get_joint_states.side_effect = get_joint_states
    arm.currents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
    arm.get_joints_torque.return_value = (0, np.zeros(6).tolist())
    session = make_session(arm)

    samples = session.collect_hold_timeseries(duration_s=0.5, poll_s=0.1)

    assert len(samples) == 6
    assert isinstance(samples[0], HoldTimeSeriesSample)
    assert samples[0].t_s == pytest.approx(0.0, abs=1e-3)
    assert samples[-1].t_s == pytest.approx(0.5, abs=0.15)
    assert samples[0].current.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert samples[0].tau.shape == (6,)
    assert samples[1].tau[0] == pytest.approx(-11.0)
    assert samples[5].tau[0] == pytest.approx(-15.0)
