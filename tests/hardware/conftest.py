"""Shared mocks for ufactory.hardware unit tests (no SDK, no controller, any arm model)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufactory.hardware.session import RealRobotSession
from ufactory.hardware.xarm import MODE_POSITION

# The xArm SDK reports 7 joint slots on every arm; sessions slice them to the robot DOF.
SDK_JOINT_SLOTS = 7


def _healthy_arm(
    *,
    state: int = 0,
    mode: int = MODE_POSITION,
    error_code: int = 0,
    warn_code: int = 0,
    axis: int | None = None,
    device_type: int | None = None,
) -> MagicMock:
    """MagicMock ``XArmAPI`` in a healthy state whose commands succeed; no model unless given."""
    arm = MagicMock()
    arm.state = state
    arm.mode = mode
    arm.error_code = error_code
    arm.warn_code = warn_code
    arm.axis = axis
    arm.device_type = device_type
    arm.clean_warn.return_value = 0
    arm.clean_error.return_value = 0
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.get_state.return_value = (0, 0)
    return arm


def _joint_arm(initial_q: Sequence[float]) -> MagicMock:
    """Healthy arm whose joint readback follows ``set_servo_angle``, padded to the SDK slot count."""
    current_q = np.asarray(initial_q, dtype=np.float64).copy()
    arm = _healthy_arm()

    def padded(values: np.ndarray) -> list[float]:
        slots = np.zeros(max(SDK_JOINT_SLOTS, values.size), dtype=np.float64)
        slots[: values.size] = values
        return slots.tolist()

    def get_joint_states(*, is_radian=True, num=3):
        del is_radian, num
        zeros = np.zeros(current_q.size)
        return 0, [padded(current_q), padded(zeros), padded(zeros)]

    def set_servo_angle(*, angle, speed, mvtime, wait, is_radian, radius):
        del speed, mvtime, wait, is_radian, radius
        current_q[: len(angle)] = np.asarray(angle, dtype=np.float64)
        return 0

    arm.get_joint_states.side_effect = get_joint_states
    arm.set_servo_angle.side_effect = set_servo_angle
    return arm


def _session(arm: MagicMock, *, dof: int = 6) -> RealRobotSession:
    """Session wired to ``arm`` without connecting (``__init__`` needs the SDK and a controller)."""
    session = object.__new__(RealRobotSession)
    session.ip = "mock"
    session.robot_key = "mock"
    session.dof = dof
    session.home_qpos = np.zeros(dof)
    session._motion_mode = MODE_POSITION
    session.arm = arm
    return session


@pytest.fixture
def make_arm() -> Callable[..., MagicMock]:
    return _healthy_arm


@pytest.fixture
def make_joint_arm() -> Callable[..., MagicMock]:
    return _joint_arm


@pytest.fixture
def make_session() -> Callable[..., RealRobotSession]:
    return _session
