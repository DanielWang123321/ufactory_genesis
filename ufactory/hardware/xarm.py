"""xArm Python SDK helpers shared by every UFACTORY arm (xArm5/6/7, Lite6, UF850).

Motion preparation (state / mode per API docs) and controller model identification.
"""

from __future__ import annotations

import numbers
import time
from typing import Any

from ufactory.robots.registry import get_robot_profile, robot_name_for_controller

# set_mode values
MODE_POSITION = 0  # set_servo_angle, set_position
MODE_SERVO = 1  # set_servo_angle_j, set_servo_cartesian

# set_state parameter values
STATE_MOTION = 0

# get_state() / arm.state reported values (xArm API docs, not set_state params)
REPORT_STATE_IN_MOTION = 1
REPORT_STATE_SLEEPING = 2  # idle; normal after set_state(0) in MODE_SERVO
REPORT_STATE_STOPPING = 4

POLL_INTERVAL_S = 0.15
POLL_TIMEOUT_S = 2.0
SERVO_READY_TIMEOUT_S = 2.0
SERVO_PRIME_RETRIES = 10
SERVO_PRIME_RETRY_S = 0.02
SERVO_CMD_RETRY_S = 0.01
# xArm UxbusState.STATE_NOT_READY (controller not ready for move_servoj)
STATE_NOT_READY_SDK = 9


def format_arm_status(arm: Any) -> str:
    """Human-readable arm status for error messages."""
    parts = [
        f"state={arm.state}",
        f"mode={arm.mode}",
        f"error_code={arm.error_code}",
        f"warn_code={arm.warn_code}",
    ]
    code, reported = arm.get_state()
    if code == 0:
        parts.append(f"get_state()={reported}")
    return ", ".join(parts)


def _reported_int(arm: Any, name: str) -> int | None:
    """Positive integer SDK attribute (``axis`` / ``device_type``), else ``None``."""
    value = getattr(arm, name, None)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) > 0:
        return int(value)
    return None


def controller_robot_name(arm: Any) -> str | None:
    """Robot name the controller reports (``axis`` + ``device_type``); ``None`` if unreadable or unknown."""
    axis = _reported_int(arm, "axis")
    device_type = _reported_int(arm, "device_type")
    if axis is None or device_type is None:
        return None
    return robot_name_for_controller(axis, device_type)


def verify_controller_robot(arm: Any, robot_key: str) -> str | None:
    """Refuse a controller that reports a different robot model than ``robot_key``.

    Returns the verified robot name. When the controller model cannot be read or is not a
    known model, prints a warning and returns ``None``: the caller's declared robot is trusted.
    Raises ``RuntimeError`` on a definite mismatch; call it before any controller command.
    """
    profile = get_robot_profile(robot_key)
    axis = _reported_int(arm, "axis")
    device_type = _reported_int(arm, "device_type")
    if axis is not None and axis != profile.dof:
        raise RuntimeError(
            f"controller reports {axis} axes (device_type={device_type}) but robot {profile.robot_name} "
            f"has {profile.dof}; refusing to continue"
        )
    detected = controller_robot_name(arm)
    if detected is None:
        print(
            f"[WARN] cannot identify the controller model (axis={axis}, device_type={device_type}); "
            f"trusting the declared robot {profile.robot_name}"
        )
        return None
    if detected != profile.robot_name:
        raise RuntimeError(
            f"controller reports {detected} (axis={axis}, device_type={device_type}) "
            f"but the requested robot is {profile.robot_name}; refusing to continue"
        )
    return detected


def assert_motion_ready(arm: Any) -> None:
    """Fast pre-motion check: not in stop state and no active error."""
    if arm.state == REPORT_STATE_STOPPING:
        raise RuntimeError(
            f"Arm not ready for motion ({format_arm_status(arm)}). "
            "Inspect the physical E-stop and recover manually in UFACTORY Studio."
        )
    if arm.error_code != 0:
        raise RuntimeError(
            f"Arm has active error ({format_arm_status(arm)}). Inspect and recover manually in UFACTORY Studio."
        )


def _wait_until_not_stopping(arm: Any, *, timeout_s: float = POLL_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if arm.state != REPORT_STATE_STOPPING:
            return True
        time.sleep(POLL_INTERVAL_S)
    return arm.state != REPORT_STATE_STOPPING


def prepare_arm_for_motion(
    arm: Any,
    *,
    mode: int = MODE_POSITION,
    poll_timeout_s: float = POLL_TIMEOUT_S,
) -> None:
    """Enable an already healthy arm; never clear/reset/recover controller faults."""
    assert_motion_ready(arm)
    operations = (
        ("motion_enable", lambda: arm.motion_enable(enable=True)),
        (f"set_mode({mode})", lambda: arm.set_mode(mode)),
        ("set_state(0)", lambda: arm.set_state(STATE_MOTION)),
    )
    for label, operation in operations:
        code = operation()
        if code != 0:
            arm.set_state(REPORT_STATE_STOPPING)
            raise RuntimeError(f"{label} returned {code}; controller stopped without automatic recovery")
    if not _wait_until_not_stopping(arm, timeout_s=poll_timeout_s):
        raise RuntimeError(
            f"controller did not confirm motion state ({format_arm_status(arm)}); "
            "inspect it manually in UFACTORY Studio"
        )
    assert_motion_ready(arm)


def _report_state_allows_servo(state: int) -> bool:
    """Whether reported get_state() value allows set_servo_angle_j streaming."""
    return state in (REPORT_STATE_IN_MOTION, REPORT_STATE_SLEEPING)


def wait_for_servo_motion_ready(
    arm: Any,
    *,
    timeout_s: float = SERVO_READY_TIMEOUT_S,
    poll_s: float = POLL_INTERVAL_S,
) -> None:
    """Wait until arm is in MODE_SERVO with a servoj-ready reported state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        assert_motion_ready(arm)
        code, state = arm.get_state()
        if code == 0 and arm.mode == MODE_SERVO and _report_state_allows_servo(state):
            return
        time.sleep(poll_s)
    raise RuntimeError(f"Servo mode not ready after {timeout_s}s ({format_arm_status(arm)})")


def prime_servo_angle_j(
    arm: Any,
    angles: list[float],
    *,
    speed: float,
    mvacc: float,
    retries: int = SERVO_PRIME_RETRIES,
    retry_s: float = SERVO_PRIME_RETRY_S,
) -> None:
    """Send initial ``set_servo_angle_j`` until controller accepts servoj stream.

    The SDK ``speed``/``mvacc`` fields are reserved for this servo-refresh API;
    host-side target deltas are the real velocity/acceleration limit.
    """
    last_code = -1
    for _ in range(retries):
        last_code = arm.set_servo_angle_j(
            angles=angles,
            speed=speed,
            mvacc=mvacc,
            mvtime=0,
            is_radian=True,
        )
        if last_code == 0:
            return
        time.sleep(retry_s)
    raise RuntimeError(
        f"prime set_servo_angle_j failed after {retries} attempts, last code={last_code} ({format_arm_status(arm)})"
    )
