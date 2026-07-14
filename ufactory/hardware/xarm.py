"""xArm Python SDK motion preparation helpers (state / mode per API docs)."""

from __future__ import annotations

import time
from typing import Any

# set_mode values
MODE_POSITION = 0  # set_servo_angle, set_position
MODE_SERVO = 1  # set_servo_angle_j, set_servo_cartesian
MODE_JOINT_TEACH = 2
MODE_JOINT_VEL = 4  # vc_set_joint_velocity
MODE_CART_VEL = 5  # vc_set_cartesian_velocity
MODE_JOINT_ONLINE_PLANNING = 6  # set_servo_angle in joint online trajectory planning
MODE_CART_ONLINE_PLANNING = 7  # set_position in cartesian online trajectory planning

# set_state parameter values
STATE_MOTION = 0
STATE_PAUSE = 3
STATE_STOP = 4

# get_state() / arm.state reported values (xArm API docs, not set_state params)
REPORT_STATE_IN_MOTION = 1
REPORT_STATE_SLEEPING = 2  # idle; normal after set_state(0) in MODE_SERVO
REPORT_STATE_SUSPENDED = 3
REPORT_STATE_STOPPING = 4

POLL_INTERVAL_S = 0.15
POLL_TIMEOUT_S = 2.0
SERVO_READY_TIMEOUT_S = 2.0
SERVO_PRIME_RETRIES = 10
SERVO_PRIME_RETRY_S = 0.02
SERVO_CMD_RETRIES = 5
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


def assert_motion_ready(arm: Any) -> None:
    """Fast pre-motion check: not in stop state and no active error."""
    if arm.state == REPORT_STATE_STOPPING:
        raise RuntimeError(
            f"Arm not ready for motion ({format_arm_status(arm)}). "
            "Inspect the physical E-stop and recover manually in xArm Studio."
        )
    if arm.error_code != 0:
        raise RuntimeError(
            f"Arm has active error ({format_arm_status(arm)}). Inspect and recover manually in xArm Studio."
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
    retries: int = 3,
    poll_timeout_s: float = POLL_TIMEOUT_S,
) -> None:
    """Enable an already healthy arm; never clear/reset/recover controller faults."""
    del retries
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
            f"controller did not confirm motion state ({format_arm_status(arm)}); inspect it manually in xArm Studio"
        )
    assert_motion_ready(arm)


def prepare_gripper_g2_for_motion(arm: Any, *, retries: int = 3, poll_s: float = 0.15) -> None:
    """Enable an already healthy G2; never clear gripper errors automatically."""
    last_error = ""

    for attempt in range(1, retries + 1):
        code, err_code = arm.get_gripper_err_code()
        if code == 0 and err_code != 0:
            raise RuntimeError(f"Gripper G2 has error {err_code}; recover manually before motion")

        code = arm.set_gripper_enable(True)
        if code != 0:
            last_error = f"set_gripper_enable(True) returned {code}"
            time.sleep(poll_s)
            continue

        code = arm.set_gripper_mode(0)
        if code == 0:
            return
        last_error = f"set_gripper_mode(0) returned {code}"
        time.sleep(poll_s)

    raise RuntimeError(f"Failed to prepare Gripper G2 for motion after {retries} attempts. Last: {last_error}.")


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


def prime_servo_cartesian(
    arm: Any,
    pose: list[float],
    *,
    speed: float,
    mvacc: float,
    retries: int = SERVO_PRIME_RETRIES,
    retry_s: float = SERVO_PRIME_RETRY_S,
) -> None:
    """Send initial ``set_servo_cartesian`` until MODE_SERVO accepts the stream.

    ``pose`` is ``[x_mm, y_mm, z_mm, roll, pitch, yaw]``; the roll/pitch/yaw are
    radians (``is_radian=True``), matching the xArm SDK contract
    ``set_servo_cartesian(mvpose, speed, mvacc, mvtime=0, is_radian=True)``.
    The SDK ``speed``/``mvacc`` fields are reserved for this servo-refresh API;
    host-side target deltas are the real velocity/acceleration limit.
    """
    last_code = -1
    for _ in range(retries):
        last_code = arm.set_servo_cartesian(
            mvpose=pose,
            speed=speed,
            mvacc=mvacc,
            mvtime=0,
            is_radian=True,
        )
        if last_code == 0:
            return
        time.sleep(retry_s)
    raise RuntimeError(
        f"prime set_servo_cartesian failed after {retries} attempts, last code={last_code} ({format_arm_status(arm)})"
    )


def prime_online_joint_planning(
    arm: Any,
    angles: list[float],
    *,
    speed: float,
    mvacc: float,
    retries: int = SERVO_PRIME_RETRIES,
    retry_s: float = SERVO_PRIME_RETRY_S,
) -> None:
    """Send current joint target until mode-6 online joint planning accepts commands."""
    last_code = -1
    for _ in range(retries):
        last_code = arm.set_servo_angle(
            angle=angles,
            speed=speed,
            mvacc=mvacc,
            mvtime=0,
            wait=False,
            is_radian=True,
            radius=None,
        )
        if last_code == 0:
            return
        time.sleep(retry_s)
    raise RuntimeError(
        f"prime online joint planning failed after {retries} attempts, last code={last_code} ({format_arm_status(arm)})"
    )
