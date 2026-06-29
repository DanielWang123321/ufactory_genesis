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

# set_state parameter values
STATE_MOTION = 0
STATE_PAUSE = 3
STATE_STOP = 4

# arm.state report value when motion APIs return STATE_NOT_READY (-2)
REPORT_STATE_STOPPING = 4

POLL_INTERVAL_S = 0.15
POLL_TIMEOUT_S = 2.0


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
            "Call prepare_arm_for_motion() or clear e-stop in xArm Studio."
        )
    if arm.error_code != 0:
        raise RuntimeError(
            f"Arm has active error ({format_arm_status(arm)}). "
            "Call clean_error() then prepare_arm_for_motion()."
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
    """SDK-aligned sequence before any motion command.

    Order: clean_warn -> clean_error (if needed) -> motion_enable ->
           set_mode(mode) -> set_state(0) -> poll until arm.state != 4.
    """
    last_error = ""

    for attempt in range(1, retries + 1):
        arm.clean_warn()
        time.sleep(0.1)

        if arm.error_code != 0:
            code = arm.clean_error()
            if code != 0:
                last_error = f"clean_error returned {code}"
                time.sleep(0.15)
                continue
            time.sleep(0.15)

        code = arm.motion_enable(enable=True)
        if code != 0:
            last_error = f"motion_enable returned {code}"
            time.sleep(0.15)
            continue

        code = arm.set_mode(mode)
        if code != 0:
            last_error = f"set_mode({mode}) returned {code}"
            time.sleep(0.15)
            continue

        code = arm.set_state(STATE_MOTION)
        if code != 0:
            last_error = f"set_state(0) returned {code}"
            time.sleep(0.15)
            continue

        if _wait_until_not_stopping(arm, timeout_s=poll_timeout_s):
            assert_motion_ready(arm)
            return

        last_error = f"arm.state still {REPORT_STATE_STOPPING} after set_state(0)"
        time.sleep(0.2)

    # Last resort: SDK emergency_stop recovery (motion_enable + set_state(0))
    if hasattr(arm, "emergency_stop"):
        arm.emergency_stop()
        time.sleep(0.3)
        if _wait_until_not_stopping(arm, timeout_s=poll_timeout_s) and arm.error_code == 0:
            assert_motion_ready(arm)
            return

    raise RuntimeError(
        f"Failed to prepare arm for motion (mode={mode}) after {retries} attempts. "
        f"Last: {last_error}. Status: {format_arm_status(arm)}. "
        "If state=4 persists, clear e-stop / resume in xArm Studio."
    )
