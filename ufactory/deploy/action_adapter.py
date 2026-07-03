"""Map reach policy actions to xArm servo joint commands."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ufactory.deploy.action_postprocess import ReachActionCommand, process_reach_action_np
from ufactory.deploy.reach_config import EXECUTOR_ONLINE_JOINT, EXECUTOR_SERVO_J, ReachDeployConfig
from ufactory.deploy.safety import SafetyGuard
from ufactory.hardware.xarm import SERVO_CMD_RETRIES, SERVO_CMD_RETRY_S, STATE_NOT_READY_SDK


def build_servo_joint_command(
    current_q: np.ndarray,
    action: np.ndarray,
    *,
    config: ReachDeployConfig,
    safety: SafetyGuard,
) -> ReachActionCommand:
    """Convert a normalized policy action into a checked joint target."""
    command = process_reach_action_np(
        current_q,
        action,
        action_scale=config.action_scale,
        action_clip=config.action_clip,
        max_joint_delta_rad=config.max_joint_delta_rad,
    )
    safety.check_joint_limits(command.target_q)
    return command


def send_servo_joint_target(
    arm: Any,
    q_cmd: np.ndarray,
    *,
    config: ReachDeployConfig,
) -> None:
    """Send a joint target through xArm servo streaming with standard retries."""
    last_code = -1
    for attempt in range(SERVO_CMD_RETRIES):
        last_code = arm.set_servo_angle_j(
            angles=np.asarray(q_cmd, dtype=np.float64).reshape(-1).tolist(),
            speed=config.servo_speed_rad_s,
            mvacc=config.servo_mvacc_rad_s2,
            mvtime=0,
            is_radian=True,
        )
        if last_code == 0:
            return
        if last_code == STATE_NOT_READY_SDK and attempt + 1 < SERVO_CMD_RETRIES:
            time.sleep(SERVO_CMD_RETRY_S)
            continue
        break
    raise RuntimeError(f"set_servo_angle_j failed with code {last_code}")


def send_online_joint_target(
    arm: Any,
    q_cmd: np.ndarray,
    *,
    config: ReachDeployConfig,
) -> None:
    """Send a joint target through firmware mode-6 online trajectory planning."""
    last_code = -1
    target = np.asarray(q_cmd, dtype=np.float64).reshape(-1).tolist()
    for attempt in range(SERVO_CMD_RETRIES):
        last_code = arm.set_servo_angle(
            angle=target,
            speed=config.online_joint_speed_rad_s,
            mvacc=config.online_joint_mvacc_rad_s2,
            mvtime=0,
            wait=False,
            is_radian=True,
            radius=None,
        )
        if last_code == 0:
            return
        if last_code == STATE_NOT_READY_SDK and attempt + 1 < SERVO_CMD_RETRIES:
            time.sleep(SERVO_CMD_RETRY_S)
            continue
        break
    raise RuntimeError(f"set_servo_angle(mode6 online joint) failed with code {last_code}")


def apply_servo_joint_delta(
    arm: Any,
    current_q: np.ndarray,
    action: np.ndarray,
    *,
    config: ReachDeployConfig,
    safety: SafetyGuard,
    on_command: Callable[[ReachActionCommand], None] | None = None,
) -> np.ndarray:
    """Clip action, compute target q, and send ``set_servo_angle_j``."""
    command = build_servo_joint_command(current_q, action, config=config, safety=safety)
    if on_command is not None:
        on_command(command)
    send_servo_joint_target(arm, command.target_q, config=config)
    return command.target_q


def apply_online_joint_delta(
    arm: Any,
    current_q: np.ndarray,
    action: np.ndarray,
    *,
    config: ReachDeployConfig,
    safety: SafetyGuard,
    on_command: Callable[[ReachActionCommand], None] | None = None,
) -> np.ndarray:
    """Clip action, compute target q, and send it through mode-6 online planning."""
    command = build_servo_joint_command(current_q, action, config=config, safety=safety)
    if on_command is not None:
        on_command(command)
    send_online_joint_target(arm, command.target_q, config=config)
    return command.target_q


def apply_reach_joint_delta(
    arm: Any,
    current_q: np.ndarray,
    action: np.ndarray,
    *,
    config: ReachDeployConfig,
    safety: SafetyGuard,
    on_command: Callable[[ReachActionCommand], None] | None = None,
) -> np.ndarray:
    """Dispatch a reach joint-delta command through the configured executor."""
    if config.executor == EXECUTOR_SERVO_J:
        return apply_servo_joint_delta(
            arm,
            current_q,
            action,
            config=config,
            safety=safety,
            on_command=on_command,
        )
    if config.executor == EXECUTOR_ONLINE_JOINT:
        return apply_online_joint_delta(
            arm,
            current_q,
            action,
            config=config,
            safety=safety,
            on_command=on_command,
        )
    raise ValueError(f"Unknown reach executor: {config.executor}")
