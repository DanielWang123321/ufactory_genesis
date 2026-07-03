"""Safety checks for real-robot reach deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ufactory.dynamics import parse_joint_limits
from ufactory.deploy.action_postprocess import process_reach_action_np
from ufactory.robots.paths import robot_urdf
from ufactory.robots.runtime import RobotRuntimeProfile, get_robot_runtime_profile
from ufactory.hardware.xarm import assert_motion_ready

from ufactory.deploy.reach_config import ReachDeployConfig


class SafetyViolation(RuntimeError):
    """Raised when a command would violate deploy safety limits."""


@dataclass
class SafetyLimits:
    z_min_m: float
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    joint_margin_rad: float
    max_joint_delta_rad: float
    action_clip: float


class SafetyGuard:
    """Clip actions and validate robot state before motion commands."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    @classmethod
    def from_runtime(
        cls,
        runtime: RobotRuntimeProfile,
        config: ReachDeployConfig,
        *,
        joint_margin_rad: float = 0.02,
    ) -> SafetyGuard:
        lower, upper = parse_joint_limits(
            robot_urdf(runtime.model.key),
            runtime.arm.joint_names,
        )
        limits = SafetyLimits(
            z_min_m=config.z_min_m,
            joint_lower=lower,
            joint_upper=upper,
            joint_margin_rad=joint_margin_rad,
            max_joint_delta_rad=config.max_joint_delta_rad,
            action_clip=config.action_clip,
        )
        return cls(limits)

    def check_arm_ready(self, arm: Any) -> None:
        assert_motion_ready(arm)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        return np.clip(action, -self.limits.action_clip, self.limits.action_clip)

    def joint_delta(self, action: np.ndarray, action_scale: float) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        command = process_reach_action_np(
            np.zeros_like(action_arr),
            action_arr,
            action_scale=action_scale,
            action_clip=self.limits.action_clip,
            max_joint_delta_rad=self.limits.max_joint_delta_rad,
        )
        return command.joint_delta

    def command_qpos(self, current_q: np.ndarray, action: np.ndarray, action_scale: float) -> np.ndarray:
        command = process_reach_action_np(
            current_q,
            action,
            action_scale=action_scale,
            action_clip=self.limits.action_clip,
            max_joint_delta_rad=self.limits.max_joint_delta_rad,
        )
        self.check_joint_limits(command.target_q)
        return command.target_q

    def check_joint_limits(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        margin = self.limits.joint_margin_rad
        lower = self.limits.joint_lower + margin
        upper = self.limits.joint_upper - margin
        if np.any(q < lower) or np.any(q > upper):
            raise SafetyViolation(
                f"Joint command out of limits: q={q.tolist()} "
                f"allowed=[{lower.tolist()}, {upper.tolist()}]"
            )

    def check_ee_position(self, ee_pos_m: np.ndarray) -> None:
        ee_pos_m = np.asarray(ee_pos_m, dtype=np.float64).reshape(3)
        if ee_pos_m[2] < self.limits.z_min_m:
            raise SafetyViolation(
                f"EE z={ee_pos_m[2] * 1000.0:.1f} mm below minimum "
                f"{self.limits.z_min_m * 1000.0:.1f} mm"
            )
