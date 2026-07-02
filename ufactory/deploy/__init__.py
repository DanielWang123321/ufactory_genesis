"""Sim-to-real deployment helpers for UFACTORY xArm reach tasks."""

from ufactory.deploy.action_postprocess import (
    ReachActionCommand,
    effective_max_joint_delta_rad,
    process_reach_action_np,
    reach_action_delta_torch,
)
from ufactory.deploy.obs_adapter import build_reach_obs
from ufactory.deploy.reach_config import (
    EXECUTOR_ONLINE_JOINT,
    EXECUTOR_SERVO_J,
    REACH_EXECUTORS,
    ReachDeployConfig,
    normalize_reach_executor,
)
from ufactory.deploy.safety import SafetyGuard, SafetyViolation

__all__ = [
    "ReachActionCommand",
    "ReachDeployConfig",
    "EXECUTOR_ONLINE_JOINT",
    "EXECUTOR_SERVO_J",
    "REACH_EXECUTORS",
    "SafetyGuard",
    "SafetyViolation",
    "build_reach_obs",
    "effective_max_joint_delta_rad",
    "normalize_reach_executor",
    "process_reach_action_np",
    "reach_action_delta_torch",
]
