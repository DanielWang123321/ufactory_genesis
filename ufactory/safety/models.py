"""Pure domain models for trajectory preflight and runtime safety.

This module deliberately has no dependency on Genesis, Torch, xArm SDK, or a
specific kinematics implementation.  Hardware and simulation adapters depend
on these models, never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, Mapping


class FaultClass(StrEnum):
    """Required controller action for a runtime fault."""

    PAUSE = "pause"
    STOP = "stop"


class ViolationType(StrEnum):
    INVALID_INPUT = "invalid_input"
    JOINT_LIMIT = "joint_limit"
    WORKSPACE = "workspace"
    Z_MIN = "z_min"
    IK_DISCONTINUITY = "ik_discontinuity"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    ORIENTATION = "orientation"
    SELF_COLLISION = "self_collision"
    ENVIRONMENT_COLLISION = "environment_collision"
    CLEARANCE = "clearance"
    TIMING = "timing"
    FEEDBACK = "feedback"
    IDENTITY = "identity"
    HASH_MISMATCH = "hash_mismatch"
    BACKEND_UNAVAILABLE = "backend_unavailable"


@dataclass(frozen=True)
class SafetyViolation:
    """One machine-readable preflight or runtime safety failure."""

    type: ViolationType
    stage: str
    message: str
    fault_class: FaultClass = FaultClass.STOP
    sample_index: int | None = None
    joint: str | None = None
    link: str | None = None
    actual: float | None = None
    limit: float | None = None

    def __post_init__(self) -> None:
        for name in ("actual", "limit"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"SafetyViolation.{name} must be finite")


@dataclass(frozen=True)
class CollisionExemption:
    """Version-controlled and narrowly scoped collision exemption."""

    link_a: str
    link_b: str
    stage: str
    reason: str
    urdf_sha256: str
    expires_version: str

    def matches(self, link_a: str, link_b: str, stage: str, urdf_sha256: str) -> bool:
        pair = frozenset((self.link_a, self.link_b))
        return pair == frozenset((link_a, link_b)) and self.stage == stage and self.urdf_sha256 == urdf_sha256


@dataclass(frozen=True)
class SafetyPolicy:
    """Immutable limits used by both static preflight and live monitoring."""

    schema_version: int = 1
    joint_limit_margin_rad: float = 0.02
    workspace_lower_m: tuple[float, float, float] = (-0.8, -0.8, 0.0)
    workspace_upper_m: tuple[float, float, float] = (0.8, 0.8, 1.2)
    z_min_m: float = 0.0
    min_collision_distance_m: float = 0.005
    max_ik_jump_rad: float = 0.35
    max_orientation_step_rad: float = 0.20
    max_shadow_joint_error_rad: float = 0.05
    max_shadow_ee_error_m: float = 0.005
    minor_lateness_ratio: float = 0.25
    minor_lateness_limit_per_s: int = 3
    feedback_stale_periods: float = 2.0
    require_kinematics: bool = True
    require_collision: bool = True
    exemptions: tuple[CollisionExemption, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported safety schema_version: {self.schema_version}")
        finite = {
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "z_min_m": self.z_min_m,
            "min_collision_distance_m": self.min_collision_distance_m,
            "max_ik_jump_rad": self.max_ik_jump_rad,
            "max_orientation_step_rad": self.max_orientation_step_rad,
            "max_shadow_joint_error_rad": self.max_shadow_joint_error_rad,
            "max_shadow_ee_error_m": self.max_shadow_ee_error_m,
            "minor_lateness_ratio": self.minor_lateness_ratio,
            "feedback_stale_periods": self.feedback_stale_periods,
            **{f"workspace_lower_m[{i}]": value for i, value in enumerate(self.workspace_lower_m)},
            **{f"workspace_upper_m[{i}]": value for i, value in enumerate(self.workspace_upper_m)},
        }
        for name, value in finite.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if len(self.workspace_lower_m) != 3 or len(self.workspace_upper_m) != 3:
            raise ValueError("workspace bounds must each contain exactly three values")
        if any(lo >= hi for lo, hi in zip(self.workspace_lower_m, self.workspace_upper_m, strict=True)):
            raise ValueError("each workspace lower bound must be below its upper bound")
        if not self.workspace_lower_m[2] <= self.z_min_m <= self.workspace_upper_m[2]:
            raise ValueError("z_min_m must lie within the workspace z bounds")
        if self.joint_limit_margin_rad < 0.0 or self.min_collision_distance_m < 0.0:
            raise ValueError("joint margin and collision distance cannot be negative")
        if not 0.0 <= self.minor_lateness_ratio <= 1.0:
            raise ValueError("minor_lateness_ratio must be in [0, 1]")
        if self.minor_lateness_limit_per_s < 1 or self.feedback_stale_periods <= 0.0:
            raise ValueError("timing thresholds must be positive")


@dataclass(frozen=True)
class PreflightCheck:
    """Result of one named check, including optional timing data."""

    name: str
    passed: bool
    checked_samples: int = 0
    duration_ms: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        if self.checked_samples < 0 or not math.isfinite(self.duration_ms) or self.duration_ms < 0.0:
            raise ValueError("invalid check metrics")


@dataclass(frozen=True)
class PreflightReport:
    """Complete, serializable result of all safety preflight checks."""

    schema_version: int
    passed: bool
    program_sha256: str
    robot_key: str
    executor: str
    config_sha256: str
    urdf_sha256: str
    calibration_sha256: str
    scene_sha256: str
    checks: tuple[PreflightCheck, ...]
    violations: tuple[SafetyViolation, ...]
    warnings: tuple[str, ...] = ()
    exemption_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = not self.violations and all(check.passed for check in self.checks)
        if self.passed != expected:
            raise ValueError("PreflightReport.passed does not match checks and violations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "program_sha256": self.program_sha256,
            "robot_key": self.robot_key,
            "executor": self.executor,
            "config_sha256": self.config_sha256,
            "urdf_sha256": self.urdf_sha256,
            "calibration_sha256": self.calibration_sha256,
            "scene_sha256": self.scene_sha256,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "checked_samples": check.checked_samples,
                    "duration_ms": check.duration_ms,
                    "details": dict(check.details),
                }
                for check in self.checks
            ],
            "violations": [
                {
                    "type": violation.type.value,
                    "stage": violation.stage,
                    "message": violation.message,
                    "fault_class": violation.fault_class.value,
                    "sample_index": violation.sample_index,
                    "joint": violation.joint,
                    "link": violation.link,
                    "actual": violation.actual,
                    "limit": violation.limit,
                }
                for violation in self.violations
            ],
            "warnings": list(self.warnings),
            "exemption_ids": list(self.exemption_ids),
        }
