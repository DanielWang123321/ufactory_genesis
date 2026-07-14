"""Dependency inversion ports used by the safety and execution domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ufactory.types import FloatArray


@dataclass(frozen=True)
class ArmState:
    q_rad: FloatArray
    monotonic_ns: int
    ready: bool
    error_code: int = 0
    state_code: int = 0
    serial_number: str = ""
    robot_key: str = ""


@dataclass(frozen=True)
class CollisionResult:
    colliding: bool
    min_distance_m: float
    link_a: str = ""
    link_b: str = ""
    environment: bool = False


@runtime_checkable
class ArmTransport(Protocol):
    def read_state(self) -> ArmState: ...
    def send_joint_target(self, q_rad: FloatArray) -> int: ...
    def send_cartesian_target(self, pose: FloatArray) -> int: ...
    def pause(self) -> int: ...
    def stop(self) -> int: ...
    def disconnect(self) -> None: ...


@runtime_checkable
class KinematicsBackend(Protocol):
    def forward(self, q_rad: FloatArray) -> FloatArray:
        """Return base-frame end-effector pose as xyz or xyz+quaternion."""

    def inverse(self, pose: FloatArray, seed_q_rad: FloatArray) -> FloatArray:
        """Solve xyz or xyz+quaternion_xyzw while preserving seed continuity."""


@runtime_checkable
class CollisionBackend(Protocol):
    def check(self, q_rad: FloatArray, *, stage: str, gripper_drive: float | None = None) -> CollisionResult: ...
    def check_all(
        self, q_rad: FloatArray, *, stage: str, gripper_drive: float | None = None
    ) -> tuple[CollisionResult, ...]: ...


@runtime_checkable
class MarginCollisionBackend(Protocol):
    """Optional capability returning only pairs at or inside a safety margin."""

    def check_all_within_margin(
        self,
        q_rad: FloatArray,
        *,
        stage: str,
        margin_m: float,
        gripper_drive: float | None = None,
    ) -> tuple[CollisionResult, ...]: ...


@runtime_checkable
class Clock(Protocol):
    def monotonic_ns(self) -> int: ...
    def wait_until_ns(self, deadline_ns: int) -> None: ...
