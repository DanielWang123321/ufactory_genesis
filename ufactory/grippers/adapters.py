"""Capability-driven gripper mapping without robot-family branching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from ufactory.config.models import GripperProfile


@dataclass(frozen=True)
class GripperCapabilities:
    real_command: bool
    feedback: bool
    closed_loop: bool
    allowed_contact_links: frozenset[str]


@runtime_checkable
class GripperAdapter(Protocol):
    profile: GripperProfile

    @property
    def capabilities(self) -> GripperCapabilities: ...
    def gap_to_drive(self, gap_m: float) -> float: ...
    def drive_to_gap(self, drive: float) -> float: ...
    def prepare_real(self, arm: object) -> int: ...
    def send_real_gap(self, arm: object, gap_m: float) -> int: ...


class ConfiguredGripperAdapter:
    """Linear mapping shared by reversed and conventional drive directions."""

    def __init__(self, profile: GripperProfile) -> None:
        self.profile = profile

    @property
    def capabilities(self) -> GripperCapabilities:
        return GripperCapabilities(
            real_command=self.profile.real_command,
            feedback=self.profile.feedback,
            closed_loop=self.profile.closed_loop,
            allowed_contact_links=self.profile.allowed_contact_links,
        )

    def _finite(self, value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def gap_to_drive(self, gap_m: float) -> float:
        gap = self._finite(gap_m, "gap_m")
        if not self.profile.closed_gap_m <= gap <= self.profile.open_gap_m:
            raise ValueError(f"gap_m {gap} outside [{self.profile.closed_gap_m}, {self.profile.open_gap_m}]")
        ratio = (gap - self.profile.closed_gap_m) / (self.profile.open_gap_m - self.profile.closed_gap_m)
        return self.profile.closed_drive + ratio * (self.profile.open_drive - self.profile.closed_drive)

    def drive_to_gap(self, drive: float) -> float:
        value = self._finite(drive, "drive")
        lo, hi = sorted((self.profile.open_drive, self.profile.closed_drive))
        if not lo <= value <= hi:
            raise ValueError(f"drive {value} outside [{lo}, {hi}]")
        denominator = self.profile.open_drive - self.profile.closed_drive
        if denominator == 0.0:
            raise ValueError("open and closed drive values cannot be equal")
        ratio = (value - self.profile.closed_drive) / denominator
        return self.profile.closed_gap_m + ratio * (self.profile.open_gap_m - self.profile.closed_gap_m)

    def send_real_gap(self, arm: object, gap_m: float) -> int:
        raise NotImplementedError

    def prepare_real(self, arm: object) -> int:
        del arm
        return 0


class G2Adapter(ConfiguredGripperAdapter):
    def prepare_real(self, arm: object) -> int:
        code, error = getattr(arm, "get_gripper_err_code")()
        if int(code) != 0:
            return int(code)
        if int(error) != 0:
            raise RuntimeError(f"G2 has error {error}; recover manually before real motion")
        code = int(getattr(arm, "set_gripper_enable")(True))
        return code if code != 0 else int(getattr(arm, "set_gripper_mode")(0))

    def send_real_gap(self, arm: object, gap_m: float) -> int:
        if not self.capabilities.real_command:
            raise RuntimeError("real G2 commands are not enabled for this robot profile")
        # xArm SDK G2 position uses total finger gap in millimetres.
        method = getattr(arm, "set_gripper_g2_position")
        return int(method(pos=self._finite(gap_m, "gap_m") * 1000.0, wait=False))


class Lite6Adapter(ConfiguredGripperAdapter):
    def send_real_gap(self, arm: object, gap_m: float) -> int:
        if not self.capabilities.real_command:
            raise RuntimeError("real Lite6 gripper commands are not enabled")
        gap = self._finite(gap_m, "gap_m")
        midpoint = (self.profile.open_gap_m + self.profile.closed_gap_m) / 2.0
        # Lite6 firmware exposes binary open/close commands rather than a
        # continuous gap command.  This quantisation belongs in the adapter,
        # never in the generic executor.
        method_name = "open_lite6_gripper" if gap >= midpoint else "close_lite6_gripper"
        return int(getattr(arm, method_name)(sync=False))


_ADAPTERS = {"g2": G2Adapter, "lite6": Lite6Adapter}


def create_gripper_adapter(profile: GripperProfile) -> GripperAdapter:
    try:
        adapter_type = _ADAPTERS[profile.adapter]
    except KeyError as exc:
        raise ValueError(f"unknown gripper adapter: {profile.adapter}") from exc
    return adapter_type(profile)
