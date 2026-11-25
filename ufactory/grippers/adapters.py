"""Capability-driven gripper mapping without robot-family branching."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol, runtime_checkable

from ufactory.config.models import GripperProfile
from ufactory.grippers.bio_g2 import bio_gripper_g2_gap_m_to_sim_drive, bio_gripper_g2_sim_drive_to_gap_m
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive, gripper_g2_sim_drive_to_gap_m
from ufactory.grippers.robotiq import RobotiqSpec, get_robotiq_spec, robotiq_percent_to_register


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
    def gap_to_drive(self, gap_m: float) -> float:
        gap = self._finite(gap_m, "gap_m")
        if not self.profile.closed_gap_m <= gap <= self.profile.open_gap_m:
            raise ValueError(f"gap_m {gap} outside [{self.profile.closed_gap_m}, {self.profile.open_gap_m}]")
        return gripper_g2_gap_m_to_sim_drive(gap)

    def drive_to_gap(self, drive: float) -> float:
        value = self._finite(drive, "drive")
        lo, hi = sorted((self.profile.open_drive, self.profile.closed_drive))
        if not lo <= value <= hi:
            raise ValueError(f"drive {value} outside [{lo}, {hi}]")
        return gripper_g2_sim_drive_to_gap_m(value)

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
    def prepare_real(self, arm: object) -> int:
        missing = [
            name
            for name in ("open_lite6_gripper", "close_lite6_gripper", "stop_lite6_gripper")
            if not callable(getattr(arm, name, None))
        ]
        if missing:
            raise RuntimeError(f"Lite6 gripper SDK capability is unavailable: {', '.join(missing)}")
        return 0

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


class BioG2Adapter(ConfiguredGripperAdapter):
    def gap_to_drive(self, gap_m: float) -> float:
        gap = self._finite(gap_m, "gap_m")
        if not self.profile.closed_gap_m <= gap <= self.profile.open_gap_m:
            raise ValueError(f"gap_m {gap} outside [{self.profile.closed_gap_m}, {self.profile.open_gap_m}]")
        return bio_gripper_g2_gap_m_to_sim_drive(gap)

    def drive_to_gap(self, drive: float) -> float:
        value = self._finite(drive, "drive")
        lo, hi = sorted((self.profile.open_drive, self.profile.closed_drive))
        if not lo <= value <= hi:
            raise ValueError(f"drive {value} outside [{lo}, {hi}]")
        return bio_gripper_g2_sim_drive_to_gap_m(value)

    def prepare_real(self, arm: object) -> int:
        code, error = getattr(arm, "get_bio_gripper_error")()
        if int(code) != 0:
            return int(code)
        if int(error) != 0:
            raise RuntimeError(f"Bio Gripper G2 has error {error}; recover manually before real motion")
        return int(getattr(arm, "set_bio_gripper_enable")(True))

    def send_real_gap(self, arm: object, gap_m: float) -> int:
        if not self.capabilities.real_command:
            raise RuntimeError("real Bio Gripper G2 commands are not enabled for this robot profile")
        method = getattr(arm, "set_bio_gripper_g2_position")
        return int(method(pos=self._finite(gap_m, "gap_m") * 1000.0, wait=False))


class RobotiqAdapter(ConfiguredGripperAdapter):
    """Adapter for Robotiq 2F-85 and 2F-140 grippers driven over xArm RS-485 Modbus."""

    def __init__(
        self,
        profile: GripperProfile,
        *,
        speed_percent: float = 50.0,
        force_percent: float = 50.0,
    ) -> None:
        super().__init__(profile)
        self.spec: RobotiqSpec = get_robotiq_spec(profile.adapter)
        self.speed_register: int = robotiq_percent_to_register(speed_percent)
        self.force_register: int = robotiq_percent_to_register(force_percent)

    def gap_to_drive(self, gap_m: float) -> float:
        gap = self._finite(gap_m, "gap_m")
        if not self.profile.closed_gap_m <= gap <= self.profile.open_gap_m:
            raise ValueError(f"gap_m {gap} outside [{self.profile.closed_gap_m}, {self.profile.open_gap_m}]")
        return self.spec.gap_m_to_sim_drive(gap)

    def drive_to_gap(self, drive: float) -> float:
        value = self._finite(drive, "drive")
        lo, hi = sorted((self.profile.open_drive, self.profile.closed_drive))
        if not lo <= value <= hi:
            raise ValueError(f"drive {value} outside [{lo}, {hi}]")
        return self.spec.sim_drive_to_gap_m(value)

    # gFLT 0x09 ("no communication for at least 1 s") is reported by the first poll after any idle period and clears
    # once polling resumes (2F-85 on xArm6, 2026-11-25: 9 on the first read, 0 from the next read 0.25 s later).
    COMM_TIMEOUT_FAULT = 0x09
    COMM_TIMEOUT_REREADS = 3
    COMM_TIMEOUT_REREAD_S = 0.1

    @staticmethod
    def _read_status(arm: object) -> tuple[int, dict, int]:
        code, _ret = getattr(arm, "robotiq_get_status")(number_of_registers=3)
        status = getattr(arm, "robotiq_status", {}) or {}
        return int(code), status, int(status.get("gFLT", getattr(arm, "robotiq_error_code", 0)))

    def prepare_real(self, arm: object) -> int:
        code, status, fault = self._read_status(arm)
        for _ in range(self.COMM_TIMEOUT_REREADS):
            if code != 0 or fault != self.COMM_TIMEOUT_FAULT:
                break
            time.sleep(self.COMM_TIMEOUT_REREAD_S)
            code, status, fault = self._read_status(arm)
        if code != 0:
            return code
        if fault != 0:
            raise RuntimeError(f"{self.spec.display_name} has fault gFLT={fault}; recover manually before real motion")
        gsta = int(status.get("gSTA", 0))
        if bool(getattr(arm, "robotiq_is_activated", False)) or gsta == 3:
            return 0
        # Never call robotiq_reset(); fail on faults and activate explicitly only if unactivated without fault.
        act_result = getattr(arm, "robotiq_set_activate")(wait=True)
        return int(act_result[0]) if isinstance(act_result, tuple) else int(act_result)

    def send_real_gap(self, arm: object, gap_m: float) -> int:
        if not self.capabilities.real_command:
            raise RuntimeError(f"real {self.spec.display_name} commands are not enabled for this robot profile")
        pos = self.spec.gap_m_to_register(self._finite(gap_m, "gap_m"))
        method = getattr(arm, "robotiq_set_position")
        result = method(
            pos,
            speed=self.speed_register,
            force=self.force_register,
            wait=False,
            wait_motion=False,
            auto_enable=False,
        )
        return int(result[0]) if isinstance(result, tuple) else int(result)


class Robotiq2F85Adapter(RobotiqAdapter):
    """Robotiq 2F-85 adapter."""


class Robotiq2F140Adapter(RobotiqAdapter):
    """Robotiq 2F-140 adapter."""


_ADAPTERS = {
    "g2": G2Adapter,
    "bio_g2": BioG2Adapter,
    "lite6": Lite6Adapter,
    "robotiq_2f_85": Robotiq2F85Adapter,
    "robotiq_2f_140": Robotiq2F140Adapter,
}


def create_gripper_adapter(profile: GripperProfile) -> GripperAdapter:
    try:
        adapter_type = _ADAPTERS[profile.adapter]
    except KeyError as exc:
        raise ValueError(f"unknown gripper adapter: {profile.adapter}") from exc
    return adapter_type(profile)
