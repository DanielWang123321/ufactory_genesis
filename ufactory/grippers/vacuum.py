"""xArm / UF850 vacuum gripper SDK adapter and TGPIO pin mapping."""

from __future__ import annotations

_VACUUM_IO_PINS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (3, 4),
}


def vacuum_io_pins(hardware_version: int) -> tuple[int, int]:
    """Return ``(on_pin, off_pin)`` TGPIO numbers for vacuum gripper ``hardware_version`` (1 or 2)."""
    if isinstance(hardware_version, bool) or not isinstance(hardware_version, int):
        raise ValueError(f"hardware_version must be integer 1 or 2, got {hardware_version!r}")
    try:
        return _VACUUM_IO_PINS[hardware_version]
    except KeyError as exc:
        raise ValueError(f"unsupported vacuum gripper hardware_version={hardware_version}; expected 1 or 2") from exc


class VacuumGripperAdapter:
    """Adapter for xArm / UF850 pneumatic vacuum gripper driven over TGPIO."""

    def __init__(self, *, hardware_version: int = 1) -> None:
        self.on_pin, self.off_pin = vacuum_io_pins(hardware_version)
        self.hardware_version = int(hardware_version)

    def prepare_real(self, arm: object) -> int:
        missing = [
            name for name in ("set_vacuum_gripper", "get_vacuum_gripper") if not callable(getattr(arm, name, None))
        ]
        if missing:
            raise RuntimeError(f"Vacuum gripper SDK capability is unavailable: {', '.join(missing)}")
        code, _state = getattr(arm, "get_vacuum_gripper")(hardware_version=self.hardware_version)
        return int(code)

    def send_real_suction(self, arm: object, on: bool) -> int:
        if isinstance(on, bool):
            suction_on = on
        elif isinstance(on, (int, float)) and on in (0, 1):
            suction_on = bool(on)
        else:
            raise ValueError(f"on must be a boolean, got {on!r}")
        method = getattr(arm, "set_vacuum_gripper")
        return int(method(suction_on, wait=False, hardware_version=self.hardware_version))

    def read_real(self, arm: object) -> tuple[int, int]:
        code, state = getattr(arm, "get_vacuum_gripper")(hardware_version=self.hardware_version)
        return int(code), int(state)
