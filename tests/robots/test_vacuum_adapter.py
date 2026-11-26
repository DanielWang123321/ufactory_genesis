"""Mock contract tests for VacuumGripperAdapter and TGPIO pin mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ufactory.grippers import VacuumGripperAdapter, vacuum_io_pins


@pytest.mark.parametrize(("version", "expected"), [(1, (0, 1)), (2, (3, 4))])
def test_vacuum_io_pins_mapping(version: int, expected: tuple[int, int]) -> None:
    assert vacuum_io_pins(version) == expected


@pytest.mark.parametrize("bad_version", [0, 3, -1, True, False, 1.5, "1"])
def test_vacuum_io_pins_rejects_invalid_version(bad_version) -> None:
    with pytest.raises(ValueError):
        vacuum_io_pins(bad_version)
    with pytest.raises(ValueError):
        VacuumGripperAdapter(hardware_version=bad_version)


@pytest.mark.parametrize("version", [1, 2])
def test_vacuum_adapter_sdk_contract(version: int) -> None:
    adapter = VacuumGripperAdapter(hardware_version=version)
    arm = MagicMock()
    arm.get_vacuum_gripper.return_value = (0, 0)
    arm.set_vacuum_gripper.return_value = 0

    assert adapter.prepare_real(arm) == 0
    arm.get_vacuum_gripper.assert_called_with(hardware_version=version)

    assert adapter.send_real_suction(arm, True) == 0
    arm.set_vacuum_gripper.assert_called_with(True, wait=False, hardware_version=version)

    assert adapter.send_real_suction(arm, False) == 0
    arm.set_vacuum_gripper.assert_called_with(False, wait=False, hardware_version=version)

    arm.get_vacuum_gripper.return_value = (0, 1)
    assert adapter.read_real(arm) == (0, 1)


def test_vacuum_adapter_prepare_real_requires_sdk_methods() -> None:
    adapter = VacuumGripperAdapter()

    class IncompleteArm:
        def get_vacuum_gripper(self, **_kwargs):
            return (0, 0)

    with pytest.raises(RuntimeError, match="set_vacuum_gripper"):
        adapter.prepare_real(IncompleteArm())


def test_vacuum_adapter_rejects_non_boolean_suction_argument() -> None:
    adapter = VacuumGripperAdapter()
    arm = MagicMock()
    with pytest.raises(ValueError, match="boolean"):
        adapter.send_real_suction(arm, "on")  # type: ignore[arg-type]
    arm.set_vacuum_gripper.assert_not_called()
