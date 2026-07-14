from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ufactory.config import load_runtime_config
from ufactory.grippers import create_gripper_adapter


def test_g2_mapping_and_real_sdk_contract():
    adapter = create_gripper_adapter(load_runtime_config("xarm6").gripper)
    assert adapter.drive_to_gap(adapter.gap_to_drive(0.05)) == pytest.approx(0.05)
    arm = MagicMock()
    arm.get_gripper_err_code.return_value = (0, 0)
    arm.set_gripper_enable.return_value = 0
    arm.set_gripper_mode.return_value = 0
    arm.set_gripper_g2_position.return_value = 0
    assert adapter.prepare_real(arm) == 0
    assert adapter.send_real_gap(arm, 0.05) == 0
    arm.set_gripper_g2_position.assert_called_once_with(pos=50.0, wait=False)


def test_g2_never_cleans_existing_error():
    adapter = create_gripper_adapter(load_runtime_config("xarm6").gripper)
    arm = MagicMock()
    arm.get_gripper_err_code.return_value = (0, 7)
    with pytest.raises(RuntimeError, match="recover manually"):
        adapter.prepare_real(arm)
    arm.clean_gripper_error.assert_not_called()


def test_lite6_quantizes_gap_to_binary_commands():
    adapter = create_gripper_adapter(load_runtime_config("lite6").gripper)
    arm = MagicMock()
    arm.open_lite6_gripper.return_value = 0
    arm.close_lite6_gripper.return_value = 0
    assert adapter.prepare_real(arm) == 0
    assert adapter.send_real_gap(arm, adapter.profile.open_gap_m) == 0
    assert adapter.send_real_gap(arm, adapter.profile.closed_gap_m) == 0
    arm.open_lite6_gripper.assert_called_once_with(sync=False)
    arm.close_lite6_gripper.assert_called_once_with(sync=False)


def test_lite6_real_precheck_requires_all_sdk_commands():
    adapter = create_gripper_adapter(load_runtime_config("lite6").gripper)

    class IncompleteArm:
        def open_lite6_gripper(self, **_kwargs):
            return 0

        def close_lite6_gripper(self, **_kwargs):
            return 0

    with pytest.raises(RuntimeError, match="stop_lite6_gripper"):
        adapter.prepare_real(IncompleteArm())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_gripper_mapping_rejects_invalid_gap(value):
    adapter = create_gripper_adapter(load_runtime_config("xarm6").gripper)
    with pytest.raises(ValueError):
        adapter.gap_to_drive(value)
