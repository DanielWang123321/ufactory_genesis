from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

import pytest

from ufactory.config import load_runtime_config
from ufactory.grippers import create_gripper_adapter
from ufactory.grippers.lite6 import (
    LITE6_GRIPPER_CLOSED_GAP_M,
    LITE6_GRIPPER_DEMO_HOLD_STEPS,
    LITE6_GRIPPER_OPEN_GAP_M,
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    lite6_gripper_demo_drive,
    lite6_gripper_demo_label,
    lite6_gripper_sim_drive_to_gap_m,
)
from ufactory.robots.paths import lite6_gripper_movable_visual_urdf


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


def test_standalone_lite6_gripper_collision_assets_and_gap_mapping():
    urdf = Path(lite6_gripper_movable_visual_urdf()).resolve()
    root = ET.parse(urdf).getroot()
    for link_name in ("uflite_finger1", "uflite_finger2"):
        link = root.find(f".//link[@name='{link_name}']")
        assert link is not None
        collisions = link.findall("collision")
        assert len(collisions) == 2
        for collision in collisions:
            assert collision.find("./geometry/box") is not None
            assert collision.find("./geometry/mesh") is None
        visual_mesh = link.find("./visual/geometry/mesh")
        assert visual_mesh is not None
        mesh_path = (urdf.parent / str(visual_mesh.get("filename"))).resolve()
        assert mesh_path.is_file()
    assert lite6_gripper_sim_drive_to_gap_m(LITE6_GRIPPER_SIM_CLOSED_DRIVE) == pytest.approx(LITE6_GRIPPER_CLOSED_GAP_M)
    assert lite6_gripper_sim_drive_to_gap_m(LITE6_GRIPPER_SIM_OPEN_DRIVE) == pytest.approx(LITE6_GRIPPER_OPEN_GAP_M)


def test_lite6_collision_demo_drive_matches_glb_keyframes():
    cases = (
        (0, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (LITE6_GRIPPER_DEMO_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (LITE6_GRIPPER_DEMO_HOLD_STEPS, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
        (2 * LITE6_GRIPPER_DEMO_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
    )
    for step, expected in cases:
        assert lite6_gripper_demo_drive(step) == pytest.approx(expected)


def test_lite6_collision_demo_labels_snap_at_hold_boundary():
    assert lite6_gripper_demo_label(0) == "open"
    assert lite6_gripper_demo_label(LITE6_GRIPPER_DEMO_HOLD_STEPS - 1) == "open"
    assert lite6_gripper_demo_label(LITE6_GRIPPER_DEMO_HOLD_STEPS) == "closed"
    assert lite6_gripper_demo_label(2 * LITE6_GRIPPER_DEMO_HOLD_STEPS - 1) == "closed"
