"""Tracked Lite6 geometry and command mapping are self-consistent."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ufactory.grippers.lite6 import (
    LITE6_GRIPPER_CLOSED_GAP_M,
    LITE6_GRIPPER_OPEN_GAP_M,
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    lite6_gripper_sim_drive_to_gap_m,
)
from ufactory.robots.paths import lite6_gripper_movable_visual_urdf


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
