"""Robot profile resolution tests."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ufactory.robots.paths import (
    lite6_gripper_movable_visual_urdf,
    robot_urdf,
    robot_visual_glb_urdf,
    xarm5_urdf,
    xarm6_urdf,
    xarm7_urdf,
)
from ufactory.robots.registry import (
    get_profile_key_for_robot_name,
    get_robot_profile,
    robot_cli_choices,
)


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        ("xarm5", "xarm5_1305"),
        ("xarm6", "xarm6_1305"),
        ("xarm7", "xarm7_1305"),
        ("xarm5_1305", "xarm5_1305"),
        ("xarm6_1305", "xarm6_1305"),
        ("xarm7_1305", "xarm7_1305"),
        ("lite6", "lite6"),
        ("uf850", "uf850"),
    ],
)
def test_get_profile_key_for_robot_name(name: str, expected_key: str) -> None:
    assert get_profile_key_for_robot_name(name) == expected_key


def test_get_robot_profile_accepts_short_names() -> None:
    profile = get_robot_profile("xarm6")
    assert profile.key == "xarm6_1305"
    assert profile.default_urdf == "xarm6_1305.urdf"


def test_xarm_urdf_defaults_point_to_1305() -> None:
    for path_fn, suffix in (
        (xarm5_urdf, "xarm5_1305.urdf"),
        (xarm6_urdf, "xarm6_1305.urdf"),
        (xarm7_urdf, "xarm7_1305.urdf"),
    ):
        path = Path(path_fn())
        assert path.name == suffix
        assert path.is_file()


def test_robot_urdf_short_name() -> None:
    assert Path(robot_urdf("xarm6")).name == "xarm6_1305.urdf"


def test_robot_cli_choices_includes_aliases() -> None:
    choices = robot_cli_choices()
    assert "xarm6" in choices
    assert "xarm6_1305" in choices


def test_lite6_movable_gripper_defaults_to_reversed_assets() -> None:
    combo_path = Path(robot_visual_glb_urdf("lite6", with_lite6_gripper=True, movable=True))
    standalone_path = Path(lite6_gripper_movable_visual_urdf())

    assert combo_path.name == "lite6_gripper_movable_visual.glb.urdf"

    for path in (combo_path, standalone_path):
        root = ET.parse(path).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        assert joints["finger_joint1"].find("origin").get("xyz") == "0 0 0.0543"
        assert joints["finger_joint2"].find("origin").get("xyz") == "0 0 0.0543"
        finger_visual_meshes = {
            link.get("name"): link.find("./visual/geometry/mesh").get("filename")
            for link in root.findall("link")
            if link.get("name") in {"uflite_finger1", "uflite_finger2"}
        }
        assert finger_visual_meshes == {
            "uflite_finger1": "../lite6_gripper/meshes/collision/finger1.stl",
            "uflite_finger2": "../lite6_gripper/meshes/collision/finger2.stl",
        }
        for link_name in ("uflite_finger1", "uflite_finger2"):
            link = root.find(f".//link[@name='{link_name}']")
            collision_origin = link.find("./collision/origin") if link is not None else None
            visual_origin = link.find("./visual/origin") if link is not None else None
            assert collision_origin is not None
            assert visual_origin is not None
            assert collision_origin.get("xyz") == "0 0 0"
            assert visual_origin.get("xyz") == "0 0 0"
