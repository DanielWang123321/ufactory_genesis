"""Robot profile resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.paths import robot_urdf, xarm5_urdf, xarm6_urdf, xarm7_urdf
from ufactory.robot_registry import (
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
