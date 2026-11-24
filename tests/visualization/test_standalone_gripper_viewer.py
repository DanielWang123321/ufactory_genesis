"""Standalone tool viewer: deferred window startup, mounted home poses, and CLI parsing."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("genesis")

from ufactory.kinematics.tcp_offset import rpy_matrix  # noqa: E402
from ufactory.trajectory.bio_g2_frames import HOME_GRASP_RPY_DEG  # noqa: E402
from ufactory.visualization import standalone_gripper  # noqa: E402
from ufactory.visualization.standalone_gripper import (  # noqa: E402
    MOUNTED_QUAT_WXYZ,
    STANDALONE_TOOL_SPECS,
    UPRIGHT_QUAT_WXYZ,
    parse_standalone_tool_args,
)


def test_viewer_opens_after_warmup_through_deferred_viewer(monkeypatch):
    calls: list[str] = []
    scene = SimpleNamespace(step=lambda: calls.append("step"))
    monkeypatch.setattr(
        standalone_gripper, "start_deferred_viewer", lambda target: calls.append(f"open:{target is scene}")
    )

    standalone_gripper._open_viewer(scene)

    assert calls == ["step"] * standalone_gripper.VIEWER_WARMUP_STEPS + ["open:True"]


def _pose(xyz, rotation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, xyz
    return out


def _joint_origin(root: ET.Element, name: str) -> np.ndarray:
    origin = root.find(f"./joint[@name='{name}']/origin")
    assert origin is not None, f"missing joint {name!r}"
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return _pose(xyz, rpy_matrix(*[float(v) for v in origin.get("rpy", "0 0 0").split()]))


def _quat_wxyz_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_bio_g2_standalone_view_uses_the_mounted_home_orientation():
    spec = STANDALONE_TOOL_SPECS["bio_gripper_g2"]
    root = ET.parse(spec.urdf_loader()).getroot()
    base = _pose(spec.base_pos, _quat_wxyz_matrix(spec.base_quat)) @ _joint_origin(root, "world_joint")
    tcp = base @ _joint_origin(root, "bio_gripper_g2_tcp_joint")
    # Same TCP orientation as on the arm at home (grasp rpy 180, 0, 0): TCP down, adapter flange up.
    np.testing.assert_allclose(tcp[:3, :3], rpy_matrix(*np.radians(HOME_GRASP_RPY_DEG)), atol=1e-12)
    np.testing.assert_allclose(base[:3, 3], [0.0, 0.0, 0.17], atol=1e-12)


@pytest.mark.parametrize(
    ("tool", "base_joint", "tcp_joint", "expected_base_z", "expected_tcp_z"),
    [
        ("gripper_g2", "world_joint", "joint_tcp", 0.23, 0.23 - 0.1735),
        ("robotiq_2f_85", "world_joint", "robotiq_2f_85_tcp_joint", 0.23, 0.23 - 0.162253),
        ("robotiq_2f_140", "world_joint", "robotiq_2f_140_tcp_joint", 0.28, 0.28 - 0.234466),
        ("lite6_gripper", "world_joint", "joint_tcp", 0.18, 0.18 - 0.0836),
        ("vacuum_gripper", "vacuum_gripper_fix", "vacuum_joint_tcp", 0.20, 0.20 - 0.126),
        ("lite6_vacuum_gripper", "vacuum_gripper_fix", "joint_tcp", 0.17, 0.17 - 0.061),
    ],
)
def test_standalone_tools_use_mounted_home_orientation(
    tool: str,
    base_joint: str,
    tcp_joint: str,
    expected_base_z: float,
    expected_tcp_z: float,
):
    spec = STANDALONE_TOOL_SPECS[tool]
    assert spec.base_quat == MOUNTED_QUAT_WXYZ
    root = ET.parse(spec.urdf_loader()).getroot()
    base = _pose(spec.base_pos, _quat_wxyz_matrix(spec.base_quat)) @ _joint_origin(root, base_joint)
    tcp = base @ _joint_origin(root, tcp_joint)
    np.testing.assert_allclose(tcp[:3, :3], rpy_matrix(*np.radians((180.0, 0.0, 0.0))), atol=1e-12)
    np.testing.assert_allclose(base[:3, 3], [0.0, 0.0, expected_base_z], atol=1e-12)
    np.testing.assert_allclose(tcp[:3, 3], [0.0, 0.0, expected_tcp_z], atol=1e-12)


@pytest.mark.parametrize(
    ("plate", "thickness_m"),
    [
        ("xarm_base_plate", 0.008),
        ("uf850_base_plate", 0.010),
        ("lite6_base_plate", 0.005),
    ],
)
def test_standalone_base_plates_use_upright_floor_pose(plate: str, thickness_m: float):
    spec = STANDALONE_TOOL_SPECS[plate]
    assert spec.base_quat == UPRIGHT_QUAT_WXYZ
    assert spec.base_pos == pytest.approx((0.0, 0.0, thickness_m))
    root = ET.parse(spec.urdf_loader()).getroot()
    fix = root.find("./joint[@name='base_plate_fix']")
    assert fix is not None
    assert fix.find("parent").get("link") == "world"
    assert fix.find("child").get("link") == plate


@pytest.mark.parametrize(
    ("tool", "exp_pos", "exp_speed", "exp_force"),
    [
        ("gripper_g2", 0.0, 50.0, 50.0),
        ("robotiq_2f_85", 0.0, 50.0, 50.0),
        ("robotiq_2f_140", 0.0, 50.0, 50.0),
        ("bio_gripper_g2", 71.0, 50.0, 50.0),
        ("lite6_gripper", None, None, None),
        ("vacuum_gripper", None, None, None),
        ("lite6_vacuum_gripper", None, None, None),
        ("xarm_base_plate", None, None, None),
        ("uf850_base_plate", None, None, None),
        ("lite6_base_plate", None, None, None),
    ],
)
def test_parse_standalone_tool_args_defaults(tool: str, exp_pos, exp_speed, exp_force):
    spec, args = parse_standalone_tool_args(["--tool", tool, "--headless"])
    assert spec.name == tool
    assert args.headless is True
    assert args.backend == "gpu"
    assert args.position_mm == exp_pos
    assert args.speed_percent == exp_speed
    assert args.force_percent == exp_force


@pytest.mark.parametrize(
    "argv",
    [
        ["--tool", "lite6_gripper", "--position-mm", "10"],
        ["--tool", "vacuum_gripper", "--position-mm", "10"],
        ["--tool", "lite6_vacuum_gripper", "--speed-percent", "50"],
        ["--tool", "xarm_base_plate", "--position-mm", "10"],
        ["--tool", "uf850_base_plate", "--speed-percent", "50"],
        ["--tool", "lite6_base_plate", "--force-percent", "50"],
        ["--tool", "gripper_g2", "--speed-mm-s", "20"],
        ["--tool", "gripper_g2", "--position-mm", "100"],
        ["--tool", "robotiq_2f_85", "--position-mm", "100"],
        ["--tool", "robotiq_2f_140", "--position-mm", "160"],
        ["--tool", "robotiq_2f_85", "--speed-mm-s", "20"],
        ["--tool", "bio_gripper_g2", "--position-mm", "40"],
    ],
)
def test_parse_standalone_tool_args_rejects_invalid_combinations(argv: list[str]):
    with pytest.raises(SystemExit):
        parse_standalone_tool_args(argv)
