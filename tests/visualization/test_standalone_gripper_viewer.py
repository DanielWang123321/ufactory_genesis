"""Standalone gripper viewer: deferred window startup and the Bio G2 mounted pose."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("genesis")

from ufactory.kinematics.tcp_offset import rpy_matrix  # noqa: E402
from ufactory.robots.paths import bio_gripper_g2_movable_visual_urdf  # noqa: E402
from ufactory.trajectory.bio_g2_frames import HOME_GRASP_RPY_DEG  # noqa: E402
from ufactory.visualization import standalone_gripper  # noqa: E402


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
    xyz = [float(v) for v in origin.get("xyz").split()]
    return _pose(xyz, rpy_matrix(*[float(v) for v in origin.get("rpy").split()]))


def test_bio_g2_standalone_view_uses_the_mounted_home_orientation():
    from examples.visualization.view_bio_gripper_g2 import MOUNTED_QUAT_WXYZ, ROOT_POS

    w, x, y, z = MOUNTED_QUAT_WXYZ
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    root = ET.parse(bio_gripper_g2_movable_visual_urdf()).getroot()
    base = _pose(ROOT_POS, rotation) @ _joint_origin(root, "world_joint")
    tcp = base @ _joint_origin(root, "bio_gripper_g2_tcp_joint")
    # Same TCP orientation as on the arm at home (grasp rpy 180, 0, 0): TCP down, adapter flange up.
    np.testing.assert_allclose(tcp[:3, :3], rpy_matrix(*np.radians(HOME_GRASP_RPY_DEG)), atol=1e-12)
    np.testing.assert_allclose(base[:3, 3], [0.0, 0.0, 0.17], atol=1e-12)
