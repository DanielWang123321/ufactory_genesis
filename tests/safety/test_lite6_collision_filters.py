"""Lite6 gripper: preflight skips pairs among its own geometries and keeps every pair with the environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.robots.paths import robot_urdf
from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend


def _finger_pairs(results):
    return [r for r in results if r.link_a.startswith("uflite_finger") and r.link_b.startswith("uflite_finger")]


@pytest.mark.parametrize("urdf", ["assembly:lite6_gripper:stl", "assembly:lite6_gripper"])
def test_lite6_finger_pairs_are_skipped_and_external_collisions_remain(urdf):
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    config = load_runtime_config("lite6")
    assert config.gripper is not None
    kwargs = dict(
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions={config.gripper.drive_joint: 0.0},
    )
    path = Path(robot_urdf(config.robot.key, urdf))
    plain = PinocchioCollisionBackend(path, **kwargs)
    q = np.asarray(config.arm.default_qpos_rad)
    closed = plain.check_all(q, stage="start", gripper_drive=0.0)
    # Fully closed, the finger-root boxes touch: the pair Genesis physics excludes through the gripper config.
    root_pair = set(config.gripper.excluded_geometry_pairs[0])
    assert [r.colliding for r in _finger_pairs(closed) if {r.geometry_a, r.geometry_b} == root_pair] == [True]

    # Put obstacles inside a root and a pad: their collisions with the environment must survive.
    centers = []
    for name in ("uflite_finger1_0", "uflite_finger1_1"):
        index = plain.geometry_model.getGeometryId(name)
        centers.append(tuple(plain.geometry_data.oMg[index].translation))
    kwargs["obstacles"] = tuple(
        EnvironmentObstacle(name, (0.004, 0.004, 0.004), center)
        for name, center in zip(("root_probe", "pad_probe"), centers, strict=True)
    )
    filtered = PinocchioCollisionBackend(path, tool_root_link=config.robot.ee_link, **kwargs)
    assert {"uflite_finger1", "uflite_finger2"} <= filtered.tool_links
    for drive in (0.0, 0.001, 0.0025, 0.00445, 0.0089):
        results = filtered.check_all(q, stage="start", gripper_drive=drive)
        assert not _finger_pairs(results)
        if drive == 0.0:
            for name in ("root_probe", "pad_probe"):
                assert any(r.colliding and r.environment and name in (r.geometry_a, r.geometry_b) for r in results)
