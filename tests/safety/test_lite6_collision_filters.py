from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend


@pytest.mark.parametrize("urdf", ["lite6_with_gripper.urdf", "lite6_gripper_movable_visual.glb.urdf"])
def test_lite6_filter_removes_only_root_pair_from_exact_and_margin_queries(urdf):
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    config = load_runtime_config("lite6")
    kwargs = dict(
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions={config.gripper.drive_joint: 0.0},
    )
    path = Path(config.robot.assets_dir) / urdf
    baseline = PinocchioCollisionBackend(path, **kwargs)
    q = np.asarray(config.arm.default_qpos_rad)
    baseline.check_all(q, stage="start", gripper_drive=0.0)
    # Put obstacles inside a root and a pad: external collisions must survive.
    centers = []
    for name in ("uflite_finger1_0", "uflite_finger1_1"):
        index = baseline.geometry_model.getGeometryId(name)
        centers.append(tuple(baseline.geometry_data.oMg[index].translation))
    kwargs["obstacles"] = tuple(
        EnvironmentObstacle(name, (0.004, 0.004, 0.004), center)
        for name, center in zip(("root_probe", "pad_probe"), centers, strict=True)
    )
    baseline = PinocchioCollisionBackend(path, **kwargs)
    filtered = PinocchioCollisionBackend(path, excluded_geometry_pairs=config.safety.excluded_geometry_pairs, **kwargs)
    root_pair = frozenset(config.safety.excluded_geometry_pairs[0])
    assert filtered.collision_pair_count == baseline.collision_pair_count - 1
    for drive in (0.0, 0.001, 0.0025, 0.00445, 0.0089):
        raw = baseline.check_all(q, stage="start", gripper_drive=drive)
        expected = [result for result in raw if frozenset((result.geometry_a, result.geometry_b)) != root_pair]
        actual = filtered.check_all(q, stage="start", gripper_drive=drive)
        assert actual == tuple(expected)
        candidates = filtered.check_all_within_margin(q, stage="grip", gripper_drive=drive, margin_m=0.005)
        unsafe = [result for result in expected if result.colliding or result.min_distance_m <= 0.005]
        assert [(r.geometry_a, r.geometry_b) for r in candidates] == [(r.geometry_a, r.geometry_b) for r in unsafe]
        for candidate, reference in zip(candidates, unsafe, strict=True):
            assert candidate.colliding == reference.colliding
            assert np.isfinite(candidate.min_distance_m)
        if drive == 0.0:
            for name in ("root_probe", "pad_probe"):
                assert any(r.colliding and name in (r.geometry_a, r.geometry_b) for r in actual)
            finger_pairs = [
                r for r in actual if r.link_a.startswith("uflite_finger") and r.link_b.startswith("uflite_finger")
            ]
            assert len(finger_pairs) == 3


def test_unknown_excluded_geometry_fails_closed():
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    config = load_runtime_config("lite6")
    with pytest.raises(ValueError, match="missing from URDF"):
        PinocchioCollisionBackend(
            Path(config.robot.assets_dir) / config.robot.urdf,
            joint_names=config.robot.joint_names,
            ee_link=config.robot.ee_link,
            excluded_geometry_pairs=(("uflite_finger1_0", "unknown_0"),),
        )
