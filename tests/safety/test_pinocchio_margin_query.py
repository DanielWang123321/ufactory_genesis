from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend
from ufactory.safety.adapters.pinocchio import StageAwareObjectCollisionBackend


@pytest.fixture()
def collision_backend() -> tuple[object, PinocchioCollisionBackend]:
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    config = load_runtime_config("xarm6", task="packaging_showcase")
    urdf = RepositoryAssetStore.discover().require(Path(config.robot.assets_dir) / config.robot.urdf)
    backend = PinocchioCollisionBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions={config.gripper.drive_joint: config.gripper.open_drive},
        adjacent_link_pairs=config.robot.adjacent_collision_pairs,
        obstacles=(
            EnvironmentObstacle("table", (1.2, 1.2, 0.05), (0.3, 0.0, -0.025)),
            EnvironmentObstacle("object", (0.03, 0.03, 0.03), (0.30, 0.0, 0.015)),
        ),
    )
    return config, backend


def test_margin_query_matches_exact_unsafe_pairs_and_restores_requests(collision_backend) -> None:
    config, backend = collision_backend
    q = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)
    margin = float(config.safety.min_collision_distance_m)
    initial_margins = [float(request.security_margin) for request in backend.geometry_data.collisionRequests]

    exact = backend.check_all(q, stage="start", gripper_drive=config.gripper.open_drive)
    expected = tuple(result for result in exact if result.colliding or result.min_distance_m <= margin)
    candidates = backend.check_all_within_margin(
        q,
        stage="start",
        gripper_drive=config.gripper.open_drive,
        margin_m=margin,
    )

    assert [(item.link_a, item.link_b, item.environment) for item in candidates] == [
        (item.link_a, item.link_b, item.environment) for item in expected
    ]
    for candidate, exact_result in zip(candidates, expected, strict=True):
        assert candidate.colliding is exact_result.colliding
        assert candidate.min_distance_m == pytest.approx(exact_result.min_distance_m, abs=1e-9)
    assert [float(request.security_margin) for request in backend.geometry_data.collisionRequests] == initial_margins

    exact_after = backend.check_all(q, stage="start", gripper_drive=config.gripper.open_drive)
    assert [item.colliding for item in exact_after] == [item.colliding for item in exact]
    assert [item.min_distance_m for item in exact_after] == pytest.approx(
        [item.min_distance_m for item in exact],
        abs=1e-9,
    )


@pytest.mark.parametrize("margin", [-0.001, np.nan, np.inf])
def test_margin_query_rejects_invalid_margin(collision_backend, margin: float) -> None:
    config, backend = collision_backend
    with pytest.raises(ValueError, match="margin"):
        backend.check_all_within_margin(
            np.asarray(config.arm.default_qpos_rad, dtype=np.float64),
            stage="start",
            gripper_drive=config.gripper.open_drive,
            margin_m=margin,
        )


def test_margin_query_restores_requests_when_coal_query_fails(collision_backend, monkeypatch) -> None:
    config, backend = collision_backend
    initial_margins = [float(request.security_margin) for request in backend.geometry_data.collisionRequests]

    def fail_compute(*args, **kwargs):
        raise RuntimeError("synthetic collision failure")

    monkeypatch.setattr(backend.pin, "computeCollisions", fail_compute)
    with pytest.raises(RuntimeError, match="synthetic"):
        backend.check_all_within_margin(
            np.asarray(config.arm.default_qpos_rad, dtype=np.float64),
            stage="start",
            gripper_drive=config.gripper.open_drive,
            margin_m=config.safety.min_collision_distance_m,
        )
    assert [float(request.security_margin) for request in backend.geometry_data.collisionRequests] == initial_margins


def test_stage_aware_margin_query_relocates_object(collision_backend) -> None:
    config, backend = collision_backend
    spawn = (0.30, 0.0, 0.015)
    place = (0.30, 0.30, 0.018)
    wrapped = StageAwareObjectCollisionBackend(
        backend,
        spawn_center_m=spawn,
        place_center_m=place,
        relocated_stages=("release",),
    )
    q = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)

    wrapped.check_all_within_margin(
        q,
        stage="release",
        gripper_drive=config.gripper.open_drive,
        margin_m=config.safety.min_collision_distance_m,
    )
    geometry = backend.geometry_model.geometryObjects[wrapped._object_index]
    np.testing.assert_allclose(geometry.placement.translation, place)

    wrapped.check_all_within_margin(
        q,
        stage="start",
        gripper_drive=config.gripper.open_drive,
        margin_m=config.safety.min_collision_distance_m,
    )
    np.testing.assert_allclose(geometry.placement.translation, spawn)
