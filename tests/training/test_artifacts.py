from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ufactory.training import (
    ArtifactError,
    load_training_config,
    validate_checkpoint_artifacts,
    write_artifact_inventory,
    write_checkpoint_manifest,
    write_training_config,
)


def test_safe_checkpoint_manifest_and_weights_only_compatible(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 18, "num_actions": 6},
        reward={},
        robot={"joint_names": [f"joint{i}" for i in range(1, 7)]},
        train={"policy": {}},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(2)}, "iter": 0}, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(
        checkpoint,
        training_config=artifact,
        executor_action_contract="servo_j",
    )
    validated, manifest = validate_checkpoint_artifacts(
        checkpoint, config_path, expected_task="pick_place", expected_robot_key="xarm6_1305"
    )
    assert validated["config_sha256"] == manifest.config_sha256
    loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert loaded["iter"] == 0


def test_checkpoint_tamper_is_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 18, "num_actions": 6},
        reward={},
        robot={"joint_names": ["joint1"] * 6},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(checkpoint, training_config=artifact, executor_action_contract="servo_j")
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ArtifactError, match="checkpoint hash"):
        validate_checkpoint_artifacts(checkpoint, config_path)


def test_checkpoint_with_stale_runtime_config_hash_is_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 30, "num_actions": 7, "runtime_config_sha256": "old"},
        reward={},
        robot={"joint_names": [f"joint{i}" for i in range(1, 7)]},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(checkpoint, training_config=artifact, executor_action_contract="servo_j")

    with pytest.raises(ArtifactError, match="runtime config hash"):
        validate_checkpoint_artifacts(
            checkpoint,
            config_path,
            expected_runtime_config_sha256="new",
        )


def test_artifact_inventory_contains_resolved_config_and_checkpoint_hash(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 18, "num_actions": 6, "runtime_config_sha256": "runtime"},
        reward={"keypoints": 1.0},
        robot={"joint_names": [f"joint{i}" for i in range(1, 7)]},
        train={"seed": 1},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    artifact = load_training_config(config_path)
    output = write_artifact_inventory(
        tmp_path / "evaluation_artifacts.yaml",
        training_config=artifact,
        checkpoints=[checkpoint],
        selected_checkpoint=checkpoint,
    )

    inventory = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == 1
    assert inventory["training_config"] == artifact
    assert inventory["selected_checkpoint"] == "model_0.pt"
    assert inventory["checkpoints"][0]["file"] == "model_0.pt"
    assert len(inventory["checkpoints"][0]["sha256"]) == 64
