from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ufactory.training import (
    ArtifactError,
    load_training_config,
    validate_and_load_rsl_checkpoint,
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


def test_checkpoint_with_matching_hash_but_conflicting_runtime_body_is_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    saved_env = {
        "num_obs": 35,
        "num_actions": 4,
        "runtime_config_sha256": "same",
        "fixed_target_pos": [0.30, -0.30, 0.015],
    }
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env=saved_env,
        reward={},
        robot={"joint_names": [f"joint{i}" for i in range(1, 7)]},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": {}}, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(checkpoint, training_config=artifact, executor_action_contract="cartesian_delta")

    expected_env = dict(saved_env)
    expected_env["fixed_target_pos"] = [0.30, 0.30, 0.015]
    with pytest.raises(ArtifactError, match=r"runtime config body .*fixed_target_pos"):
        validate_checkpoint_artifacts(
            checkpoint,
            config_path,
            expected_runtime_config_sha256="same",
            expected_runtime_env=expected_env,
        )


def test_checkpoint_layout_transfer_allows_only_explicit_layout_mismatch(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    saved_env = {
        "num_obs": 44,
        "num_actions": 4,
        "runtime_config_sha256": "same",
        "fixed_demo_layout": True,
        "constraint_solver": "newton",
    }
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env=saved_env,
        reward={},
        robot={},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": {}, "critic_state_dict": {}}, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(
        checkpoint,
        training_config=artifact,
        executor_action_contract="cartesian_delta",
    )

    expected_env = {**saved_env, "fixed_demo_layout": False}
    validate_checkpoint_artifacts(
        checkpoint,
        config_path,
        expected_runtime_config_sha256="same",
        expected_runtime_env=expected_env,
        allowed_runtime_env_mismatches=("fixed_demo_layout",),
    )

    expected_env["constraint_solver"] = "cg"
    with pytest.raises(ArtifactError, match=r"runtime config body .*constraint_solver"):
        validate_checkpoint_artifacts(
            checkpoint,
            config_path,
            expected_runtime_config_sha256="same",
            expected_runtime_env=expected_env,
            allowed_runtime_env_mismatches=("fixed_demo_layout",),
        )


def test_checkpoint_layout_transfer_rejects_unknown_allowed_key(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 44, "num_actions": 4},
        reward={},
        robot={},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": {}, "critic_state_dict": {}}, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(
        checkpoint,
        training_config=artifact,
        executor_action_contract="cartesian_delta",
    )

    with pytest.raises(ArtifactError, match="unsupported allowed runtime config mismatch"):
        validate_checkpoint_artifacts(
            checkpoint,
            config_path,
            expected_runtime_env=artifact["env"],
            allowed_runtime_env_mismatches=("typo",),
        )

    with pytest.raises(ArtifactError, match="unsupported allowed runtime config mismatch"):
        validate_checkpoint_artifacts(
            checkpoint,
            config_path,
            expected_runtime_env=artifact["env"],
            allowed_runtime_env_mismatches=("constraint_solver",),
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


def test_validated_rsl_loader_uses_weights_only_mode(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 44, "num_actions": 4},
        reward={},
        robot={},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    payload = {
        "actor_state_dict": {"weight": torch.ones(2)},
        "critic_state_dict": {"weight": torch.ones(2)},
        "iter": 0,
    }
    torch.save(payload, checkpoint)
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(
        checkpoint,
        training_config=artifact,
        executor_action_contract="cartesian_delta",
    )
    real_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    loaded, validated, manifest = validate_and_load_rsl_checkpoint(
        checkpoint,
        config_path,
        map_location="cpu",
        expected_task="pick_place",
    )

    assert loaded["iter"] == 0
    assert validated["config_sha256"] == manifest.config_sha256
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_validated_rsl_loader_rejects_config_before_deserialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 44, "num_actions": 4},
        reward={},
        robot={},
        train={},
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save(
        {"actor_state_dict": {}, "critic_state_dict": {}, "iter": 0},
        checkpoint,
    )
    artifact = load_training_config(config_path)
    write_checkpoint_manifest(
        checkpoint,
        training_config=artifact,
        executor_action_contract="cartesian_delta",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("num_obs: 44", "num_obs: 45"),
        encoding="utf-8",
    )
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before artifact validation")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ArtifactError, match="training config hash mismatch"):
        validate_and_load_rsl_checkpoint(checkpoint, config_path)
    assert called is False
