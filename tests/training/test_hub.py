"""Offline tests for HuggingFace checkpoint hub helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.training.artifacts import (
    ArtifactError,
    load_training_config,
    validate_checkpoint_artifacts,
    write_checkpoint_manifest,
    write_training_config,
)
from ufactory.training.hub import collect_checkpoint_bundle, upload_checkpoint_bundle


def _write_fake_bundle(tmp_path: Path) -> tuple[Path, Path]:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    checkpoint = log_dir / "model_10.pt"
    checkpoint.write_bytes(b"fake-checkpoint-bytes")
    write_training_config(
        log_dir / "config.yaml",
        task="pick_place",
        robot_key="xarm6_1305",
        env={"num_obs": 18, "num_actions": 6, "runtime_config_sha256": "a" * 64},
        reward={"keypoints": 1.0},
        robot={"joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
        train={"seed": 1},
    )
    (log_dir / "artifacts.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    config = load_training_config(log_dir / "config.yaml")
    write_checkpoint_manifest(
        checkpoint,
        training_config=config,
        executor_action_contract="servo_j_joint_delta_rad",
    )
    return log_dir, checkpoint


def test_collect_checkpoint_bundle_lists_existing_files(tmp_path: Path) -> None:
    log_dir, checkpoint = _write_fake_bundle(tmp_path)
    files = collect_checkpoint_bundle(log_dir, checkpoint)
    names = {path.name for path in files}
    assert "model_10.pt" in names
    assert "model_10.checkpoint_manifest.json" in names
    assert "config.yaml" in names
    assert "artifacts.yaml" in names
    (log_dir / "artifacts.yaml").unlink()
    names2 = {path.name for path in collect_checkpoint_bundle(log_dir, checkpoint)}
    assert "artifacts.yaml" not in names2


def test_upload_dry_run_lists_remote_names(tmp_path: Path) -> None:
    log_dir, checkpoint = _write_fake_bundle(tmp_path)
    names = upload_checkpoint_bundle(
        log_dir,
        checkpoint,
        repo_id="org/repo",
        path_in_repo="pick_place/",
        dry_run=True,
    )
    assert names == [
        "pick_place/model_10.pt",
        "pick_place/model_10.checkpoint_manifest.json",
        "pick_place/config.yaml",
        "pick_place/artifacts.yaml",
    ]


def test_validate_detects_tampered_checkpoint(tmp_path: Path) -> None:
    log_dir, checkpoint = _write_fake_bundle(tmp_path)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="checkpoint hash"):
        validate_checkpoint_artifacts(checkpoint, log_dir / "config.yaml")
