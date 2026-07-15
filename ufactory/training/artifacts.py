"""Strict YAML configuration and integrity manifests for RL checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml


class ArtifactError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ArtifactError(f"training configuration contains unsupported type: {type(value).__name__}")


def _finite(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"{path} must be finite")


def _canonical_sha256(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_config(
    path: str | Path,
    *,
    task: str,
    robot_key: str,
    env: Mapping[str, Any],
    reward: Mapping[str, Any],
    robot: Mapping[str, Any],
    train: Mapping[str, Any],
) -> str:
    target = Path(path)
    body = {
        "schema_version": 1,
        "task": str(task),
        "robot_key": str(robot_key),
        "env": _plain(env),
        "reward": _plain(reward),
        "robot": _plain(robot),
        "train": _plain(train),
    }
    _finite(body)
    body["config_sha256"] = _canonical_sha256(body)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return str(body["config_sha256"])


def load_training_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot read training config {source}: {exc}") from exc
    required = {"schema_version", "task", "robot_key", "env", "reward", "robot", "train", "config_sha256"}
    if not isinstance(data, dict) or set(data) != required:
        raise ArtifactError(f"training config fields must be exactly {sorted(required)}")
    if data["schema_version"] != 1:
        raise ArtifactError("unsupported training config schema")
    _finite(data)
    claimed = str(data.pop("config_sha256"))
    actual = _canonical_sha256(data)
    data["config_sha256"] = claimed
    if claimed != actual:
        raise ArtifactError("training config hash mismatch")
    for section in ("env", "reward", "robot", "train"):
        if not isinstance(data[section], dict):
            raise ArtifactError(f"training config {section} must be a mapping")
    return data


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    task: str
    robot_key: str
    dof: int
    observation_dim: int
    action_dim: int
    executor_action_contract: str
    config_sha256: str
    checkpoint_sha256: str
    code_version: str
    checkpoint_file: str


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    training_config: Mapping[str, Any],
    executor_action_contract: str,
    output_path: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise ArtifactError(f"checkpoint does not exist: {checkpoint}")
    env = training_config["env"]
    robot = training_config["robot"]
    manifest = CheckpointManifest(
        schema_version=1,
        task=str(training_config["task"]),
        robot_key=str(training_config["robot_key"]),
        dof=int(env.get("num_actions", len(robot.get("joint_names", ())))),
        observation_dim=int(env["num_obs"]),
        action_dim=int(env["num_actions"]),
        executor_action_contract=str(executor_action_contract),
        config_sha256=str(training_config["config_sha256"]),
        checkpoint_sha256=_file_sha256(checkpoint),
        code_version=_code_version(),
        checkpoint_file=checkpoint.name,
    )
    target = Path(output_path) if output_path is not None else checkpoint.with_suffix(".checkpoint_manifest.json")
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_artifact_inventory(
    path: str | Path,
    *,
    training_config: Mapping[str, Any],
    checkpoints: list[Path],
    selected_checkpoint: Path | None = None,
) -> Path:
    """Write the complete resolved config and deterministic checkpoint inventory."""

    body = {
        "schema_version": 1,
        "training_config": _plain(training_config),
        "checkpoints": [
            {
                "file": checkpoint.name,
                "sha256": _file_sha256(checkpoint),
                "manifest": checkpoint.with_suffix(".checkpoint_manifest.json").name,
            }
            for checkpoint in checkpoints
        ],
        "selected_checkpoint": None if selected_checkpoint is None else selected_checkpoint.name,
    }
    _finite(body)
    target = Path(path)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return target


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read checkpoint manifest {source}: {exc}") from exc
    required = set(CheckpointManifest.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != required:
        raise ArtifactError(f"checkpoint manifest fields must be exactly {sorted(required)}")
    manifest = CheckpointManifest(**data)
    if manifest.schema_version != 1 or len(manifest.checkpoint_sha256) != 64:
        raise ArtifactError("invalid checkpoint manifest schema or hash")
    return manifest


def validate_checkpoint_artifacts(
    checkpoint_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    expected_task: str | None = None,
    expected_robot_key: str | None = None,
    expected_runtime_config_sha256: str | None = None,
) -> tuple[dict[str, Any], CheckpointManifest]:
    checkpoint = Path(checkpoint_path)
    config = load_training_config(config_path)
    path = Path(manifest_path) if manifest_path else checkpoint.with_suffix(".checkpoint_manifest.json")
    manifest = load_checkpoint_manifest(path)
    failures: list[str] = []
    if _file_sha256(checkpoint) != manifest.checkpoint_sha256:
        failures.append("checkpoint hash")
    if config["config_sha256"] != manifest.config_sha256:
        failures.append("config hash")
    if manifest.checkpoint_file != checkpoint.name:
        failures.append("checkpoint filename")
    if expected_task is not None and manifest.task != expected_task:
        failures.append("task")
    if expected_robot_key is not None and manifest.robot_key != expected_robot_key:
        failures.append("robot")
    env = config["env"]
    if expected_runtime_config_sha256 is not None and env.get("runtime_config_sha256") != str(
        expected_runtime_config_sha256
    ):
        failures.append("runtime config hash")
    if int(env["num_obs"]) != manifest.observation_dim or int(env["num_actions"]) != manifest.action_dim:
        failures.append("observation/action dimensions")
    if failures:
        raise ArtifactError(f"checkpoint artifact mismatch: {', '.join(failures)}")
    return config, manifest
