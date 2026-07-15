"""Strict user-editable reinforcement-learning recipe loader."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from ufactory.training.artifacts import ArtifactError

_RUNTIME_OWNED_KEYS = {
    "runtime_config_sha256",
    "ctrl_dt",
    "episode_length_s",
    "object_size_m",
    "object_mass_kg",
    "obj_size",
    "obj_mass_kg",
    "table_height",
    "workspace_lower",
    "workspace_upper",
    "urdf_path",
    "joint_names",
    "default_qpos",
    "kp",
    "kv",
    "force_lower",
    "force_upper",
}


def _validate_finite(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _RUNTIME_OWNED_KEYS:
                raise ArtifactError(f"{path}.{key} is owned by ResolvedRuntimeConfig, not the RL recipe")
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"{path} must be finite")


def load_training_recipe(path: str | Path) -> dict[str, Any]:
    """Load a strict recipe containing only environment, reward, and PPO settings."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot read training recipe {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ArtifactError("training recipe root must be a mapping")
    required = {"schema_version", "environment", "reward", "train"}
    if set(raw) != required or raw.get("schema_version") != 1:
        raise ArtifactError(f"training recipe fields must be exactly {sorted(required)} with schema_version 1")
    for section in ("environment", "reward", "train"):
        if not isinstance(raw[section], dict):
            raise ArtifactError(f"training recipe {section} must be a mapping")
    _validate_finite(raw, "recipe")
    return cast(dict[str, Any], raw)
