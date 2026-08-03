"""Deterministic pick-place scenario banks for independent evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ufactory.training.artifacts import ArtifactError


def generate_pick_place_scenario_bank(
    *,
    count: int,
    seed: int,
    mode: str,
    runtime_config_sha256: str,
    fixed_obj: list[float],
    fixed_target: list[float],
    obj_spawn_lower: list[float],
    obj_spawn_upper: list[float],
    target_spawn_lower: list[float],
    target_spawn_upper: list[float],
) -> dict[str, Any]:
    """Generate a deterministic fixed, uniform, or boundary-focused bank.

    ``stageN_uniform`` and ``stageN_edge`` use the same spatial envelope as
    curriculum stages 1-3, while plain ``uniform``/``edge`` cover the full
    configured range.
    """

    if count <= 0:
        raise ValueError("scenario count must be positive")
    supported_modes = {
        "fixed",
        "uniform",
        "edge",
        "stage1_uniform",
        "stage1_edge",
        "stage2_uniform",
        "stage2_edge",
        "stage3_uniform",
        "stage3_edge",
    }
    if mode not in supported_modes:
        raise ValueError(f"scenario mode must be one of {sorted(supported_modes)}")

    fixed_obj_np = np.asarray(fixed_obj, dtype=np.float64)
    fixed_target_np = np.asarray(fixed_target, dtype=np.float64)
    obj_lower = np.asarray(obj_spawn_lower, dtype=np.float64)
    obj_upper = np.asarray(obj_spawn_upper, dtype=np.float64)
    target_lower = np.asarray(target_spawn_lower, dtype=np.float64)
    target_upper = np.asarray(target_spawn_upper, dtype=np.float64)
    for name, value in {
        "fixed_obj": fixed_obj_np,
        "fixed_target": fixed_target_np,
        "obj_spawn_lower": obj_lower,
        "obj_spawn_upper": obj_upper,
        "target_spawn_lower": target_lower,
        "target_spawn_upper": target_upper,
    }.items():
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain three finite values")
    if np.any(obj_lower > obj_upper) or np.any(target_lower > target_upper):
        raise ValueError("scenario lower bounds must not exceed upper bounds")

    base_mode = mode
    if mode.startswith("stage"):
        stage_text, base_mode = mode.split("_", maxsplit=1)
        stage = int(stage_text.removeprefix("stage"))
        if stage == 1:
            obj_half_span = target_half_span = np.full(3, 0.01, dtype=np.float64)
        else:
            scale = 0.25 if stage == 2 else 0.5
            obj_half_span = scale * (obj_upper - obj_lower)
            target_half_span = scale * (target_upper - target_lower)
        obj_lower = np.maximum(obj_lower, fixed_obj_np - obj_half_span)
        obj_upper = np.minimum(obj_upper, fixed_obj_np + obj_half_span)
        target_lower = np.maximum(target_lower, fixed_target_np - target_half_span)
        target_upper = np.minimum(target_upper, fixed_target_np + target_half_span)

    rng = np.random.default_rng(seed)
    if base_mode == "fixed":
        objects = np.repeat(fixed_obj_np[None, :], count, axis=0)
        targets = np.repeat(fixed_target_np[None, :], count, axis=0)
    elif base_mode == "uniform":
        objects = rng.uniform(obj_lower, obj_upper, size=(count, 3))
        targets = rng.uniform(target_lower, target_upper, size=(count, 3))
    else:
        # Cycle through every XY corner, then add a tiny deterministic interior
        # jitter so repeated cycles remain distinct while staying near boundaries.
        corners = np.asarray(
            [
                [ox, oy, tx, ty]
                for ox in (obj_lower[0], obj_upper[0])
                for oy in (obj_lower[1], obj_upper[1])
                for tx in (target_lower[0], target_upper[0])
                for ty in (target_lower[1], target_upper[1])
            ],
            dtype=np.float64,
        )
        selected = corners[np.arange(count) % len(corners)]
        objects = np.repeat(fixed_obj_np[None, :], count, axis=0)
        targets = np.repeat(fixed_target_np[None, :], count, axis=0)
        objects[:, :2] = selected[:, :2]
        targets[:, :2] = selected[:, 2:]
        if count > len(corners):
            obj_jitter = rng.uniform(-0.001, 0.001, size=(count, 2))
            target_jitter = rng.uniform(-0.001, 0.001, size=(count, 2))
            objects[:, :2] = np.clip(objects[:, :2] + obj_jitter, obj_lower[:2], obj_upper[:2])
            targets[:, :2] = np.clip(targets[:, :2] + target_jitter, target_lower[:2], target_upper[:2])

    # Object and target rest heights are runtime-owned constants.
    objects[:, 2] = fixed_obj_np[2]
    targets[:, 2] = fixed_target_np[2]
    scenarios = [
        {
            "id": index,
            "object_position_m": objects[index].tolist(),
            "target_position_m": targets[index].tolist(),
        }
        for index in range(count)
    ]
    return {
        "schema_version": 1,
        "task": "pick_place",
        "seed": int(seed),
        "mode": mode,
        "count": count,
        "runtime_config_sha256": str(runtime_config_sha256),
        "scenarios": scenarios,
    }


def write_scenario_bank(path: str | Path, bank: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_scenario_bank(target)
    return target


def load_scenario_bank(
    path: str | Path,
    *,
    expected_runtime_config_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        bank = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read scenario bank {source}: {exc}") from exc
    required = {
        "schema_version",
        "task",
        "seed",
        "mode",
        "count",
        "runtime_config_sha256",
        "scenarios",
    }
    if not isinstance(bank, dict) or set(bank) != required:
        raise ArtifactError(f"scenario bank fields must be exactly {sorted(required)}")
    if bank["schema_version"] != 1 or bank["task"] != "pick_place":
        raise ArtifactError("unsupported scenario bank schema or task")
    if expected_runtime_config_sha256 is not None and bank["runtime_config_sha256"] != str(
        expected_runtime_config_sha256
    ):
        raise ArtifactError("scenario bank runtime config hash mismatch")
    scenarios = bank["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != int(bank["count"]) or not scenarios:
        raise ArtifactError("scenario bank count does not match its scenario list")
    for expected_id, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict) or set(scenario) != {
            "id",
            "object_position_m",
            "target_position_m",
        }:
            raise ArtifactError(f"invalid scenario entry {expected_id}")
        if scenario["id"] != expected_id:
            raise ArtifactError("scenario IDs must be contiguous and ordered")
        for key in ("object_position_m", "target_position_m"):
            vector = scenario[key]
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or not all(isinstance(value, int | float) and math.isfinite(value) for value in vector)
            ):
                raise ArtifactError(f"scenario {expected_id} has invalid {key}")
    return bank


def scenario_bank_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
