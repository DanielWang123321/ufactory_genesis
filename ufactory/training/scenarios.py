"""Deterministic pick-place scenario banks for independent evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ufactory.training.artifacts import ArtifactError


PICK_PLACE_SCENARIO_MODES = (
    "fixed",
    "uniform",
    "edge",
    "object_uniform",
    "object_edge",
    "stage1_uniform",
    "stage1_edge",
    "stage2_uniform",
    "stage2_edge",
    "stage3_uniform",
    "stage3_edge",
)


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
    if mode not in PICK_PLACE_SCENARIO_MODES:
        raise ValueError(f"scenario mode must be one of {PICK_PLACE_SCENARIO_MODES}")

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

    object_only = mode.startswith("object_")
    base_mode = mode.removeprefix("object_") if object_only else mode
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
        if object_only:
            corners = np.asarray(
                [[ox, oy] for ox in (obj_lower[0], obj_upper[0]) for oy in (obj_lower[1], obj_upper[1])],
                dtype=np.float64,
            )
        else:
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
        if not object_only:
            targets[:, :2] = selected[:, 2:]
        if count > len(corners):
            if object_only:
                remainder = slice(len(corners), count)
                inward = np.where(selected[remainder, :2] == obj_lower[:2], 1.0, -1.0)
                magnitude = rng.uniform(1e-9, 0.001, size=(count - len(corners), 2))
                objects[remainder, :2] = selected[remainder, :2] + inward * magnitude
            else:
                obj_jitter = rng.uniform(-0.001, 0.001, size=(count, 2))
                objects[:, :2] = np.clip(objects[:, :2] + obj_jitter, obj_lower[:2], obj_upper[:2])
                target_jitter = rng.uniform(-0.001, 0.001, size=(count, 2))
                targets[:, :2] = np.clip(targets[:, :2] + target_jitter, target_lower[:2], target_upper[:2])

    if object_only:
        targets = np.repeat(fixed_target_np[None, :], count, axis=0)

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
    expected_env: Mapping[str, Any] | None = None,
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
    if bank["mode"] not in PICK_PLACE_SCENARIO_MODES:
        raise ArtifactError(f"unsupported pick-place scenario mode: {bank['mode']!r}")
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
    if expected_env is not None:
        _validate_scenario_semantics(bank, expected_env)
    return bank


def _validate_scenario_semantics(bank: Mapping[str, Any], env: Mapping[str, Any]) -> None:
    """Validate positions against the resolved task rather than trusting JSON."""

    required = {
        "fixed_obj_pos",
        "fixed_target_pos",
        "obj_spawn_lower",
        "obj_spawn_upper",
        "target_spawn_lower",
        "target_spawn_upper",
    }
    missing = sorted(required - set(env))
    if missing:
        raise ArtifactError(f"scenario validation environment is missing fields: {', '.join(missing)}")

    vectors = {name: np.asarray(env[name], dtype=np.float64) for name in required}
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in vectors.values()):
        raise ArtifactError("scenario validation environment contains invalid position vectors")
    mode = str(bank["mode"])
    object_positions: list[tuple[float, float, float]] = []
    for scenario in bank["scenarios"]:
        obj = np.asarray(scenario["object_position_m"], dtype=np.float64)
        target = np.asarray(scenario["target_position_m"], dtype=np.float64)
        if np.any(obj < vectors["obj_spawn_lower"] - 1e-12) or np.any(obj > vectors["obj_spawn_upper"] + 1e-12):
            raise ArtifactError(f"scenario {scenario['id']} object position is outside runtime bounds")
        if np.any(target < vectors["target_spawn_lower"] - 1e-12) or np.any(
            target > vectors["target_spawn_upper"] + 1e-12
        ):
            raise ArtifactError(f"scenario {scenario['id']} target position is outside runtime bounds")
        if not math.isclose(obj[2], vectors["fixed_obj_pos"][2], rel_tol=0.0, abs_tol=1e-12):
            raise ArtifactError(f"scenario {scenario['id']} object is not at the runtime rest height")
        if not math.isclose(target[2], vectors["fixed_target_pos"][2], rel_tol=0.0, abs_tol=1e-12):
            raise ArtifactError(f"scenario {scenario['id']} target is not at the runtime rest height")
        if mode == "fixed" and (
            not np.allclose(obj, vectors["fixed_obj_pos"], rtol=0.0, atol=1e-12)
            or not np.allclose(target, vectors["fixed_target_pos"], rtol=0.0, atol=1e-12)
        ):
            raise ArtifactError(f"fixed scenario {scenario['id']} differs from the runtime fixed layout")
        if mode.startswith("object_") and not np.allclose(
            target,
            vectors["fixed_target_pos"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ArtifactError(f"object-only scenario {scenario['id']} randomizes the target")
        object_positions.append(tuple(float(value) for value in obj))
    if mode.startswith("object_") and len(object_positions) > 1 and len(set(object_positions)) != len(object_positions):
        raise ArtifactError("object-only random scenario positions must be unique")


def scenario_bank_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PICK_PLACE_SCENARIO_MODES",
    "generate_pick_place_scenario_bank",
    "load_scenario_bank",
    "scenario_bank_sha256",
    "write_scenario_bank",
]
