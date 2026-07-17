"""Strict RL recipe and runtime-binding tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from ufactory.config import load_runtime_config
from ufactory.training import (
    ArtifactError,
    build_pick_place_task_configs,
    build_train_config,
    load_training_recipe,
)

GRASP_RECIPE = PROJECT_ROOT / "tests" / "fixtures" / "training" / "pick_place_recipe.yaml"


def test_pick_place_binds_the_resolved_runtime_hash() -> None:
    grasp_env, _reward, _robot = build_pick_place_task_configs("xarm6", recipe_path=GRASP_RECIPE)

    assert grasp_env["runtime_config_sha256"] == load_runtime_config("xarm6", task="pick_place").sha256
    assert grasp_env["num_obs"] == 30
    assert grasp_env["num_actions"] == 4
    assert grasp_env["lift_height_m"] == pytest.approx(0.08)
    assert grasp_env["workspace_lower"] == pytest.approx([0.10, -0.35, -0.02])
    assert grasp_env["workspace_upper"] == pytest.approx([0.65, 0.35, 0.55])


def test_pick_place_rl_rejects_non_xarm6_robots() -> None:
    with pytest.raises(ValueError, match="only xArm6"):
        build_pick_place_task_configs("lite6", recipe_path=GRASP_RECIPE)


def test_command_line_train_settings_override_recipe() -> None:
    train = build_train_config(GRASP_RECIPE, experiment_name="override", max_iterations=3)
    assert train["runner"]["experiment_name"] == "override"
    assert train["runner"]["max_iterations"] == 3
    assert train["algorithm"]["gamma"] == pytest.approx(0.99)


def test_recipe_rejects_unknown_top_level_and_runtime_owned_fields(tmp_path: Path) -> None:
    recipe = load_training_recipe(GRASP_RECIPE)
    recipe["unknown"] = {}
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(_yaml(recipe), encoding="utf-8")
    with pytest.raises(ArtifactError, match="fields must be exactly"):
        load_training_recipe(unknown)

    recipe.pop("unknown")
    recipe["environment"]["object_mass_kg"] = 0.1
    physical = tmp_path / "physical.yaml"
    physical.write_text(_yaml(recipe), encoding="utf-8")
    with pytest.raises(ArtifactError, match="ResolvedRuntimeConfig"):
        load_training_recipe(physical)


def test_recipe_rejects_non_finite_values(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "schema_version: 1\nenvironment: {num_envs: 1}\nreward: {keypoints: .nan}\ntrain: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="must be finite"):
        load_training_recipe(invalid)


def _yaml(value: object) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False)
