"""Strict RL recipe and runtime-binding tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conftest import PROJECT_ROOT
from ufactory.config import load_runtime_config
from ufactory.training import (
    ArtifactError,
    build_pick_place_task_configs,
    build_reach_task_configs,
    build_train_config,
    effective_max_joint_delta_rad,
    load_training_recipe,
    reach_action_delta_torch,
)

REACH_RECIPE = PROJECT_ROOT / "tests" / "fixtures" / "training" / "reach_recipe.yaml"
GRASP_RECIPE = PROJECT_ROOT / "tests" / "fixtures" / "training" / "pick_place_recipe.yaml"


def test_reach_and_pick_place_bind_the_resolved_runtime_hash() -> None:
    reach_env, _reward, _robot = build_reach_task_configs("xarm5", recipe_path=REACH_RECIPE)
    grasp_env, _reward, _robot = build_pick_place_task_configs("xarm6", recipe_path=GRASP_RECIPE)

    assert reach_env["runtime_config_sha256"] == load_runtime_config("xarm5", task="reach").sha256
    assert grasp_env["runtime_config_sha256"] == load_runtime_config("xarm6", task="pick_place").sha256
    assert reach_env["num_obs"] == 16
    assert reach_env["num_actions"] == 5
    assert grasp_env["num_obs"] == 30
    assert grasp_env["num_actions"] == 7
    assert grasp_env["lift_height_m"] == pytest.approx(0.08)
    assert grasp_env["workspace_lower"] == pytest.approx([0.10, -0.35, -0.02])
    assert grasp_env["workspace_upper"] == pytest.approx([0.65, 0.35, 0.55])


def test_pick_place_rl_rejects_non_xarm6_robots() -> None:
    with pytest.raises(ValueError, match="only xArm6"):
        build_pick_place_task_configs("lite6", recipe_path=GRASP_RECIPE)


def test_command_line_train_settings_override_recipe() -> None:
    train = build_train_config(REACH_RECIPE, experiment_name="override", max_iterations=3)
    assert train["runner"]["experiment_name"] == "override"
    assert train["runner"]["max_iterations"] == 3
    assert train["algorithm"]["gamma"] == pytest.approx(0.99)


def test_reach_action_scaling_contract_is_preserved() -> None:
    limit = effective_max_joint_delta_rad(
        action_scale=0.05,
        action_clip=1.0,
        ctrl_dt=0.02,
        servo_speed_rad_s=0.5,
    )
    assert limit == pytest.approx(0.01)
    action = torch.tensor([[-2.0, 0.1, 2.0]])
    delta = reach_action_delta_torch(
        action,
        action_scale=0.05,
        action_clip=1.0,
        max_joint_delta_rad=limit,
    )
    assert delta.tolist()[0] == pytest.approx([-0.01, 0.005, 0.01])


def test_recipe_rejects_unknown_top_level_and_runtime_owned_fields(tmp_path: Path) -> None:
    recipe = load_training_recipe(REACH_RECIPE)
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
        "schema_version: 1\nenvironment: {num_envs: 1}\nreward: {reach: .nan}\ntrain: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="must be finite"):
        load_training_recipe(invalid)


def _yaml(value: object) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False)
