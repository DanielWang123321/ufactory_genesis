"""CPU checks for the bare-arm xArm6 reach task."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ufactory.config import ConfigError, load_runtime_config
from ufactory.training import (
    build_reach_task_configs,
    load_training_config,
    load_training_recipe,
    write_training_config,
)
from ufactory.training.artifacts import runtime_env_contract
from ufactory.training.artifacts import ArtifactError, validate_checkpoint_artifacts, write_checkpoint_manifest
from ufactory.training.logic import (
    action_penalty,
    effective_max_joint_delta_rad,
    reach_action_delta_torch,
    reach_observation,
    reward_reach_position,
)

RECIPE = Path(__file__).resolve().parents[2] / "examples" / "rl" / "reach" / "recipe.yaml"


def test_joint_delta_respects_servo_speed_cap() -> None:
    limit = effective_max_joint_delta_rad(
        action_scale=0.05,
        action_clip=1.0,
        ctrl_dt=0.02,
        servo_speed_rad_s=0.5,
    )
    assert limit == pytest.approx(0.01)
    actions = torch.tensor([[2.0, -2.0, 0.5, 0.0, 1.0, -0.25]])
    delta = reach_action_delta_torch(actions, action_scale=0.05, action_clip=1.0, max_joint_delta_rad=limit)
    assert torch.allclose(delta, torch.tensor([[0.01, -0.01, 0.005, 0.0, 0.01, -0.0025]]))
    ramp = torch.linspace(-1.0, 1.0, 1001).unsqueeze(0)
    delta = reach_action_delta_torch(ramp, action_scale=0.05, action_clip=1.0, max_joint_delta_rad=limit)
    assert torch.all(delta.diff() > 0)
    assert torch.allclose(delta, ramp * limit)


def test_reach_observation_and_position_reward() -> None:
    joint_pos = torch.zeros(2, 6)
    joint_vel = torch.ones(2, 6)
    ee_pos = torch.tensor([[0.2, 0.0, 0.3], [0.2, 0.0, 0.3]])
    target_pos = torch.tensor([[0.2, 0.0, 0.3], [0.3, 0.0, 0.3]])
    command = torch.full_like(joint_pos, 0.015)
    observation = reach_observation(joint_pos, joint_vel, ee_pos, target_pos, command)
    assert observation.shape == (2, 24)
    assert torch.allclose(observation[0, 12:15], ee_pos[0])
    assert torch.allclose(observation[1, 15:18], target_pos[1] - ee_pos[1])
    assert torch.allclose(observation[:, 18:], command - joint_pos)
    reward = reward_reach_position(ee_pos, target_pos)
    assert reward[0].item() == pytest.approx(1.0)
    assert reward[1].item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())
    assert action_penalty(joint_vel).shape == (2,)


def test_reach_runtime_config_uses_the_bare_arm_and_target_box() -> None:
    config = load_runtime_config("xarm6", task="reach")
    assert config.robot.urdf == "xarm6_1305.urdf"
    assert config.task.name == "reach"
    assert config.task.parameters["success_distance_m"] == pytest.approx(0.005)
    assert config.task.parameters["target_position_lower_m"] == (0.15, -0.30, 0.05)
    assert config.task.parameters["target_position_upper_m"] == (0.55, 0.30, 0.50)
    assert "assets/configs/runtime/tasks/robots/xarm6_1305_reach.yaml" in config.sources


def test_reach_task_config_is_xarm6_only_and_round_trips() -> None:
    with pytest.raises(ValueError, match="only xArm6"):
        build_reach_task_configs("xarm5", recipe_path=RECIPE)
    env_cfg, reward_cfg, robot_cfg = build_reach_task_configs("xarm6", recipe_path=RECIPE)
    assert env_cfg["task_name"] == "reach"
    assert env_cfg["env_version"] == 2
    assert env_cfg["num_obs"] == 24
    assert env_cfg["num_actions"] == 6
    assert env_cfg["max_joint_delta_rad"] == pytest.approx(0.01)
    assert robot_cfg["urdf_path"].endswith("assets/urdf/xarm6/xarm6_1305.urdf")
    assert reward_cfg["reach"] == pytest.approx(1.0)
    reach_contract = runtime_env_contract(env_cfg)
    pick_place_contract = runtime_env_contract({"workspace_upper": [0.65, 0.35, 0.55]})
    assert "max_joint_delta_rad" in reach_contract
    assert "workspace_upper" not in reach_contract
    assert "workspace_upper" in pick_place_contract
    assert "success_distance_m" not in pick_place_contract


def test_reach_training_config_keeps_the_runtime_contract(tmp_path: Path) -> None:
    env_cfg, reward_cfg, robot_cfg = build_reach_task_configs("xarm6", recipe_path=RECIPE)
    train_cfg = {"seed": 1}
    target = tmp_path / "config.yaml"
    write_training_config(
        target,
        task="reach",
        robot_key="xarm6_1305",
        env=env_cfg,
        reward=reward_cfg,
        robot=robot_cfg,
        train=train_cfg,
    )
    loaded = load_training_config(target)
    assert loaded["task"] == "reach"
    assert runtime_env_contract(loaded["env"]) == runtime_env_contract(env_cfg)


def test_reach_recipe_has_the_three_training_sections() -> None:
    recipe = load_training_recipe(RECIPE)
    assert set(recipe) == {"schema_version", "environment", "reward", "train"}
    assert recipe["environment"]["num_envs"] == 16384
    assert recipe["environment"]["genesis_performance_mode"] is True
    assert recipe["train"]["algorithm"]["learning_rate"] == pytest.approx(3.0e-4)
    assert recipe["train"]["runner"]["max_iterations"] == 300
    assert recipe["train"]["num_steps_per_env"] == 24
    assert recipe["train"]["seed"] == 7
    assert recipe["reward"]["success"] == 2.0
    assert recipe["train"]["algorithm"]["schedule"] == "fixed"
    assert recipe["train"]["algorithm"]["entropy_coef"] == pytest.approx(0.001)
    assert recipe["train"]["save_interval"] == 25
    assert recipe["train"]["actor"]["obs_normalization"] is True
    assert recipe["train"]["critic"]["obs_normalization"] is True
    assert recipe["train"]["actor"]["distribution_cfg"] == {
        "class_name": "BetaDistribution",
        "action_range": [-1.0, 1.0],
    }


@pytest.mark.parametrize("distance", [0.004, 0.006, 0.01])
def test_reach_rejects_non_5mm_success_distance(tmp_path: Path, distance: float) -> None:
    overlay = tmp_path / "runtime.yaml"
    overlay.write_text(f"schema_version: 1\ntask:\n  parameters:\n    success_distance_m: {distance}\n")
    with pytest.raises(ValueError, match="success_distance_m must be 0.005"):
        build_reach_task_configs("xarm6", recipe_path=RECIPE, runtime_config_path=overlay)


def test_legacy_reach_checkpoint_requests_retraining(tmp_path: Path) -> None:
    env, reward, robot = build_reach_task_configs("xarm6", recipe_path=RECIPE)
    legacy = dict(env, num_obs=18)
    legacy.pop("env_version")
    config = tmp_path / "config.yaml"
    write_training_config(
        config, task="reach", robot_key="xarm6_1305", env=legacy, reward=reward, robot=robot, train={"seed": 1}
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": {}, "critic_state_dict": {}}, checkpoint)
    write_checkpoint_manifest(
        checkpoint,
        training_config=load_training_config(config),
        executor_action_contract="normalized_joint_position_delta",
    )
    with pytest.raises(ArtifactError, match="retrain"):
        validate_checkpoint_artifacts(checkpoint, config, expected_task="reach", expected_runtime_env=env)


def test_reach_checkpoint_rejects_non_5mm_success_distance(tmp_path: Path) -> None:
    env, reward, robot = build_reach_task_configs("xarm6", recipe_path=RECIPE)
    env["success_distance_m"] = 0.01
    config = tmp_path / "config.yaml"
    write_training_config(
        config, task="reach", robot_key="xarm6_1305", env=env, reward=reward, robot=robot, train={"seed": 7}
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": {}, "critic_state_dict": {}}, checkpoint)
    write_checkpoint_manifest(
        checkpoint,
        training_config=load_training_config(config),
        executor_action_contract="normalized_joint_position_delta",
    )
    with pytest.raises(ArtifactError, match="success_distance_m.*5 mm"):
        validate_checkpoint_artifacts(checkpoint, config, expected_task="reach")


def test_native_beta_distribution_bounds_and_finite_likelihood() -> None:
    distribution_module = pytest.importorskip("rsl_rl.modules.distribution")
    distribution = distribution_module.BetaDistribution(output_dim=6, action_range=(-1.0, 1.0))
    parameters = torch.zeros(4096, 2, 6)
    distribution.update(parameters)
    actions = distribution.sample()
    assert torch.all((actions >= -1.0) & (actions <= 1.0))
    assert torch.isfinite(distribution.log_prob(actions)).all()
    assert torch.equal(distribution.deterministic_output(parameters), torch.zeros(4096, 6))


@pytest.mark.parametrize(
    ("text", "match"),
    (
        ("schema_version: 1\ntask:\n  parameters:\n    success_distance_m: 0\n", "success_distance_m"),
        (
            "schema_version: 1\ntask:\n  parameters:\n    target_position_lower_m: [0.6, 0.0, 0.1]\n"
            "    target_position_upper_m: [0.2, 0.1, 0.2]\n",
            "target_position_lower_m",
        ),
    ),
)
def test_invalid_reach_parameters_are_rejected(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_runtime_config("xarm6", task="reach", config_path=path)
