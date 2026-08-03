"""Repository-level contracts for the curated pick-place RL examples."""

from __future__ import annotations

import json

from conftest import PROJECT_ROOT
import pytest
from ufactory.training import (
    load_scenario_bank,
    load_training_config,
    scenario_bank_sha256,
    validate_checkpoint_artifacts,
)
from ufactory.training import load_training_recipe


EXAMPLE = PROJECT_ROOT / "examples" / "rl" / "pick_place"
BUNDLE = EXAMPLE / "pretrained"
RANDOM_START = EXAMPLE / "random_start"
RANDOM_BUNDLE = RANDOM_START / "pretrained"


@pytest.mark.parametrize(
    ("key_path", "expected"),
    (
        (("environment", "fixed_demo_layout"), True),
        (("environment", "include_commanded_gap"), True),
        (("environment", "include_previous_action"), True),
        (("environment", "include_quality_observations"), True),
        (("environment", "include_contact_observations"), True),
        (("environment", "include_normalized_layout_offsets"), False),
        (("environment", "privileged_critic_obs"), True),
        (("environment", "strict_action_bounds"), True),
        (("environment", "num_obs"), 44),
        (("environment", "num_actions"), 4),
        (("environment", "action_scale"), 0.005),
        (("environment", "gripper_min_command_gap_m"), 0.012),
        (("reward", "valid_release"), 200.0),
        (("reward", "pre_lift_xy_progress"), 200.0),
        (("reward", "grasp_centering"), 60.0),
        (("reward", "transport_progress"), 40.0),
        (("reward", "success"), 100.0),
        (("train", "algorithm", "class_name"), "PPO"),
        (("train", "actor", "class_name"), "MLPModel"),
        (("train", "critic", "class_name"), "MLPModel"),
        (("train", "runner", "max_iterations"), 300),
        (("train", "num_steps_per_env"), 128),
    ),
)
def test_canonical_recipe_release_contract(
    key_path: tuple[str, ...],
    expected: object,
) -> None:
    value = load_training_recipe(EXAMPLE / "recipe.yaml")
    for key in key_path:
        value = value[key]
    assert value == expected


@pytest.mark.parametrize(
    ("key_path", "expected"),
    (
        (("environment", "fixed_demo_layout"), False),
        (("environment", "randomize_target"), False),
        (("environment", "include_normalized_layout_offsets"), True),
        (("environment", "include_scripted_action_hint"), True),
        (
            ("environment", "scripted_action_hint_controller"),
            "scripted_pick_place_expert_v2",
        ),
        (("environment", "scripted_action_hint_config", "close_gap_m"), 0.018),
        (("environment", "scripted_action_hint_config", "landing_brake_speed_m_s"), 0.025),
        (("environment", "scripted_action_hint_config", "landing_brake_max_step_m"), 0.0005),
        (("environment", "scripted_action_hint_config", "release_hover_m"), 0.00002),
        (("environment", "scripted_action_hint_config", "settle_dwell_steps"), 2),
        (("environment", "scripted_action_hint_config", "release_open_step_m"), 0.0005),
        (("environment", "num_obs"), 54),
        (("environment", "curriculum_min_stage_steps"), 1280),
        (("environment", "curriculum_stage0_grasp_rate"), 0.92),
        (("environment", "curriculum_stage1_grasp_rate"), 0.90),
        (("environment", "curriculum_stage2_place_rate"), 0.65),
        (("environment", "curriculum_stage3_place_rate"), 0.60),
        (("train", "algorithm", "learning_rate"), 3e-4),
        (("train", "actor", "class_name"), "ufactory.training.models:GuidedPickPlaceMLPModel"),
        (("train", "actor", "source_observation_dim"), 44),
        (("train", "actor", "layout_offset_start"), 44),
        (("train", "actor", "guide_action_start"), 50),
        (("train", "actor", "obs_normalization"), False),
        (
            ("train", "runner", "observation_projection_initializer"),
            "scripted_action_hint_actor_v1",
        ),
        (("train", "runner", "observation_projection_train_scope"), "frozen_guided_actor"),
    ),
)
def test_random_start_recipe_release_contract(
    key_path: tuple[str, ...],
    expected: object,
) -> None:
    value = load_training_recipe(RANDOM_START / "recipe.yaml")
    for key in key_path:
        value = value[key]
    assert value == expected


def test_public_rl_tree_contains_only_the_canonical_release_assets() -> None:
    assert sorted(path.name for path in EXAMPLE.glob("*.yaml")) == ["recipe.yaml"]
    assert sorted(path.name for path in (EXAMPLE / "scenarios").glob("*.json")) == ["fixed_seed17000_n512.json"]
    assert sorted(path.name for path in BUNDLE.iterdir()) == [
        "config.yaml",
        "evaluation_summary.json",
        "model_199.checkpoint_manifest.json",
        "model_199.pt",
    ]
    assert sorted(path.name for path in RANDOM_START.glob("*.yaml")) == ["recipe.yaml"]
    assert sorted(path.name for path in RANDOM_START.glob("*.md")) == ["README.md", "README_cn.md"]
    assert sorted(path.name for path in (RANDOM_START / "scenarios").glob("*.json")) == [
        "object_edge_seed31213_n256.json",
        "object_uniform_seed31212_n512.json",
    ]
    assert sorted(path.name for path in RANDOM_BUNDLE.iterdir()) == [
        "config.yaml",
        "evaluation_edge_seed31213_n256.json",
        "evaluation_uniform_seed31212_n512.json",
        "model_0.checkpoint_manifest.json",
        "model_0.pt",
    ]
    assert not any(EXAMPLE.rglob("*.sh"))


def test_public_entries_are_module_only_and_never_use_unsafe_rsl_loading() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLE.glob("*.py"))
    assert "sys.path" not in combined
    assert "weights_only=False" not in combined
    assert ".load(str(" not in combined
    assert 'Path("outputs") / "rl" / "pick_place"' in combined
    assert 'DEFAULT_CHECKPOINT_DIR = EXAMPLE_DIR / "pretrained"' in combined


def test_bundled_checkpoint_hashes_validate_without_writing_to_bundle() -> None:
    checkpoint = BUNDLE / "model_199.pt"
    config = BUNDLE / "config.yaml"
    before = {path.name: path.stat().st_mtime_ns for path in BUNDLE.iterdir()}

    artifact, manifest = validate_checkpoint_artifacts(
        checkpoint,
        config,
        expected_task="pick_place",
        expected_robot_key="xarm6_1305",
        expected_runtime_config_sha256=("5f924f41a54b789f2653d71fe993ebf5bffbc5ae273de8429f75dd56bac5289c"),
    )

    assert artifact["config_sha256"] == ("cc8590c6cb029b27969efc01f315e6ad9de66d3475d5dc72f67832052406508a")
    assert manifest.checkpoint_sha256 == ("9381387a30efbbc102289052569c0804c868db8d5d5eb5332b88a7513f61b0f7")
    assert checkpoint.stat().st_size == 2_699_115
    assert before == {path.name: path.stat().st_mtime_ns for path in BUNDLE.iterdir()}


def test_bundled_config_is_sanitized_and_summary_discloses_rebaselined_robustness() -> None:
    config_text = (BUNDLE / "config.yaml").read_text(encoding="utf-8")
    assert "/home/" not in config_text
    assert "/Users/" not in config_text
    assert "logs/pick_place" not in config_text
    artifact = load_training_config(BUNDLE / "config.yaml")
    assert artifact["robot"]["urdf_path"].startswith("assets/urdf/")
    assert artifact["train"]["runner"]["transfer_checkpoint"] is None

    summary = json.loads((BUNDLE / "evaluation_summary.json").read_text(encoding="utf-8"))
    deterministic = summary["evaluation"]["deterministic_gate"]
    noise = summary["evaluation"]["fixed_action_noise_bank"]
    assert deterministic["seed_batch_matrix"] == "9/9"
    assert deterministic["seed_batch_matrix_episodes"] == "219/219"
    assert deterministic["fixed_64"] == "64/64"
    assert noise["success"] == "512/512"
    assert noise["quality_success"] == "512/512"
    assert noise["success_rate"] == 1.0
    assert noise["post_release_recontact_count"] == 120
    assert noise["post_release_recontact_is_gate"] is False
    assert noise["robustness_target"] == 0.99
    assert noise["robustness_target_pass"] is True


def test_random_start_bundle_is_sanitized_validated_and_discloses_guided_policy() -> None:
    checkpoint = RANDOM_BUNDLE / "model_0.pt"
    config = RANDOM_BUNDLE / "config.yaml"
    before = {path.name: path.stat().st_mtime_ns for path in RANDOM_BUNDLE.iterdir()}

    artifact, manifest = validate_checkpoint_artifacts(
        checkpoint,
        config,
        expected_task="pick_place",
        expected_robot_key="xarm6_1305",
        expected_runtime_config_sha256=("5f924f41a54b789f2653d71fe993ebf5bffbc5ae273de8429f75dd56bac5289c"),
    )

    assert artifact["config_sha256"] == ("dd4be840e15a3c2e30d2d6d65249b825f1edbcb76e2f81ae3616f3ccfc70ef76")
    assert manifest.checkpoint_sha256 == ("eb0b0d61c978e12600442243db0ad1ecc807da04303b92eda587c03732b02c33")
    assert checkpoint.stat().st_size == 1_832_991
    assert artifact["env"]["num_obs"] == 54
    assert artifact["env"]["scripted_action_hint_controller"] == "scripted_pick_place_expert_v2"
    assert artifact["train"]["actor"]["class_name"] == "ufactory.training.models:GuidedPickPlaceMLPModel"
    assert "/home/" not in config.read_text(encoding="utf-8")
    assert "/tmp/" not in config.read_text(encoding="utf-8")
    assert before == {path.name: path.stat().st_mtime_ns for path in RANDOM_BUNDLE.iterdir()}

    for name, count, mode in (
        ("evaluation_uniform_seed31212_n512.json", 512, "object_uniform"),
        ("evaluation_edge_seed31213_n256.json", 256, "object_edge"),
    ):
        summary = json.loads((RANDOM_BUNDLE / name).read_text(encoding="utf-8"))
        assert summary["checkpoint"] == "model_0.pt"
        assert summary["checkpoint_sha256"] == manifest.checkpoint_sha256
        assert summary["episodes"] == count
        assert summary["scenario_bank"]["mode"] == mode
        assert summary["outcomes"]["success"]["count"] == count
        assert summary["outcomes"]["success"]["rate"] == 1.0
        assert summary["outcomes"]["quality"]["count"] == count
        assert summary["quality"]["standard_target_pass"] is True
        assert summary["quality"]["robustness_target_pass"] is True
        assert summary["diagnostics"]["max_action_clip_fraction"] == 0.0
        assert summary["diagnostics"]["max_ik_failure_fraction"] == 0.0
        assert summary["diagnostics"]["post_release_recontact_count"] == 0


def test_random_start_final_scenario_banks_are_object_only_unique_and_runtime_bound() -> None:
    env = load_training_config(RANDOM_BUNDLE / "config.yaml")["env"]
    expected = (
        (
            "object_uniform_seed31212_n512.json",
            512,
            "object_uniform",
            "ace5b85ac7f5d7cfe5b9603ff57cbccd183aa8dde72d1d289f7cceb2fbbb3c90",
        ),
        (
            "object_edge_seed31213_n256.json",
            256,
            "object_edge",
            "ba063fac720ba38855902f683df7374947f9c51c7643859e4a022ea856853f96",
        ),
    )
    for name, count, mode, digest in expected:
        path = RANDOM_START / "scenarios" / name
        bank = load_scenario_bank(
            path,
            expected_runtime_config_sha256=env["runtime_config_sha256"],
            expected_env=env,
        )
        assert bank["mode"] == mode
        assert bank["count"] == count
        assert len({tuple(item["object_position_m"]) for item in bank["scenarios"]}) == count
        assert {tuple(item["target_position_m"]) for item in bank["scenarios"]} == {tuple(env["fixed_target_pos"])}
        assert scenario_bank_sha256(path) == digest


def test_published_scenario_bank_is_fixed_complete_and_runtime_bound() -> None:
    path = EXAMPLE / "scenarios" / "fixed_seed17000_n512.json"
    bank = load_scenario_bank(
        path,
        expected_runtime_config_sha256=("5f924f41a54b789f2653d71fe993ebf5bffbc5ae273de8429f75dd56bac5289c"),
    )
    assert bank["mode"] == "fixed"
    assert bank["count"] == 512
    assert len(bank["scenarios"]) == 512
    assert scenario_bank_sha256(path) == ("eb4f8514539fd94e53181b2f69996b74a0bc7049389dadb1410cb32e57af92bd")
