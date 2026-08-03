"""Repository-level contract for the curated fixed-layout RL example."""

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


def test_public_rl_tree_contains_only_the_canonical_release_assets() -> None:
    assert sorted(path.name for path in EXAMPLE.glob("*.yaml")) == ["recipe.yaml"]
    assert sorted(path.name for path in (EXAMPLE / "scenarios").glob("*.json")) == ["fixed_seed17000_n512.json"]
    assert sorted(path.name for path in BUNDLE.iterdir()) == [
        "config.yaml",
        "evaluation_summary.json",
        "model_199.checkpoint_manifest.json",
        "model_199.pt",
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
        expected_runtime_config_sha256=("3200dea21d28acac2384e658d640e00f1348ee1cad8337e061db6eefa699f529"),
    )

    assert artifact["config_sha256"] == ("a86fdc6450f6d8c8c8c6a75cc19a0d557ba470e2e7d03eabe437fad3f61ecdb8")
    assert manifest.checkpoint_sha256 == ("9381387a30efbbc102289052569c0804c868db8d5d5eb5332b88a7513f61b0f7")
    assert checkpoint.stat().st_size == 2_699_115
    assert before == {path.name: path.stat().st_mtime_ns for path in BUNDLE.iterdir()}


def test_bundled_config_is_sanitized_and_summary_discloses_failed_robustness_goal() -> None:
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
    assert deterministic["fixed_64"] == "64/64"
    assert noise["success"] == "442/512"
    assert noise["success_rate"] == 442 / 512
    assert noise["robustness_target"] == 0.99
    assert noise["robustness_target_pass"] is False


def test_published_scenario_bank_is_fixed_complete_and_runtime_bound() -> None:
    path = EXAMPLE / "scenarios" / "fixed_seed17000_n512.json"
    bank = load_scenario_bank(
        path,
        expected_runtime_config_sha256=("3200dea21d28acac2384e658d640e00f1348ee1cad8337e061db6eefa699f529"),
    )
    assert bank["mode"] == "fixed"
    assert bank["count"] == 512
    assert len(bank["scenarios"]) == 512
    assert scenario_bank_sha256(path) == ("dd42004205fd354b084050e092cff20dcfbf93ef1eb386c3ab5d457937aab2e5")
