"""Repository-level contracts for the curated fixed-layout pick-place RL example."""

from __future__ import annotations

import json

from conftest import PROJECT_ROOT
import pytest

from ufactory.training import (
    load_scenario_bank,
    load_training_config,
    load_training_recipe,
    scenario_bank_sha256,
    validate_checkpoint_artifacts,
)


EXAMPLE = PROJECT_ROOT / "examples" / "rl" / "pick_place"
BUNDLE = EXAMPLE / "pretrained"
CHECKPOINT_SHA256 = "f2142a639e850928310e764464b83706a08b01f111fce5dcb975e456def783f2"
CONFIG_SHA256 = "a02b646e73ac3174c8f30afa476b92c3db3ffde97f0431b7d24ad19b5d149175"
CURRENT_RUNTIME_SHA256 = "654b538ed4062d7d82955ce617fc37a7dc974c6a53d3df0ab0e38c91dd80e7ed"


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
        (("environment", "num_obs"), 48),
        (("environment", "include_scripted_action_hint"), True),
        (("environment", "num_actions"), 4),
        (("environment", "action_scale"), 0.005),
        (("environment", "gripper_min_command_gap_m"), 0.012),
        (("environment", "place_phase_reset_frac"), 0.40),
        (("environment", "release_command_margin_m"), 0.018),
        (("environment", "scripted_action_hint_controller"), "scripted_pick_place_expert_v1"),
        (("environment", "scripted_action_hint_config", "latch_layout_setpoints"), True),
        (("environment", "scripted_action_hint_config", "brake_only_near_table"), True),
        (("environment", "scripted_action_hint_config", "close_gap_m"), 0.022),
        (("environment", "scripted_action_hint_config", "near_table_margin_m"), 0.045),
        (("environment", "scripted_action_hint_config", "near_table_xy_step_m"), 0.0004),
        (("environment", "scripted_action_hint_config", "near_table_z_step_m"), 0.0006),
        (("environment", "scripted_action_hint_config", "release_hover_m"), 0.0001),
        (("reward", "valid_release"), 200.0),
        (("reward", "pre_lift_xy_progress"), 200.0),
        (("reward", "grasp_centering"), 60.0),
        (("reward", "transport_progress"), 40.0),
        (("reward", "success"), 100.0),
        (("train", "algorithm", "class_name"), "PPO"),
        (("train", "algorithm", "learning_rate"), 1e-5),
        (("train", "actor", "class_name"), "MLPModel"),
        (("train", "critic", "class_name"), "MLPModel"),
        (("train", "runner", "max_iterations"), 300),
        (("train", "num_steps_per_env"), 128),
    ),
)
def test_canonical_recipe_contract(key_path: tuple[str, ...], expected: object) -> None:
    value = load_training_recipe(EXAMPLE / "recipe.yaml")
    for key in key_path:
        value = value[key]
    assert value == expected


def test_public_rl_tree_contains_only_the_fixed_layout_release_assets() -> None:
    assert sorted(path.name for path in EXAMPLE.glob("*.yaml")) == ["recipe.yaml"]
    assert sorted(path.name for path in (EXAMPLE / "scenarios").glob("*.json")) == ["fixed_seed17000_n512.json"]
    assert sorted(path.name for path in BUNDLE.iterdir()) == [
        "config.yaml",
        "evaluation_summary.json",
        "model_299_g2stable.checkpoint_manifest.json",
        "model_299_g2stable.pt",
    ]
    assert not (EXAMPLE / "random_start").exists()
    assert not any(EXAMPLE.rglob("*.sh"))


def test_public_entries_are_module_only_and_never_use_unsafe_rsl_loading() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLE.glob("*.py"))
    assert "sys.path" not in combined
    assert "weights_only=False" not in combined
    assert ".load(str(" not in combined
    assert 'Path("outputs") / "rl" / "pick_place"' in combined
    assert 'DEFAULT_CHECKPOINT = EXAMPLE_DIR / "pretrained" / "model_299_g2stable.pt"' in combined


def test_bundled_checkpoint_hashes_validate_without_writing_to_bundle() -> None:
    checkpoint = BUNDLE / "model_299_g2stable.pt"
    config = BUNDLE / "config.yaml"
    before = {path.name: path.stat().st_mtime_ns for path in BUNDLE.iterdir()}

    artifact, manifest = validate_checkpoint_artifacts(
        checkpoint,
        config,
        expected_task="pick_place",
        expected_robot_key="xarm6_1305",
        expected_runtime_config_sha256=CURRENT_RUNTIME_SHA256,
    )

    assert artifact["config_sha256"] == CONFIG_SHA256
    assert manifest.checkpoint_sha256 == CHECKPOINT_SHA256
    assert checkpoint.stat().st_size == 2_724_011
    assert before == {path.name: path.stat().st_mtime_ns for path in BUNDLE.iterdir()}


def test_bundled_config_is_sanitized_and_summary_is_fixed_layout_only() -> None:
    config_text = (BUNDLE / "config.yaml").read_text(encoding="utf-8")
    for private_fragment in ("/home/", "/Users/", "/tmp/", "logs/pick_place", "outputs/rl"):
        assert private_fragment not in config_text
    artifact = load_training_config(BUNDLE / "config.yaml")
    assert artifact["robot"]["urdf_path"].startswith("assets/urdf/")
    assert artifact["train"]["runner"]["transfer_checkpoint"] is None

    summary = json.loads((BUNDLE / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert summary["evaluation"]["seed_batch_matrix"]["runs_passed"] == "9/9"
    assert summary["evaluation"]["seed_batch_matrix"]["episodes_passed"] == "219/219"
    assert summary["evaluation"]["fixed_bank"]["success"] == "64/64"
    assert summary["evaluation"]["fixed_action_noise_bank"]["success"] == "512/512"
    assert summary["evaluation"]["fixed_action_noise_bank"]["robustness_target_pass"] is True
    assert summary["scope"]["layout"] == "fixed +Y"
    assert "random_start" not in json.dumps(summary)


@pytest.mark.parametrize("entry_name", ["train.py", "pretrain_bc.py"])
def test_training_provenance_hashes_shared_simulation_policy(entry_name: str) -> None:
    source = (EXAMPLE / entry_name).read_text(encoding="utf-8")
    for relative_path in (
        "ufactory/simulation/__init__.py",
        "ufactory/simulation/compat.py",
        "ufactory/simulation/g2.py",
        "ufactory/simulation/physics.py",
    ):
        assert f'repo_root / "{relative_path}"' in source


@pytest.mark.parametrize("entry_name", ["train.py", "pretrain_bc.py"])
def test_training_provenance_hashes_artifact_and_model_policy(entry_name: str) -> None:
    source = (EXAMPLE / entry_name).read_text(encoding="utf-8")
    for relative_path in (
        "ufactory/training/logic/__init__.py",
        "ufactory/training/artifacts.py",
        "ufactory/training/models.py",
        "ufactory/training/transfer.py",
    ):
        assert f'repo_root / "{relative_path}"' in source


def test_bc_provenance_hashes_imported_training_entry() -> None:
    source = (EXAMPLE / "pretrain_bc.py").read_text(encoding="utf-8")
    assert 'Path(__file__).with_name("train.py")' in source
    assert "near_table_weight=near_table_weight" in source
    assert "--near-table-phase-weight" in source
    assert "--near-table-down-penalty" in source
    assert "--fit-output-action-dims" in source


def test_no_legacy_checkpoint_or_random_start_contract_is_published() -> None:
    assert not (EXAMPLE / "random_start").exists()
    assert not (BUNDLE / "model_199.pt").exists()
    for readme in (EXAMPLE / "README.md", EXAMPLE / "README_cn.md"):
        text = readme.read_text(encoding="utf-8")
        assert "model_199" not in text
        assert "Genesis World 1.3.1" not in text
        assert "random_start/" not in text


def test_published_scenario_bank_is_fixed_complete_and_runtime_bound() -> None:
    path = EXAMPLE / "scenarios" / "fixed_seed17000_n512.json"
    bank = load_scenario_bank(path, expected_runtime_config_sha256=CURRENT_RUNTIME_SHA256)
    assert bank["mode"] == "fixed"
    assert bank["count"] == 512
    assert len(bank["scenarios"]) == 512
    assert scenario_bank_sha256(path) == ("2238202c52c6c76f570ad477143a81c67d2773d532a0e44d3d7e0aa375dbeb0b")
