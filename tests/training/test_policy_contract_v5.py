"""RSL-RL 5 contract tests for the one public fixed-layout recipe."""

from __future__ import annotations

from copy import deepcopy
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from packaging.version import Version
import pytest
import torch
from tensordict import TensorDict

from ufactory.training import load_training_recipe


PUBLIC_PICK_PLACE = Path(__file__).resolve().parents[2] / "examples" / "rl" / "pick_place"


def test_public_contact_recipe_keeps_fixed_layout_policy_contract() -> None:
    recipe = load_training_recipe(PUBLIC_PICK_PLACE / "recipe.yaml")
    env = recipe["environment"]
    reward = recipe["reward"]
    train = recipe["train"]

    assert env["num_obs"] == 48
    assert env["include_scripted_action_hint"] is True
    assert env["num_actions"] == 4
    assert env["fixed_demo_layout"] is True
    assert env["include_normalized_layout_offsets"] is False
    assert env["include_contact_observations"] is True
    assert env["use_contact_holding"] is True
    assert env["gripper_min_command_gap_m"] == pytest.approx(0.012)
    assert env["action_scale"] == pytest.approx(0.005)
    assert env["max_cartesian_delta_m"] == pytest.approx(0.005)
    assert env["ee_command_integration"] == "commanded"
    assert env["action_response_exponent"] == pytest.approx(2.0)
    assert env["place_success_dist_m"] == pytest.approx(0.010)
    assert env["pre_lift_max_drag_m"] == pytest.approx(0.005)
    assert env["post_release_max_drift_m"] == pytest.approx(0.003)
    assert "train_only_actor_action_index" not in env
    assert "train_only_actor_action_indices" not in env

    assert reward["pre_lift_xy_progress"] == pytest.approx(200.0)
    assert reward["grasp_centering"] == pytest.approx(60.0)
    assert reward["valid_release"] == pytest.approx(200.0)
    assert train["algorithm"]["schedule"] == "fixed"
    assert train["actor"]["distribution_cfg"]["class_name"] == "BetaDistribution"
    assert train["actor"]["distribution_cfg"]["action_range"] == [-1.0, 1.0]


def test_public_beta_policy_samples_stay_inside_action_contract() -> None:
    pytest.importorskip("rsl_rl")
    try:
        version = Version(metadata.version("rsl-rl-lib"))
    except metadata.PackageNotFoundError:
        pytest.skip("rsl-rl-lib is not installed")
    if version < Version("5.3.0"):
        pytest.skip("BetaDistribution requires rsl-rl-lib >= 5.3")

    from rsl_rl.models import MLPModel

    recipe = load_training_recipe(PUBLIC_PICK_PLACE / "recipe.yaml")
    obs = TensorDict(
        {
            "policy": torch.randn(4096, 48),
            "privileged": torch.randn(4096, 6),
        },
        batch_size=[4096],
    )
    actor_cfg = deepcopy(recipe["train"]["actor"])
    assert actor_cfg.pop("class_name") == "MLPModel"
    actor = MLPModel(
        obs,
        recipe["train"]["obs_groups"],
        "actor",
        output_dim=4,
        **actor_cfg,
    )
    stochastic = actor(obs, stochastic_output=True)
    deterministic = actor(obs, stochastic_output=False)
    assert stochastic.shape == (4096, 4)
    assert torch.all(stochastic >= -1.0)
    assert torch.all(stochastic <= 1.0)
    assert torch.all(deterministic >= -1.0)
    assert torch.all(deterministic <= 1.0)


def test_fixed_learning_rate_guard_checks_every_update() -> None:
    from examples.rl.pick_place.train import _install_fixed_learning_rate_guard

    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.Adam([parameter], lr=3e-5)
    algorithm = SimpleNamespace(
        learning_rate=3e-5,
        optimizer=optimizer,
        update=lambda: "ok",
    )
    runner = SimpleNamespace(alg=algorithm)
    cfg = {"algorithm": {"schedule": "fixed", "learning_rate": 3e-5}}
    _install_fixed_learning_rate_guard(runner, cfg)
    assert runner.alg.update() == "ok"

    def changes_rate():
        runner.alg.learning_rate = 1e-3
        optimizer.param_groups[0]["lr"] = 1e-3

    runner.alg.update = changes_rate
    _install_fixed_learning_rate_guard(runner, cfg)
    with pytest.raises(RuntimeError, match="fixed learning rate changed after"):
        runner.alg.update()
