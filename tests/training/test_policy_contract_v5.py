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
