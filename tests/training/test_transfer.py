from __future__ import annotations

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from ufactory.training.models import (
    GuidedPickPlaceMLPModel,
    LayoutPhaseResidualMLPModel,
    LayoutResidualMLPModel,
)
from ufactory.training.transfer import (
    constrain_actor_to_appended_observation_columns,
    constrain_actor_to_layout_residual,
    freeze_actor,
    initialize_guided_pick_place_actor,
    initialize_layout_residual_actor,
    project_actor_observation_expansion,
)


class _Actor(torch.nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 4),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


def test_actor_observation_expansion_preserves_old_function_exactly() -> None:
    torch.manual_seed(7)
    source = _Actor(3)
    target = _Actor(5)
    checkpoint = {"actor_state_dict": source.state_dict()}
    old_observations = torch.randn(16, 3)
    appended_features = torch.randn(16, 2)

    projection = project_actor_observation_expansion(target, checkpoint, device="cpu")

    assert projection == {
        "source_policy_observation_dim": 3,
        "target_policy_observation_dim": 5,
        "appended_observation_dim": 2,
    }
    torch.testing.assert_close(
        source(old_observations),
        target(torch.cat([old_observations, appended_features], dim=-1)),
        rtol=0.0,
        atol=1e-7,
    )
    assert torch.count_nonzero(target.mlp[0].weight[:, 3:]).item() == 0


def test_canonical_layout_projection_matches_legacy_inputs_with_layout_removed() -> None:
    torch.manual_seed(11)
    source = _Actor(44)
    target = _Actor(50)
    old_observations = torch.randn(32, 44)
    layout_offsets = torch.randn(32, 6)
    object_half_span = torch.tensor([0.03, 0.05])
    target_half_span = torch.tensor([0.03, 0.05])
    canonical_observations = old_observations.clone()
    canonical_observations[:, 16:18] -= layout_offsets[:, 0:2] * object_half_span
    canonical_observations[:, 19:21] -= layout_offsets[:, 2:4] * target_half_span
    canonical_observations[:, 25:27] -= layout_offsets[:, 4:6] * (object_half_span + target_half_span)
    inherited_first = source.mlp[0].weight.detach().clone()

    project_actor_observation_expansion(
        target,
        {"actor_state_dict": source.state_dict()},
        device="cpu",
        appended_initializer={
            "type": "canonical_pick_place_layout_offsets_v1",
            "object_half_span_xy_m": object_half_span.tolist(),
            "target_half_span_xy_m": target_half_span.tolist(),
        },
    )

    assert torch.equal(target.mlp[0].weight[:, :44], inherited_first)
    torch.testing.assert_close(
        target.mlp[0].weight[:, 44:46],
        -inherited_first[:, 16:18] * object_half_span,
    )
    torch.testing.assert_close(
        target.mlp[0].weight[:, 46:48],
        -inherited_first[:, 19:21] * target_half_span,
    )
    torch.testing.assert_close(
        target.mlp[0].weight[:, 48:50],
        -inherited_first[:, 25:27] * (object_half_span + target_half_span),
    )
    torch.testing.assert_close(
        target(torch.cat([old_observations, layout_offsets], dim=-1)),
        source(canonical_observations),
        rtol=1e-5,
        atol=2e-6,
    )


def test_canonical_layout_projection_rejects_wrong_observation_contract() -> None:
    source = _Actor(3)
    target = _Actor(9)
    with pytest.raises(ValueError, match="44->50"):
        project_actor_observation_expansion(
            target,
            {"actor_state_dict": source.state_dict()},
            device="cpu",
            appended_initializer={
                "type": "canonical_pick_place_layout_offsets_v1",
                "object_half_span_xy_m": [0.03, 0.05],
                "target_half_span_xy_m": [0.03, 0.05],
            },
        )


def _rsl_actor(model_class, observation_dim: int, *, beta_distribution: bool = False):
    obs = TensorDict({"policy": torch.zeros(2, observation_dim)}, batch_size=[2])
    kwargs = {}
    if model_class in {LayoutResidualMLPModel, LayoutPhaseResidualMLPModel}:
        kwargs = {"source_observation_dim": 44, "residual_hidden_dims": [8, 8]}
    elif model_class is GuidedPickPlaceMLPModel:
        kwargs = {
            "source_observation_dim": 44,
            "layout_offset_start": 44,
            "guide_action_start": 50,
        }
    return model_class(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=4,
        hidden_dims=[8, 8],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=(
            {"class_name": "BetaDistribution", "action_range": [-1.0, 1.0]} if beta_distribution else None
        ),
        **kwargs,
    )


def test_guided_transfer_preserves_fixed_actor_and_selects_random_layout_guide() -> None:
    torch.manual_seed(31)
    source = _rsl_actor(MLPModel, 44, beta_distribution=True)
    target = _rsl_actor(GuidedPickPlaceMLPModel, 54, beta_distribution=True)
    projection = initialize_guided_pick_place_actor(
        target,
        {"actor_state_dict": source.state_dict()},
        device="cpu",
    )
    legacy = torch.randn(16, 44)
    guide = torch.rand(16, 4) * 2.0 - 1.0
    fixed_input = TensorDict(
        {"policy": torch.cat([legacy, torch.zeros(16, 6), guide], dim=-1)},
        batch_size=[16],
    )
    random_input = TensorDict(
        {"policy": torch.cat([legacy, torch.ones(16, 6), guide], dim=-1)},
        batch_size=[16],
    )

    assert projection == {
        "type": "scripted_action_hint_actor_v1",
        "source_policy_observation_dim": 44,
        "target_policy_observation_dim": 54,
        "appended_observation_dim": 10,
    }
    assert torch.equal(target.state_dict()["mlp.0.weight"], source.state_dict()["mlp.0.weight"])
    assert torch.equal(target(fixed_input), source(TensorDict({"policy": legacy}, batch_size=[16])))
    assert torch.equal(target(random_input), guide)
    guard = freeze_actor(target)
    assert guard.trainable_parameter_count == 0
    assert not any(parameter.requires_grad for parameter in target.parameters())
    guard.assert_preserved()


def test_guided_actor_rejects_stochastic_random_layout_actions() -> None:
    actor = _rsl_actor(GuidedPickPlaceMLPModel, 54, beta_distribution=True)
    observations = torch.zeros(2, 54)
    observations[:, 44] = 1.0
    with pytest.raises(RuntimeError, match="deterministic"):
        actor(
            TensorDict({"policy": observations}, batch_size=[2]),
            stochastic_output=True,
        )


def test_guided_actor_requires_raw_hint_observations() -> None:
    with pytest.raises(ValueError, match="unnormalized"):
        obs = TensorDict({"policy": torch.zeros(2, 54)}, batch_size=[2])
        GuidedPickPlaceMLPModel(
            obs=obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=4,
            hidden_dims=[8, 8],
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "BetaDistribution", "action_range": [-1.0, 1.0]},
        )


def test_guided_actor_can_select_guide_after_custom_base_point_estimate() -> None:
    actor = _rsl_actor(GuidedPickPlaceMLPModel, 54, beta_distribution=True)
    latent = torch.zeros(2, 54)
    latent[1, 44] = 0.5
    latent[:, 50:54] = torch.tensor([[0.9, -0.8, 0.7, -0.6], [-0.5, 0.4, -0.3, 0.2]])
    base_action = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])

    selected = actor.select_guided_action(latent, base_action)

    assert torch.equal(selected[0], base_action[0])
    assert torch.equal(selected[1], latent[1, 50:54])


def test_layout_residual_transfer_is_exact_when_layout_is_zero() -> None:
    torch.manual_seed(19)
    source = _rsl_actor(MLPModel, 44)
    target = _rsl_actor(LayoutResidualMLPModel, 50)
    projection = initialize_layout_residual_actor(
        target,
        {"actor_state_dict": source.state_dict()},
        device="cpu",
    )
    old_observations = torch.randn(16, 44)
    zero_layout = torch.zeros(16, 6)
    source_input = TensorDict({"policy": old_observations}, batch_size=[16])
    target_input = TensorDict(
        {"policy": torch.cat([old_observations, zero_layout], dim=-1)},
        batch_size=[16],
    )

    assert projection == {
        "type": "layout_residual_actor_v1",
        "source_policy_observation_dim": 44,
        "target_policy_observation_dim": 50,
        "appended_observation_dim": 6,
    }
    assert torch.equal(source(source_input), target(target_input))
    residual_output = next(
        module for module in reversed(list(target.mlp.residual.modules())) if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        residual_output.bias.fill_(1.0)
    placement_observations = old_observations.clone()
    placement_observations[:, 39] = 1.0
    random_layout = torch.randn(16, 6)
    placement_source = TensorDict({"policy": placement_observations}, batch_size=[16])
    placement_target = TensorDict(
        {"policy": torch.cat([placement_observations, random_layout], dim=-1)},
        batch_size=[16],
    )
    assert torch.equal(source(placement_source), target(placement_target))


def test_layout_residual_guard_trains_adapter_without_mutating_base() -> None:
    torch.manual_seed(23)
    source = _rsl_actor(MLPModel, 44)
    target = _rsl_actor(LayoutResidualMLPModel, 50)
    initialize_layout_residual_actor(
        target,
        {"actor_state_dict": source.state_dict()},
        device="cpu",
    )
    guard = constrain_actor_to_layout_residual(target)
    base_before = {name: value.detach().clone() for name, value in target.mlp.base.named_parameters()}
    residual_before = {name: value.detach().clone() for name, value in target.mlp.residual.named_parameters()}
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-2)
    observations = torch.randn(32, 50)
    observations[:, 44:] = torch.rand(32, 6) * 2.0 - 1.0
    tensor_dict = TensorDict({"policy": observations}, batch_size=[32])

    target(tensor_dict).square().mean().backward()
    optimizer.step()
    guard.assert_preserved()

    assert guard.trainable_parameter_count > 0
    for name, before in base_before.items():
        assert torch.equal(dict(target.mlp.base.named_parameters())[name], before)
    assert any(
        not torch.equal(parameter, residual_before[name]) for name, parameter in target.mlp.residual.named_parameters()
    )


def test_layout_phase_residual_selects_pickup_and_placement_adapters() -> None:
    torch.manual_seed(29)
    source = _rsl_actor(MLPModel, 44)
    target = _rsl_actor(LayoutPhaseResidualMLPModel, 50)
    projection = initialize_layout_residual_actor(
        target,
        {"actor_state_dict": source.state_dict()},
        device="cpu",
    )
    pickup_output = next(
        module for module in reversed(list(target.mlp.pickup_residual.modules())) if isinstance(module, torch.nn.Linear)
    )
    placement_output = next(
        module
        for module in reversed(list(target.mlp.placement_residual.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        pickup_output.bias.fill_(1.0)
        placement_output.bias.fill_(2.0)
    old_observations = torch.randn(3, 44)
    old_observations[:, 39] = torch.tensor([0.0, 1.0, 1.0])
    layouts = torch.tensor([[1.0] * 6, [1.0] * 6, [0.0] * 6])
    source_input = TensorDict({"policy": old_observations}, batch_size=[3])
    target_input = TensorDict(
        {"policy": torch.cat([old_observations, layouts], dim=-1)},
        batch_size=[3],
    )
    delta = target(target_input) - source(source_input)

    assert projection["type"] == "layout_phase_residual_actor_v3"
    torch.testing.assert_close(delta[0], torch.ones(4))
    torch.testing.assert_close(delta[1], torch.full((4,), 2.0))
    assert torch.equal(delta[2], torch.zeros(4))
    guard = constrain_actor_to_layout_residual(target)
    assert guard.trainable_parameter_count > 0
    guard.assert_preserved()


def test_actor_observation_expansion_rejects_non_expansion() -> None:
    actor = _Actor(3)
    with pytest.raises(ValueError, match="must expand"):
        project_actor_observation_expansion(
            actor,
            {"actor_state_dict": actor.state_dict()},
            device="cpu",
        )


def test_actor_observation_expansion_rejects_other_shape_changes() -> None:
    source = _Actor(3)
    target = _Actor(5)
    state = source.state_dict()
    state["mlp.2.weight"] = torch.zeros(5, 8)
    with pytest.raises(ValueError, match="unsupported actor observation expansion"):
        project_actor_observation_expansion(
            target,
            {"actor_state_dict": state},
            device="cpu",
        )


def test_appended_column_guard_trains_only_new_actor_inputs() -> None:
    torch.manual_seed(17)
    source = _Actor(3)
    target = _Actor(5)
    project_actor_observation_expansion(
        target,
        {"actor_state_dict": source.state_dict()},
        device="cpu",
    )
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-2)
    inherited_before = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}
    guard = constrain_actor_to_appended_observation_columns(
        target,
        source_observation_dim=3,
    )

    observations = torch.randn(32, 5)
    target(observations).square().mean().backward()
    optimizer.step()
    guard.assert_preserved()

    assert guard.trainable_parameter_count == 16
    parameters = dict(target.named_parameters())
    assert torch.equal(parameters["mlp.0.weight"][:, :3], inherited_before["mlp.0.weight"][:, :3])
    assert not torch.equal(parameters["mlp.0.weight"][:, 3:], inherited_before["mlp.0.weight"][:, 3:])
    for name, before in inherited_before.items():
        if name != "mlp.0.weight":
            assert torch.equal(parameters[name], before)


def test_appended_column_guard_rejects_nonzero_new_columns() -> None:
    actor = _Actor(5)
    with pytest.raises(ValueError, match="must be zero"):
        constrain_actor_to_appended_observation_columns(
            actor,
            source_observation_dim=3,
        )


def test_appended_column_guard_can_resume_existing_new_columns() -> None:
    actor = _Actor(5)
    guard = constrain_actor_to_appended_observation_columns(
        actor,
        source_observation_dim=3,
        require_zero_appended=False,
    )
    guard.assert_preserved()
    assert guard.trainable_parameter_count == 16
