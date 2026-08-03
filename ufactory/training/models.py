"""Project-specific RSL-RL models with explicit transfer safety contracts."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories


class _LayoutGatedResidualMLP(torch.nn.Module):
    """Frozen-compatible base head plus a residual that is zero at zero layout."""

    transfer_type = "layout_residual_actor_v1"

    def __init__(
        self,
        base: torch.nn.Module,
        residual: torch.nn.Module,
        *,
        source_observation_dim: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.residual = residual
        self.source_observation_dim = int(source_observation_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] <= self.source_observation_dim:
            raise ValueError(
                f"layout residual actor requires appended layout observations: got {observations.shape[-1]} values"
            )
        base_output = self.base(observations[..., : self.source_observation_dim])
        residual_output = self.residual(observations)
        layout = observations[..., self.source_observation_dim :]
        gate = layout.abs().amax(dim=-1, keepdim=True).clamp(max=1.0)
        placement_started = observations[..., 39:40].clamp(0.0, 1.0)
        gate = gate * (1.0 - placement_started)
        while gate.ndim < residual_output.ndim:
            gate = gate.unsqueeze(-1)
        return base_output + gate * residual_output


def _zero_output_layer(module: torch.nn.Module) -> None:
    final_linear = next(
        (child for child in reversed(list(module.modules())) if isinstance(child, torch.nn.Linear)),
        None,
    )
    if final_linear is None:
        raise ValueError("layout residual branch must contain a linear output layer")
    torch.nn.init.zeros_(final_linear.weight)
    if final_linear.bias is not None:
        torch.nn.init.zeros_(final_linear.bias)


class _LayoutPhaseGatedResidualMLP(torch.nn.Module):
    """Separate coarse pickup and precision placement layout adapters."""

    transfer_type = "layout_phase_residual_actor_v3"

    def __init__(
        self,
        base: torch.nn.Module,
        pickup_residual: torch.nn.Module,
        placement_residual: torch.nn.Module,
        *,
        source_observation_dim: int,
        ever_carried_near_index: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.pickup_residual = pickup_residual
        self.placement_residual = placement_residual
        self.source_observation_dim = int(source_observation_dim)
        self.ever_carried_near_index = int(ever_carried_near_index)

    @staticmethod
    def _match_output_rank(gate: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        while gate.ndim < output.ndim:
            gate = gate.unsqueeze(-1)
        return gate

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] <= self.source_observation_dim:
            raise ValueError(
                "layout phase-residual actor requires appended layout observations: "
                f"got {observations.shape[-1]} values"
            )
        base_output = self.base(observations[..., : self.source_observation_dim])
        pickup_output = self.pickup_residual(observations)
        placement_output = self.placement_residual(observations)
        layout = observations[..., self.source_observation_dim :]
        layout_gate = (layout.abs().amax(dim=-1, keepdim=True) > 1e-7).to(pickup_output.dtype)
        placement_gate = observations[..., self.ever_carried_near_index : self.ever_carried_near_index + 1].clamp(
            0.0,
            1.0,
        )
        layout_gate = self._match_output_rank(layout_gate, pickup_output)
        placement_gate = self._match_output_rank(placement_gate, pickup_output)
        residual = (1.0 - placement_gate) * pickup_output + placement_gate * placement_output
        return base_output + layout_gate * residual


class LayoutResidualMLPModel(MLPModel):
    """MLP actor whose learned layout adapter cannot alter fixed-layout actions.

    The base branch consumes the immutable legacy observation prefix. The residual
    branch consumes the expanded observation, but its raw Beta-parameter correction
    is multiplied by the maximum absolute layout offset and switches off once the cube
    reaches the target neighbourhood. Fixed-layout and precision-placement actions
    therefore remain exactly owned by the inherited actor.
    """

    def __init__(
        self,
        *args,
        source_observation_dim: int = 44,
        residual_hidden_dims: tuple[int, ...] | list[int] | None = None,
        **kwargs,
    ) -> None:
        hidden_dims = kwargs.get("hidden_dims", (256, 256, 256))
        activation = kwargs.get("activation", "elu")
        output_dim = kwargs.get("output_dim")
        if output_dim is None and len(args) >= 4:
            output_dim = args[3]
        super().__init__(*args, **kwargs)
        source_dim = int(source_observation_dim)
        if source_dim <= 0 or self.obs_dim - source_dim != 6:
            raise ValueError(
                "LayoutResidualMLPModel requires exactly six appended layout values: "
                f"source={source_dim}, target={self.obs_dim}"
            )
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else int(output_dim)
        base = MLP(source_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(base)
        residual = MLP(
            self.obs_dim,
            mlp_output_dim,
            residual_hidden_dims or hidden_dims,
            activation,
        )
        _zero_output_layer(residual)
        self.mlp = _LayoutGatedResidualMLP(
            base,
            residual,
            source_observation_dim=source_dim,
        )


class LayoutPhaseResidualMLPModel(MLPModel):
    """Two-branch layout residual for coarse pickup and precision placement."""

    def __init__(
        self,
        *args,
        source_observation_dim: int = 44,
        residual_hidden_dims: tuple[int, ...] | list[int] | None = None,
        ever_carried_near_index: int = 39,
        **kwargs,
    ) -> None:
        hidden_dims = kwargs.get("hidden_dims", (256, 256, 256))
        activation = kwargs.get("activation", "elu")
        output_dim = kwargs.get("output_dim")
        if output_dim is None and len(args) >= 4:
            output_dim = args[3]
        super().__init__(*args, **kwargs)
        source_dim = int(source_observation_dim)
        if source_dim <= 0 or self.obs_dim - source_dim != 6:
            raise ValueError(
                "LayoutPhaseResidualMLPModel requires exactly six appended layout values: "
                f"source={source_dim}, target={self.obs_dim}"
            )
        phase_index = int(ever_carried_near_index)
        if not 0 <= phase_index < source_dim:
            raise ValueError("ever_carried_near_index must lie inside the legacy observation prefix")
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else int(output_dim)
        base = MLP(source_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(base)
        adapter_dims = residual_hidden_dims or hidden_dims
        pickup_residual = MLP(self.obs_dim, mlp_output_dim, adapter_dims, activation)
        placement_residual = MLP(self.obs_dim, mlp_output_dim, adapter_dims, activation)
        _zero_output_layer(pickup_residual)
        _zero_output_layer(placement_residual)
        self.mlp = _LayoutPhaseGatedResidualMLP(
            base,
            pickup_residual,
            placement_residual,
            source_observation_dim=source_dim,
            ever_carried_near_index=phase_index,
        )


class GuidedPickPlaceMLPModel(MLPModel):
    """Preserve the fixed RL actor and use a verified guide on random layouts.

    The guide action is part of the explicit policy observation contract. This model
    keeps the original 44-value actor head intact for the canonical layout and selects
    the four appended guide values only when one of the six immutable layout offsets
    is nonzero. Random-layout stochastic PPO is rejected because replacing sampled
    actions would invalidate its log probabilities; this model is a deterministic
    deployment policy and a transparent fallback after learned residuals miss gates.
    """

    def __init__(
        self,
        *args,
        source_observation_dim: int = 44,
        layout_offset_start: int = 44,
        guide_action_start: int = 50,
        **kwargs,
    ) -> None:
        hidden_dims = kwargs.get("hidden_dims", (256, 256, 256))
        activation = kwargs.get("activation", "elu")
        output_dim = kwargs.get("output_dim")
        if output_dim is None and len(args) >= 4:
            output_dim = args[3]
        super().__init__(*args, **kwargs)
        self.source_observation_dim = int(source_observation_dim)
        self.layout_offset_start = int(layout_offset_start)
        self.guide_action_start = int(guide_action_start)
        if self.obs_normalization:
            raise ValueError("guided pick-place actor requires unnormalized policy observations")
        if self.source_observation_dim != 44 or self.layout_offset_start != 44:
            raise ValueError("guided pick-place actor requires the immutable 44-value legacy prefix")
        if self.guide_action_start - self.layout_offset_start != 6 or self.obs_dim - self.guide_action_start != 4:
            raise ValueError("guided pick-place actor requires legacy(44)+layout(6)+guide_action(4) observations")
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else int(output_dim)
        self.mlp = MLP(
            self.source_observation_dim,
            mlp_output_dim,
            hidden_dims,
            activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def base_raw_output(self, latent: torch.Tensor) -> torch.Tensor:
        """Return distribution parameters from the immutable legacy observation prefix."""

        return self.mlp(latent[..., : self.source_observation_dim])

    def select_guided_action(self, latent: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        """Select the fixed actor on its canonical layout and the guide elsewhere."""

        layout = latent[..., self.layout_offset_start : self.guide_action_start]
        use_guide = layout.abs().amax(dim=-1, keepdim=True) > 1e-7
        guide_action = latent[..., self.guide_action_start : self.guide_action_start + 4].clamp(-1.0, 1.0)
        return torch.where(use_guide, guide_action, base_action)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        raw_base = self.base_raw_output(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(raw_base)
                base_action = self.distribution.sample()
            else:
                base_action = self.distribution.deterministic_output(raw_base)
        else:
            base_action = raw_base
        layout = latent[..., self.layout_offset_start : self.guide_action_start]
        use_guide = layout.abs().amax(dim=-1, keepdim=True) > 1e-7
        if stochastic_output and bool(use_guide.any().item()):
            raise RuntimeError("guided random-layout policy is deterministic and cannot be optimized with PPO")
        return self.select_guided_action(latent, base_action)


__all__ = [
    "GuidedPickPlaceMLPModel",
    "LayoutPhaseResidualMLPModel",
    "LayoutResidualMLPModel",
]
