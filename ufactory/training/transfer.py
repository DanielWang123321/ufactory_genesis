"""Explicit, safe neural-network transfer transforms for public RL workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class AppendedObservationActorGuard:
    """Keep a transferred actor fixed except for newly appended input columns."""

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        source_observation_dim: int,
        require_zero_appended: bool = True,
    ) -> None:
        parameters = dict(actor.named_parameters())
        first_key = "mlp.0.weight"
        first_weight = parameters.get(first_key)
        if first_weight is None or first_weight.ndim != 2:
            raise ValueError(f"appended-column training requires actor parameter {first_key}")
        self.source_observation_dim = int(source_observation_dim)
        self.target_observation_dim = int(first_weight.shape[-1])
        if not 0 < self.source_observation_dim < self.target_observation_dim:
            raise ValueError(
                "appended-column training requires an expanded actor input: "
                f"{self.source_observation_dim}->{self.target_observation_dim}"
            )
        if require_zero_appended and torch.count_nonzero(first_weight[..., self.source_observation_dim :]).item() != 0:
            raise ValueError("appended actor observation columns must be zero at guarded transfer")

        self._parameters = parameters
        self._protected = {
            name: (
                parameter[..., : self.source_observation_dim].detach().clone()
                if name == first_key
                else parameter.detach().clone()
            )
            for name, parameter in parameters.items()
        }
        self._first_key = first_key
        for name, parameter in parameters.items():
            parameter.requires_grad_(name == first_key)

        gradient_mask = torch.zeros_like(first_weight)
        gradient_mask[..., self.source_observation_dim :] = 1
        self._gradient_hook = first_weight.register_hook(lambda gradient: gradient * gradient_mask)

    @property
    def trainable_parameter_count(self) -> int:
        first_weight = self._parameters[self._first_key]
        return int(first_weight[..., self.source_observation_dim :].numel())

    def assert_preserved(self) -> None:
        """Fail if an optimizer or future refactor mutates any inherited actor value."""

        for name, expected in self._protected.items():
            parameter = self._parameters[name]
            current = parameter[..., : self.source_observation_dim] if name == self._first_key else parameter
            if not torch.equal(current.detach(), expected):
                raise RuntimeError(f"guarded actor transfer mutated inherited parameter {name}")


class LayoutResidualActorGuard:
    """Freeze the legacy actor branch while a layout-gated residual is trained."""

    def __init__(self, actor: torch.nn.Module) -> None:
        head = getattr(actor, "mlp", None)
        base = getattr(head, "base", None)
        adapter_prefixes = (
            "mlp.residual.",
            "mlp.pickup_residual.",
            "mlp.placement_residual.",
        )
        if not isinstance(base, torch.nn.Module):
            raise ValueError("layout residual training requires actor.mlp.base")
        self._parameters = dict(actor.named_parameters())
        self._trainable_names = {name for name in self._parameters if name.startswith(adapter_prefixes)}
        if not self._trainable_names:
            raise ValueError("layout residual actor exposes no residual parameters")
        self._protected = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters.items()
            if name not in self._trainable_names
        }
        for name, parameter in self._parameters.items():
            parameter.requires_grad_(name in self._trainable_names)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for name, parameter in self._parameters.items() if name in self._trainable_names)

    def assert_preserved(self) -> None:
        """Fail if training changes any base-actor parameter."""

        for name, expected in self._protected.items():
            if not torch.equal(self._parameters[name].detach(), expected):
                raise RuntimeError(f"layout residual training mutated inherited parameter {name}")


class FrozenActorGuard:
    """Make a deployment actor immutable while its critic artifact is initialized."""

    def __init__(self, actor: torch.nn.Module) -> None:
        self._parameters = dict(actor.named_parameters())
        self._protected = {name: parameter.detach().clone() for name, parameter in self._parameters.items()}
        for parameter in self._parameters.values():
            parameter.requires_grad_(False)

    @property
    def trainable_parameter_count(self) -> int:
        return 0

    def assert_preserved(self) -> None:
        for name, expected in self._protected.items():
            if not torch.equal(self._parameters[name].detach(), expected):
                raise RuntimeError(f"frozen actor parameter changed: {name}")


def constrain_actor_to_appended_observation_columns(
    actor: torch.nn.Module,
    *,
    source_observation_dim: int,
    require_zero_appended: bool = True,
) -> AppendedObservationActorGuard:
    """Freeze an actor while allowing only zero-appended input columns to learn."""

    return AppendedObservationActorGuard(
        actor,
        source_observation_dim=source_observation_dim,
        require_zero_appended=require_zero_appended,
    )


def constrain_actor_to_layout_residual(actor: torch.nn.Module) -> LayoutResidualActorGuard:
    """Freeze the legacy branch and expose only a layout residual to optimization."""

    return LayoutResidualActorGuard(actor)


def freeze_actor(actor: torch.nn.Module) -> FrozenActorGuard:
    """Freeze every actor parameter and retain an exact mutation guard."""

    return FrozenActorGuard(actor)


def initialize_guided_pick_place_actor(
    actor: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> dict[str, int | str]:
    """Load the legacy 44-value actor head into a guided 54-value policy."""

    source = checkpoint.get("actor_state_dict")
    if not isinstance(source, Mapping):
        raise ValueError("RSL-RL checkpoint must contain actor_state_dict")
    if actor.__class__.__name__ != "GuidedPickPlaceMLPModel":
        raise ValueError("scripted action hint transfer requires GuidedPickPlaceMLPModel")
    target = actor.state_dict()
    if source.keys() != target.keys():
        missing = sorted(target.keys() - source.keys())
        extra = sorted(source.keys() - target.keys())
        raise ValueError(f"guided/base actor state mismatch: missing={missing}, extra={extra}")
    loaded = {}
    for key, target_value in target.items():
        source_value = source[key]
        if not isinstance(source_value, torch.Tensor) or source_value.shape != target_value.shape:
            source_shape = tuple(source_value.shape) if isinstance(source_value, torch.Tensor) else None
            raise ValueError(
                f"guided/base actor tensor shape mismatch for {key}: "
                f"source={source_shape}, target={tuple(target_value.shape)}"
            )
        loaded[key] = source_value.to(device=device, dtype=target_value.dtype)
    actor.load_state_dict(loaded, strict=True)
    return {
        "type": "scripted_action_hint_actor_v1",
        "source_policy_observation_dim": 44,
        "target_policy_observation_dim": int(getattr(actor, "obs_dim", 54)),
        "appended_observation_dim": 10,
    }


def initialize_layout_residual_actor(
    actor: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> dict[str, int | str]:
    """Load a legacy actor into the base branch of a layout-residual model."""

    source = checkpoint.get("actor_state_dict")
    if not isinstance(source, Mapping):
        raise ValueError("RSL-RL checkpoint must contain actor_state_dict")
    head = getattr(actor, "mlp", None)
    base = getattr(head, "base", None)
    source_dim = getattr(head, "source_observation_dim", None)
    transfer_type = getattr(head, "transfer_type", None)
    if not isinstance(base, torch.nn.Module):
        raise ValueError("layout residual initialization requires actor.mlp.base")
    if not isinstance(source_dim, int):
        raise ValueError("layout residual actor is missing its source observation dimension")
    if transfer_type not in {
        "layout_residual_actor_v1",
        "layout_phase_residual_actor_v2",
        "layout_phase_residual_actor_v3",
    }:
        raise ValueError(f"unsupported layout residual actor type: {transfer_type!r}")

    source_base: dict[str, torch.Tensor] = {}
    unexpected = []
    for key, value in source.items():
        if key.startswith("mlp.") and isinstance(value, torch.Tensor):
            source_base[key.removeprefix("mlp.")] = value
        else:
            unexpected.append(key)
    if unexpected:
        raise ValueError(f"legacy actor contains unsupported state tensors: {unexpected}")
    target_base = base.state_dict()
    if source_base.keys() != target_base.keys():
        missing = sorted(target_base.keys() - source_base.keys())
        extra = sorted(source_base.keys() - target_base.keys())
        raise ValueError(f"legacy/base actor state mismatch: missing={missing}, extra={extra}")
    loaded = {}
    for key, target_value in target_base.items():
        source_value = source_base[key]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"legacy/base actor tensor shape mismatch for {key}: "
                f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
            )
        loaded[key] = source_value.to(device=device, dtype=target_value.dtype)
    base.load_state_dict(loaded, strict=True)
    target_dim = int(getattr(actor, "obs_dim", source_dim + 6))
    return {
        "type": transfer_type,
        "source_policy_observation_dim": source_dim,
        "target_policy_observation_dim": target_dim,
        "appended_observation_dim": target_dim - source_dim,
    }


def project_actor_observation_expansion(
    actor: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    device: str | torch.device,
    appended_initializer: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Copy an actor into a wider input layer without changing its old function.

    The checkpoint has already passed artifact integrity checks and safe
    ``weights_only=True`` loading. Every tensor must match exactly except an input
    dimension expansion. By default newly appended columns are zero, so arbitrary
    new feature values have no mathematical effect at the transfer instant. A
    versioned initializer may instead derive the appended columns from inherited
    weights while leaving every inherited value bit-for-bit unchanged.
    """

    source = checkpoint.get("actor_state_dict")
    if not isinstance(source, Mapping):
        raise ValueError("RSL-RL checkpoint must contain actor_state_dict")
    target = actor.state_dict()
    first_key = "mlp.0.weight"
    if first_key not in source or first_key not in target:
        raise ValueError(f"actor observation projection requires {first_key}")
    source_first = source[first_key]
    if not isinstance(source_first, torch.Tensor):
        raise ValueError(f"source actor tensor {first_key} is invalid")
    source_dim = int(source_first.shape[-1])
    target_dim = int(target[first_key].shape[-1])
    if target_dim <= source_dim:
        raise ValueError(f"target policy observation must expand source: {source_dim}->{target_dim}")

    projected: dict[str, torch.Tensor] = {}
    expanded_keys: list[str] = []
    for key, target_value in target.items():
        source_value = source.get(key)
        if not isinstance(source_value, torch.Tensor):
            raise ValueError(f"source checkpoint is missing actor tensor {key}")
        source_value = source_value.to(device=device, dtype=target_value.dtype)
        if source_value.shape == target_value.shape:
            projected[key] = source_value
            continue
        if (
            key.endswith(".weight")
            and source_value.ndim >= 2
            and source_value.shape[:-1] == target_value.shape[:-1]
            and source_value.shape[-1] == source_dim
            and target_value.shape[-1] == target_dim
        ):
            expanded = torch.zeros_like(target_value)
            expanded[..., :source_dim] = source_value
            projected[key] = expanded
            expanded_keys.append(key)
            continue
        raise ValueError(
            f"unsupported actor observation expansion for {key}: "
            f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
        )
    if expanded_keys != [first_key]:
        raise ValueError(f"actor projection must expand only {first_key}; got {expanded_keys}")
    if appended_initializer is not None:
        _initialize_appended_actor_columns(
            projected[first_key],
            source_first.to(device=device, dtype=projected[first_key].dtype),
            source_dim=source_dim,
            target_dim=target_dim,
            initializer=appended_initializer,
        )
    actor.load_state_dict(projected, strict=True)
    return {
        "source_policy_observation_dim": source_dim,
        "target_policy_observation_dim": target_dim,
        "appended_observation_dim": target_dim - source_dim,
    }


def _initialize_appended_actor_columns(
    projected_first_weight: torch.Tensor,
    source_first_weight: torch.Tensor,
    *,
    source_dim: int,
    target_dim: int,
    initializer: Mapping[str, Any],
) -> None:
    """Apply a closed-form, versioned initializer to appended actor inputs."""

    initializer_type = initializer.get("type")
    if initializer_type == "zero_append_normalized_layout_offsets":
        return
    if initializer_type != "canonical_pick_place_layout_offsets_v1":
        raise ValueError(f"unsupported appended actor initializer: {initializer_type!r}")
    if source_dim != 44 or target_dim != 50:
        raise ValueError(
            "canonical pick-place layout initialization requires the immutable "
            f"44->50 observation expansion, got {source_dim}->{target_dim}"
        )

    object_half_span = torch.as_tensor(
        initializer.get("object_half_span_xy_m"),
        device=projected_first_weight.device,
        dtype=projected_first_weight.dtype,
    )
    target_half_span = torch.as_tensor(
        initializer.get("target_half_span_xy_m"),
        device=projected_first_weight.device,
        dtype=projected_first_weight.dtype,
    )
    if object_half_span.shape != (2,) or target_half_span.shape != (2,):
        raise ValueError("canonical pick-place layout half-spans must each contain x and y")
    if not torch.all(torch.isfinite(object_half_span)) or not torch.all(torch.isfinite(target_half_span)):
        raise ValueError("canonical pick-place layout half-spans must be finite")
    if torch.any(object_half_span <= 0) or torch.any(target_half_span <= 0):
        raise ValueError("canonical pick-place layout half-spans must be positive")

    # The appended features are, in order:
    #   object_delta / object_half_span,
    #   target_delta / target_half_span,
    #   (target_delta - object_delta) / (object_half_span + target_half_span).
    # These weights therefore make the first hidden preactivation identical to one
    # where the legacy object, target, and object-to-target xy inputs have had their
    # episode-layout deltas subtracted. No inherited tensor is edited.
    projected_first_weight[..., 44:46] = -source_first_weight[..., 16:18] * object_half_span
    projected_first_weight[..., 46:48] = -source_first_weight[..., 19:21] * target_half_span
    projected_first_weight[..., 48:50] = -source_first_weight[..., 25:27] * (object_half_span + target_half_span)


__all__ = [
    "AppendedObservationActorGuard",
    "FrozenActorGuard",
    "LayoutResidualActorGuard",
    "constrain_actor_to_appended_observation_columns",
    "constrain_actor_to_layout_residual",
    "freeze_actor",
    "initialize_guided_pick_place_actor",
    "initialize_layout_residual_actor",
    "project_actor_observation_expansion",
]
