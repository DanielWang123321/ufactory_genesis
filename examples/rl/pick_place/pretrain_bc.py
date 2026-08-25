"""Behaviour-clone the scripted expert into the Beta actor used by PPO.

Exploring a clean grasp from scratch is the hardest part of this task: the reward can
only say "you dragged the cube" after the fact, while the expert already knows how to
descend, close and lift without touching it. Fitting the actor to expert actions first
turns the PPO run into refinement instead of discovery.

The output is an ordinary run directory, so the public training and evaluation
modules consume it exactly like any other checkpoint bundle.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from tensordict import TensorDict

from ufactory.config import load_runtime_config
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    build_pick_place_task_configs,
    build_train_config,
    load_training_config,
    write_artifact_inventory,
    write_run_provenance,
    write_training_config,
    validate_and_load_rsl_checkpoint,
)
from ufactory.training.logic.pick_place import cartesian_delta_to_action, near_table_down_penalty
from ufactory.training.transfer import (
    constrain_actor_to_appended_observation_columns,
    constrain_actor_to_layout_residual,
    initialize_layout_residual_actor,
    project_actor_observation_expansion,
)

import genesis as gs

from .env import XArm6PickPlaceEnv
from .expert import (
    PHASE_NAMES,
    ScriptedPickPlaceExpert,
    expert_phase_sample_weights,
    scripted_pick_place_expert_from_env,
)
from .trace_utils import apply_deterministic_action_noise
from .train import (
    ArtifactOnPolicyRunner,
    LAYOUT_RESIDUAL_PROJECTION_TYPES,
    _build_observation_projection,
)


def collect_demonstrations(
    env: XArm6PickPlaceEnv,
    expert: ScriptedPickPlaceExpert,
    steps: int,
    actor=None,
    *,
    execution_noise_std: float = 0.0,
    noise_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Record expert-labelled observations along a rollout.

    With ``actor`` set, the student drives the arm while the expert still supplies the
    label. Cloning only the expert's own trajectory leaves the policy untrained on the
    states its own small errors lead to, which is what turns a millimetre of drift
    during transport into centimetres by the time it reaches the target.
    """
    observations = []
    actions = []
    phases = []
    if execution_noise_std < 0.0:
        raise ValueError("execution_noise_std must be non-negative")
    if execution_noise_std > 0.0 and noise_generator is None:
        raise ValueError("positive execution noise requires a seeded generator")
    obs = env.get_observations()
    for step in range(steps):
        with torch.no_grad():
            label = expert(obs, recover_phase_from_state=actor is not None)
            # Keep the demonstrations on host memory: a full rollout at training batch
            # size is larger than the spare device memory on the visualisation machine.
            observations.append(obs["policy"].detach().to("cpu", copy=True))
            actions.append(label.detach().to("cpu", copy=True))
            phases.append(expert.phase.detach().to("cpu", copy=True))
            driving = label if actor is None else actor(obs, stochastic_output=False)
            if execution_noise_std > 0.0:
                driving = apply_deterministic_action_noise(
                    driving,
                    std=float(execution_noise_std),
                    generator=noise_generator,
                    action_clip=float(env.action_clip),
                )
            obs, _reward, _done, _extras = env.step(driving.clamp(-1.0, 1.0))
        if (step + 1) % 100 == 0:
            print(f"  collected {step + 1}/{steps} steps", flush=True)
    return torch.cat(observations), torch.cat(actions), torch.cat(phases)


def beta_mean_and_concentration(actor, obs_batch: torch.Tensor):
    """Return the Beta mean in action units and its concentration, both differentiable."""
    observations = TensorDict({"policy": obs_batch}, batch_size=[obs_batch.shape[0]])
    latent = actor.get_latent(observations)
    raw_alpha, raw_beta = torch.unbind(actor.mlp(latent), dim=-2)
    alpha = torch.nn.functional.softplus(raw_alpha) + 1.0
    beta = torch.nn.functional.softplus(raw_beta) + 1.0
    low, high = (float(value) for value in actor.distribution.action_range)
    mean = (alpha / (alpha + beta)) * (high - low) + low
    return mean, alpha + beta


def fit_actor(
    actor,
    observations: torch.Tensor,
    actions: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    *,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    target_concentration: float,
    concentration_weight: float,
    reference_actions: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    reference_anchor_weight: float = 0.0,
    action_component_weights: torch.Tensor | None = None,
    far_open_penalty: float = 0.0,
    carry_near_dist_m: float = 0.040,
    near_table_down_penalty_weight: float = 0.0,
    obj_rest_z_m: float = 0.015,
    near_table_height_m: float = 0.045,
    max_down_action: float = 0.0,
) -> tuple[float, float]:
    """Fit the Beta mean to the expert action and pin how wide the distribution is.

    Matching the mean alone leaves the concentration wherever initialisation put it,
    which makes the first PPO rollouts essentially random and throws away the cloned
    grasp before the critic has learned anything. The second term fixes the sampling
    spread so exploration starts as a small perturbation of the demonstrated action.
    """
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    sample_count = observations.shape[0]
    action_dim = int(actions.shape[-1])
    if sample_weights is None:
        sample_weights = torch.ones(sample_count, dtype=torch.float32)
    if sample_weights.shape != (sample_count,):
        raise ValueError(f"sample_weights must have shape ({sample_count},), got {tuple(sample_weights.shape)}")
    if not bool(torch.all(sample_weights > 0.0)):
        raise ValueError("sample_weights must be positive")
    if action_component_weights is None:
        action_component_weights = torch.ones(action_dim, dtype=torch.float32)
    if tuple(action_component_weights.shape) != (action_dim,):
        raise ValueError(
            f"action_component_weights must have shape ({action_dim},), got {tuple(action_component_weights.shape)}"
        )
    if not bool(torch.all(action_component_weights > 0.0)):
        raise ValueError("action_component_weights must be positive")
    if far_open_penalty < 0.0:
        raise ValueError("far_open_penalty must be non-negative")
    if near_table_down_penalty_weight < 0.0:
        raise ValueError("near_table_down_penalty_weight must be non-negative")
    if max_down_action < 0.0:
        raise ValueError("max_down_action must be non-negative")
    if carry_near_dist_m <= 0.0:
        raise ValueError("carry_near_dist_m must be positive")
    if reference_anchor_weight < 0.0:
        raise ValueError("reference_anchor_weight must be non-negative")
    if reference_actions is not None:
        if reference_actions.shape != actions.shape:
            raise ValueError("reference_actions must match the expert action shape")
        if reference_mask is None or reference_mask.shape != (sample_count,):
            raise ValueError("reference_mask must contain one value per sample")
    log_target = torch.log(torch.tensor(float(target_concentration), device=device))
    final_mse = float("nan")
    final_concentration = float("nan")
    for epoch in range(epochs):
        # Genesis switches the default torch device to the GPU; the demonstrations stay
        # on the host, so the shuffle has to be built there too.
        order = torch.randperm(sample_count, device="cpu")
        epoch_mse = 0.0
        epoch_concentration = 0.0
        batches = 0
        for start in range(0, sample_count, batch_size):
            index = order[start : start + batch_size]
            obs_batch = observations[index].to(device, non_blocking=True)
            action_batch = actions[index].to(device, non_blocking=True)
            weight_batch = sample_weights[index].to(device, non_blocking=True)
            predicted, concentration = beta_mean_and_concentration(actor, obs_batch)
            weight_sum = weight_batch.sum().clamp_min(1e-8)
            component_weights = action_component_weights.to(device=device, dtype=predicted.dtype)
            component_weights = component_weights / component_weights.mean().clamp_min(1e-8)
            mse_per_sample = torch.mean(
                component_weights * torch.square(predicted - action_batch),
                dim=-1,
            )
            mse = torch.sum(weight_batch * mse_per_sample) / weight_sum
            spread_per_sample = torch.mean(
                torch.square(torch.log(concentration) - log_target),
                dim=-1,
            )
            spread = torch.sum(weight_batch * spread_per_sample) / weight_sum
            loss = mse + concentration_weight * spread
            if far_open_penalty > 0.0:
                # 30-dim prefix: obj_to_target xyz at 25:28, grasped at 28.
                xy_to_target = torch.linalg.norm(obs_batch[:, 25:27], dim=-1)
                grasped = obs_batch[:, 28] > 0.5
                far_open = grasped & (xy_to_target > float(carry_near_dist_m))
                open_action = torch.relu(predicted[:, 3])
                extra = torch.sum(weight_batch * far_open.float() * open_action.square()) / weight_sum
                loss = loss + float(far_open_penalty) * extra
            if near_table_down_penalty_weight > 0.0:
                down_pen = near_table_down_penalty(
                    obs_batch,
                    predicted,
                    obj_rest_z_m=float(obj_rest_z_m),
                    height_m=float(near_table_height_m),
                    max_down_action=float(max_down_action),
                )
                extra = torch.sum(weight_batch * down_pen) / weight_sum
                loss = loss + float(near_table_down_penalty_weight) * extra
            if reference_actions is not None and reference_anchor_weight > 0.0:
                anchor_mask = reference_mask[index].to(device, non_blocking=True)
                if bool(anchor_mask.any()):
                    reference_batch = reference_actions[index].to(
                        device,
                        non_blocking=True,
                    )
                    anchor = torch.mean(torch.square(predicted[anchor_mask] - reference_batch[anchor_mask]))
                    loss = loss + float(reference_anchor_weight) * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            epoch_mse += float(mse.item())
            epoch_concentration += float(concentration.mean().item())
            batches += 1
        final_mse = epoch_mse / max(batches, 1)
        final_concentration = epoch_concentration / max(batches, 1)
        print(
            f"  epoch {epoch + 1}/{epochs}: action mse={final_mse:.6f}, concentration={final_concentration:.1f}",
            flush=True,
        )
    return final_mse, final_concentration


def _freeze_actor_except_beta_output(actor) -> tuple[int, ...]:
    """Freeze inherited features and leave all alpha/beta output rows trainable."""

    linear_layers = [module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise ValueError("actor does not expose a linear Beta output layer")
    output = linear_layers[-1]
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    output.weight.requires_grad_(True)
    if output.bias is not None:
        output.bias.requires_grad_(True)
    return tuple(range(output.out_features))


def _freeze_actor_except_action_dims(actor, action_dims: tuple[int, ...]) -> tuple[int, ...]:
    """Freeze everything except Beta alpha/beta rows for selected action dimensions."""

    if not action_dims:
        raise ValueError("action_dims must not be empty")
    linear_layers = [module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise ValueError("actor does not expose a linear Beta output layer")
    output = linear_layers[-1]
    if output.out_features % 2 != 0:
        raise ValueError("Beta output width must be twice the action dimension")
    action_dim = output.out_features // 2
    if any(dim < 0 or dim >= action_dim for dim in action_dims):
        raise ValueError(f"action dims must be in [0, {action_dim})")
    trainable_rows = tuple(dim + offset * action_dim for offset in (0, 1) for dim in action_dims)
    frozen_rows = tuple(index for index in range(output.out_features) if index not in set(trainable_rows))
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    output.weight.requires_grad_(True)
    if output.bias is not None:
        output.bias.requires_grad_(True)

    def _mask_rows(grad: torch.Tensor) -> torch.Tensor:
        masked = grad.clone()
        masked[list(frozen_rows)] = 0
        return masked

    output.weight.register_hook(_mask_rows)
    if output.bias is not None:
        output.bias.register_hook(_mask_rows)
    return trainable_rows


def _reference_actor_actions(
    actor,
    observations: torch.Tensor,
    *,
    device,
    batch_size: int,
) -> torch.Tensor:
    """Run a frozen source actor over host observations without retaining a graph."""

    outputs = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            obs_batch = observations[start : start + batch_size].to(
                device,
                non_blocking=True,
            )
            tensor_dict = TensorDict(
                {"policy": obs_batch},
                batch_size=[obs_batch.shape[0]],
            )
            outputs.append(actor(tensor_dict, stochastic_output=False).detach().to("cpu"))
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Behaviour cloning from the scripted expert")
    parser.add_argument("--robot", default="xarm6")
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("recipe.yaml"))
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Load an architecture-compatible actor and critic before fitting demonstrations",
    )
    parser.add_argument(
        "--fit-output-head-only",
        action="store_true",
        help="Freeze inherited actor features and fit only all Beta alpha/beta output rows",
    )
    parser.add_argument(
        "--fit-output-action-dims",
        type=int,
        nargs="+",
        choices=(0, 1, 2, 3),
        help=(
            "Freeze inherited features and fit only Beta alpha/beta rows for these "
            "action dimensions (0=x, 1=y, 2=z, 3=gripper)"
        ),
    )
    parser.add_argument(
        "--fit-appended-layout-columns-only",
        action="store_true",
        help=(
            "Expand a 44-dimensional --warm-start actor to 50 observations and fit only "
            "the six appended layout columns; every inherited actor value remains exact"
        ),
    )
    parser.add_argument(
        "--fit-layout-residual-only",
        action="store_true",
        help=("Load the legacy actor into a frozen base branch and fit only the layout-gated residual network"),
    )
    parser.add_argument(
        "--reference-anchor-weight",
        type=float,
        default=0.0,
        help="Keep pre-grasp actions close to --warm-start while fitting the expert",
    )
    parser.add_argument("-e", "--exp-name", default="pp-bc")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("-B", "--num-envs", type=int, default=2048)
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        choices=range(5),
        help="Pin demonstration collection to one curriculum stage.",
    )
    parser.add_argument(
        "--grasp-phase-reset-frac",
        type=float,
        help="Override grasp-phase reset fraction while collecting demonstrations",
    )
    parser.add_argument(
        "--carry-phase-reset-frac",
        type=float,
        help="Override carry-phase reset fraction while collecting demonstrations",
    )
    parser.add_argument(
        "--place-phase-reset-frac",
        type=float,
        help="Override place-phase reset fraction while collecting demonstrations",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=600,
        help="Control steps to record; one full episode is 500 steps",
    )
    parser.add_argument(
        "--dagger-rounds",
        type=int,
        default=0,
        help=(
            "Extra rounds recorded along the student's own trajectory with expert labels. "
            "Off by default: the expert latches its phase from history, so once the "
            "student strays it keeps labelling for a phase the arm has already left and "
            "the aggregated targets contradict each other"
        ),
    )
    parser.add_argument(
        "--target-concentration",
        type=float,
        default=400.0,
        help="Beta alpha+beta to aim for; 400 is roughly 0.05 sampling std in action units",
    )
    parser.add_argument("--concentration-weight", type=float, default=0.01)
    parser.add_argument(
        "--execution-noise-std",
        type=float,
        default=None,
        help=(
            "When set, disable recipe training noise and apply this one seeded "
            "Gaussian execution perturbation while collecting expert labels."
        ),
    )
    parser.add_argument(
        "--execution-noise-seed",
        type=int,
        default=20260801,
        help="Independent generator seed used by --execution-noise-std",
    )
    parser.add_argument(
        "--quality-phase-weighting",
        action="store_true",
        help="Weight set-down/settle samples 4x and release/retreat samples 2x",
    )
    parser.add_argument(
        "--phase-balanced-weighting",
        action="store_true",
        help="Remove phase-duration bias before applying optional quality priorities",
    )
    parser.add_argument(
        "--transport-phase-weight",
        type=float,
        default=1.0,
        help="Multiply transport-phase BC samples after optional phase balancing",
    )
    parser.add_argument(
        "--close-lift-phase-weight",
        type=float,
        default=1.0,
        help="Multiply close and lift BC samples after optional phase balancing",
    )
    parser.add_argument(
        "--release-phase-weight",
        type=float,
        help=("Multiply release and retreat BC samples. Default 2 with --quality-phase-weighting, otherwise 1"),
    )
    parser.add_argument(
        "--near-table-phase-weight",
        type=float,
        help=(
            "Multiply set-down and settle BC samples. Default 4 with "
            "--quality-phase-weighting, otherwise 1. Prefer this over "
            "--quality-phase-weighting when release should stay unboosted"
        ),
    )
    parser.add_argument(
        "--action-y-weight",
        type=float,
        default=1.0,
        help="Relative MSE weight for the Cartesian Y action versus the other three dims",
    )
    parser.add_argument(
        "--action-z-weight",
        type=float,
        default=1.0,
        help="Relative MSE weight for the Cartesian Z action versus the other three dims",
    )
    parser.add_argument(
        "--action-gripper-weight",
        type=float,
        default=1.0,
        help="Relative MSE weight for the gripper action versus the Cartesian dims",
    )
    parser.add_argument(
        "--far-open-penalty",
        type=float,
        default=0.0,
        help="Penalize positive gripper actions while grasped farther than carry_near_dist from the target",
    )
    parser.add_argument(
        "--near-table-down-penalty",
        type=float,
        default=0.0,
        help=(
            "Penalize downward Z actions faster than the expert landing-brake step "
            "while grasped within near_table_margin of the table"
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    protected_scopes = sum(
        bool(value)
        for value in (
            args.fit_output_head_only,
            args.fit_output_action_dims,
            args.fit_appended_layout_columns_only,
            args.fit_layout_residual_only,
        )
    )
    if protected_scopes > 1:
        parser.error("choose only one protected BC fitting scope")
    protected_layout_fit = args.fit_appended_layout_columns_only or args.fit_layout_residual_only
    if protected_layout_fit and args.warm_start is None:
        parser.error("protected layout BC requires --warm-start")
    if args.execution_noise_std is not None and args.execution_noise_std < 0.0:
        parser.error("--execution-noise-std must be non-negative")
    if args.reference_anchor_weight < 0.0:
        parser.error("--reference-anchor-weight must be non-negative")
    if args.reference_anchor_weight > 0.0 and args.warm_start is None:
        parser.error("--reference-anchor-weight requires --warm-start")
    if args.warm_start is not None and not args.warm_start.is_file():
        parser.error(f"warm-start checkpoint not found: {args.warm_start}")
    if args.transport_phase_weight <= 0.0:
        parser.error("--transport-phase-weight must be positive")
    if args.close_lift_phase_weight <= 0.0:
        parser.error("--close-lift-phase-weight must be positive")
    if args.release_phase_weight is not None and args.release_phase_weight <= 0.0:
        parser.error("--release-phase-weight must be positive")
    if args.near_table_phase_weight is not None and args.near_table_phase_weight <= 0.0:
        parser.error("--near-table-phase-weight must be positive")
    if args.action_y_weight <= 0.0:
        parser.error("--action-y-weight must be positive")
    if args.action_z_weight <= 0.0:
        parser.error("--action-z-weight must be positive")
    if args.action_gripper_weight <= 0.0:
        parser.error("--action-gripper-weight must be positive")
    if args.far_open_penalty < 0.0:
        parser.error("--far-open-penalty must be non-negative")
    if args.near_table_down_penalty < 0.0:
        parser.error("--near-table-down-penalty must be non-negative")
    if args.fit_output_action_dims and args.warm_start is None:
        parser.error("--fit-output-action-dims requires --warm-start")

    release_weight = (
        float(args.release_phase_weight)
        if args.release_phase_weight is not None
        else (2.0 if args.quality_phase_weighting else 1.0)
    )
    near_table_weight = (
        float(args.near_table_phase_weight)
        if args.near_table_phase_weight is not None
        else (4.0 if args.quality_phase_weighting else 1.0)
    )

    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs(
        args.robot,
        runtime_config_path=args.runtime_config,
        recipe_path=args.recipe,
    )
    env_cfg["num_envs"] = int(args.num_envs)
    if args.curriculum_stage is not None:
        env_cfg["curriculum_initial_stage"] = int(args.curriculum_stage)
        env_cfg["curriculum_max_stage"] = max(
            int(env_cfg.get("curriculum_max_stage", 4)),
            int(args.curriculum_stage),
        )
    if args.grasp_phase_reset_frac is not None:
        if not 0.0 <= args.grasp_phase_reset_frac <= 1.0:
            parser.error("--grasp-phase-reset-frac must be in [0, 1]")
        env_cfg["grasp_phase_reset_frac"] = float(args.grasp_phase_reset_frac)
        env_cfg["grasp_phase_reset_frac_final"] = float(args.grasp_phase_reset_frac)
    if args.carry_phase_reset_frac is not None:
        if not 0.0 <= args.carry_phase_reset_frac <= 1.0:
            parser.error("--carry-phase-reset-frac must be in [0, 1]")
        env_cfg["carry_phase_reset_frac"] = float(args.carry_phase_reset_frac)
    if args.place_phase_reset_frac is not None:
        if not 0.0 <= args.place_phase_reset_frac <= 1.0:
            parser.error("--place-phase-reset-frac must be in [0, 1]")
        env_cfg["place_phase_reset_frac"] = float(args.place_phase_reset_frac)
    source_train_noise_std = float(env_cfg.get("train_action_noise_std", 0.0))
    if args.execution_noise_std is not None:
        env_cfg["train_action_noise_std"] = 0.0
        env_cfg["train_action_noise_std_end"] = 0.0
        env_cfg["noise_anneal_steps"] = 0
        env_cfg["train_action_noise_clean_episode_frac"] = 1.0
    train_cfg = build_train_config(
        args.recipe,
        experiment_name=args.exp_name,
        max_iterations=0,
    )
    train_cfg["seed"] = int(args.seed)
    train_cfg["runner"]["resume"] = False
    train_cfg["runner"]["transfer_mode"] = (
        "behaviour_cloning_from_actor_critic" if args.warm_start is not None else "behaviour_cloning_fresh"
    )
    if args.warm_start is not None:
        with args.warm_start.open("rb") as source:
            source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        train_cfg["runner"]["transfer_checkpoint"] = str(args.warm_start)
        train_cfg["runner"]["transfer_checkpoint_sha256"] = source_sha256
    train_cfg["runner"]["bc_fit_output_head_only"] = bool(args.fit_output_head_only)
    train_cfg["runner"]["bc_fit_output_action_dims"] = (
        None if not args.fit_output_action_dims else [int(dim) for dim in args.fit_output_action_dims]
    )
    train_cfg["runner"]["bc_fit_appended_layout_columns_only"] = bool(args.fit_appended_layout_columns_only)
    train_cfg["runner"]["bc_fit_layout_residual_only"] = bool(args.fit_layout_residual_only)
    train_cfg["runner"]["bc_reference_anchor_weight"] = float(args.reference_anchor_weight)
    train_cfg["runner"]["bc_source_train_action_noise_std"] = source_train_noise_std
    train_cfg["runner"]["bc_execution_noise_std"] = (
        None if args.execution_noise_std is None else float(args.execution_noise_std)
    )
    train_cfg["runner"]["bc_execution_noise_seed"] = int(args.execution_noise_seed)
    train_cfg["runner"]["bc_quality_phase_weighting"] = bool(args.quality_phase_weighting)
    train_cfg["runner"]["bc_phase_balanced_weighting"] = bool(args.phase_balanced_weighting)
    train_cfg["runner"]["bc_transport_phase_weight"] = float(args.transport_phase_weight)
    train_cfg["runner"]["bc_close_lift_phase_weight"] = float(args.close_lift_phase_weight)
    train_cfg["runner"]["bc_release_phase_weight"] = float(release_weight)
    train_cfg["runner"]["bc_near_table_phase_weight"] = float(near_table_weight)
    train_cfg["runner"]["bc_action_y_weight"] = float(args.action_y_weight)
    train_cfg["runner"]["bc_action_z_weight"] = float(args.action_z_weight)
    train_cfg["runner"]["bc_action_gripper_weight"] = float(args.action_gripper_weight)
    train_cfg["runner"]["bc_far_open_penalty"] = float(args.far_open_penalty)
    train_cfg["runner"]["bc_near_table_down_penalty"] = float(args.near_table_down_penalty)
    hint_cfg = env_cfg.get("scripted_action_hint_config") or {}
    brake_step_m = float(hint_cfg.get("landing_brake_max_step_m", 0.0005))
    near_table_height_m = float(hint_cfg.get("near_table_margin_m", env_cfg.get("landing_near_table_height_m", 0.045)))
    max_down_action = float(
        cartesian_delta_to_action(
            torch.tensor([brake_step_m]),
            float(env_cfg.get("action_scale", 0.005)),
            float(env_cfg.get("action_response_exponent", 1.0)),
        ).item()
    )
    train_cfg["runner"]["bc_near_table_down_max_action"] = max_down_action
    train_cfg["runner"]["bc_near_table_down_height_m"] = near_table_height_m
    train_cfg["runner"]["bc_grasp_phase_reset_frac"] = env_cfg.get("grasp_phase_reset_frac")
    train_cfg["runner"]["bc_carry_phase_reset_frac"] = env_cfg.get("carry_phase_reset_frac")
    train_cfg["runner"]["bc_place_phase_reset_frac"] = env_cfg.get("place_phase_reset_frac")

    log_dir = args.log_dir or Path("outputs") / "rl" / "pick_place" / args.exp_name
    if log_dir.exists():
        raise FileExistsError(f"refusing to overwrite behaviour-cloning run: {log_dir}")
    log_dir.mkdir(parents=True)
    runtime_config = load_runtime_config(
        args.robot,
        task="pick_place",
        config_path=args.runtime_config,
    )
    warm_start_state = None
    transfer_projection = None
    transfer_requires_initialization = False
    if args.warm_start is not None:
        warm_start_state, source_config, source_manifest = validate_and_load_rsl_checkpoint(
            args.warm_start,
            args.warm_start.parent / "config.yaml",
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_runtime_env=env_cfg,
            allowed_runtime_env_mismatches=("fixed_demo_layout",),
        )
        source_obs = int(source_manifest.observation_dim)
        target_obs = int(env_cfg["num_obs"])
        if protected_layout_fit:
            if target_obs != 50 or source_obs not in {44, 50}:
                raise ValueError("protected layout BC requires a 44->50 transfer or a guarded 50-dim continuation")
            if not bool(env_cfg.get("include_normalized_layout_offsets", False)):
                raise ValueError("protected layout-column BC requires normalized layout offsets")
            if bool(train_cfg["actor"].get("obs_normalization", False)):
                raise ValueError("protected layout-column BC requires actor observation normalization disabled")
            transfer_projection = _build_observation_projection(
                env_cfg,
                train_cfg,
                source_observation_dim=44,
                target_observation_dim=target_obs,
            )
            if args.fit_layout_residual_only and transfer_projection["type"] not in LAYOUT_RESIDUAL_PROJECTION_TYPES:
                raise ValueError("--fit-layout-residual-only requires a layout-residual actor in the recipe")
            if (
                args.fit_appended_layout_columns_only
                and transfer_projection["type"] in LAYOUT_RESIDUAL_PROJECTION_TYPES
            ):
                raise ValueError("layout-residual actors require --fit-layout-residual-only")
            if source_obs == 50:
                if not args.fit_layout_residual_only:
                    raise ValueError("same-dimension protected BC is supported only for the layout residual actor")
                saved_projection = source_config.get("train", {}).get("runner", {}).get("observation_projection")
                if saved_projection != transfer_projection:
                    raise ValueError("guarded BC continuation requires its saved observation projection")
            else:
                transfer_requires_initialization = True
            train_cfg["runner"]["observation_projection"] = transfer_projection
        elif source_obs != target_obs:
            raise ValueError("observation expansion requires a protected layout BC scope")
    write_training_config(
        log_dir / "config.yaml",
        task="pick_place",
        robot_key=runtime_config.robot.key,
        env=env_cfg,
        reward=reward_cfg,
        robot=robot_cfg,
        train=train_cfg,
    )
    artifact = load_training_config(log_dir / "config.yaml")
    repo_root = Path(__file__).resolve().parents[3]
    provenance_sources = [
        Path(__file__),
        Path(__file__).with_name("env.py"),
        Path(__file__).with_name("expert.py"),
        Path(__file__).with_name("trace_utils.py"),
        Path(__file__).with_name("train.py"),
        args.recipe,
        repo_root / "ufactory/training/logic/__init__.py",
        repo_root / "ufactory/training/logic/pick_place.py",
        repo_root / "ufactory/training/tasks.py",
        repo_root / "ufactory/training/artifacts.py",
        repo_root / "ufactory/training/models.py",
        repo_root / "ufactory/training/transfer.py",
        repo_root / "ufactory/simulation/__init__.py",
        repo_root / "ufactory/simulation/compat.py",
        repo_root / "ufactory/simulation/g2.py",
        repo_root / "ufactory/simulation/physics.py",
    ]
    provenance_sources.extend(repo_root / source for source in runtime_config.sources)
    write_run_provenance(
        log_dir / "run_provenance.json",
        training_config=artifact,
        source_paths=provenance_sources,
    )

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(train_cfg["seed"]),
        performance_mode=bool(env_cfg.get("genesis_performance_mode", False)),
    )
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )
    env.csv_log_path = str(log_dir / "metrics.csv")
    runner = ArtifactOnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.checkpoint_training_config = artifact
    runner.checkpoint_env = env
    reference_actor = None
    transfer_guard = None
    if args.warm_start is not None:
        assert warm_start_state is not None
        if transfer_projection is not None:
            if transfer_projection["type"] in LAYOUT_RESIDUAL_PROJECTION_TYPES:
                if transfer_requires_initialization:
                    resolved_projection = initialize_layout_residual_actor(
                        runner.alg.actor,
                        warm_start_state,
                        device=gs.device,
                    )
                else:
                    runner.alg.load(
                        warm_start_state,
                        {
                            "actor": True,
                            "critic": True,
                            "optimizer": False,
                            "iteration": False,
                            "rnd": False,
                        },
                        strict=True,
                    )
                    resolved_projection = dict(transfer_projection)
                transfer_guard = constrain_actor_to_layout_residual(runner.alg.actor)
                trainable_label = "layout-residual parameters"
            else:
                resolved_projection = project_actor_observation_expansion(
                    runner.alg.actor,
                    warm_start_state,
                    device=gs.device,
                    appended_initializer=transfer_projection,
                )
                transfer_guard = constrain_actor_to_appended_observation_columns(
                    runner.alg.actor,
                    source_observation_dim=int(transfer_projection["source_policy_observation_dim"]),
                    require_zero_appended=(transfer_projection["type"] == "zero_append_normalized_layout_offsets"),
                )
                trainable_label = "appended-column weights"
            expected_projection = {key: transfer_projection[key] for key in resolved_projection}
            if resolved_projection != expected_projection:
                raise RuntimeError("resolved BC actor observation projection differs from training config")
            print(
                "Warm-started protected BC actor with "
                f"{transfer_projection['type']} ({resolved_projection['source_policy_observation_dim']} -> "
                f"{resolved_projection['target_policy_observation_dim']}); critic reset",
                flush=True,
            )
        else:
            runner.alg.load(
                warm_start_state,
                {
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
                strict=True,
            )
        reference_actor = deepcopy(runner.alg.actor).eval()
        if transfer_projection is None:
            print(
                f"Warm-started BC actor and critic from {args.warm_start}; optimizer and iteration reset",
                flush=True,
            )
    if args.fit_output_head_only:
        rows = _freeze_actor_except_beta_output(runner.alg.actor)
        print(f"BC fitting only Beta output rows {rows}", flush=True)
    if args.fit_output_action_dims:
        rows = _freeze_actor_except_action_dims(
            runner.alg.actor, tuple(int(dim) for dim in args.fit_output_action_dims)
        )
        print(f"BC fitting only Beta action dims {list(args.fit_output_action_dims)} rows {rows}", flush=True)

    expert = scripted_pick_place_expert_from_env(env)
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    execution_noise_std = 0.0 if args.execution_noise_std is None else float(args.execution_noise_std)
    noise_generator = None
    if execution_noise_std > 0.0:
        noise_generator = torch.Generator(device=gs.device)
        noise_generator.manual_seed(int(args.execution_noise_seed))
    final_loss = float("nan")
    for round_index in range(int(args.dagger_rounds) + 1):
        driver = None if round_index == 0 else runner.alg.actor
        label = "expert" if round_index == 0 else "student"
        print(
            f"Round {round_index}: collecting {args.rollout_steps} steps x {args.num_envs} envs "
            f"along the {label} trajectory",
            flush=True,
        )
        round_obs, round_actions, round_phases = collect_demonstrations(
            env,
            expert,
            int(args.rollout_steps),
            actor=driver,
            execution_noise_std=execution_noise_std,
            noise_generator=noise_generator,
        )
        observations.append(round_obs)
        actions.append(round_actions)
        phases.append(round_phases)
        dataset_obs = torch.cat(observations)
        dataset_actions = torch.cat(actions)
        dataset_phases = torch.cat(phases)
        dataset_weights = expert_phase_sample_weights(
            dataset_phases,
            near_table_weight=near_table_weight,
            release_retreat_weight=release_weight,
            transport_weight=float(args.transport_phase_weight),
            close_lift_weight=float(args.close_lift_phase_weight),
            balance_phases=bool(args.phase_balanced_weighting),
        )
        phase_counts = {PHASE_NAMES[phase]: int((dataset_phases == phase).sum().item()) for phase in PHASE_NAMES}
        phase_mass = {
            PHASE_NAMES[phase]: float(dataset_weights[dataset_phases == phase].sum().item())
            for phase in PHASE_NAMES
            if bool((dataset_phases == phase).any())
        }
        print(f"  dataset: {dataset_obs.shape[0]} samples", flush=True)
        print(f"  phase samples: {phase_counts}", flush=True)
        print(f"  phase loss mass: {phase_mass}", flush=True)
        action_component_weights = torch.tensor(
            [1.0, float(args.action_y_weight), float(args.action_z_weight), float(args.action_gripper_weight)],
            dtype=torch.float32,
        )
        dataset_reference_actions = None
        reference_mask = None
        if reference_actor is not None and args.reference_anchor_weight > 0.0:
            dataset_reference_actions = _reference_actor_actions(
                reference_actor,
                dataset_obs,
                device=gs.device,
                batch_size=int(args.batch_size),
            )
            # Observation index 29 is the retained ever_grasped latch in every
            # supported 44/47-dimensional policy layout.
            reference_mask = dataset_obs[:, 29] < 0.5
        final_loss, concentration = fit_actor(
            runner.alg.actor,
            dataset_obs,
            dataset_actions,
            dataset_weights,
            device=gs.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            target_concentration=float(args.target_concentration),
            concentration_weight=float(args.concentration_weight),
            reference_actions=dataset_reference_actions,
            reference_mask=reference_mask,
            reference_anchor_weight=float(args.reference_anchor_weight),
            action_component_weights=action_component_weights,
            far_open_penalty=float(args.far_open_penalty),
            carry_near_dist_m=float(env_cfg.get("carry_near_dist_m", 0.040)),
            near_table_down_penalty_weight=float(args.near_table_down_penalty),
            obj_rest_z_m=float(env.obj_rest_z_base),
            near_table_height_m=float(near_table_height_m),
            max_down_action=float(max_down_action),
        )

    if transfer_guard is not None:
        transfer_guard.assert_preserved()
        print(
            f"Protected BC preserved every inherited actor value; trained "
            f"{transfer_guard.trainable_parameter_count} {trainable_label}",
            flush=True,
        )

    checkpoint = log_dir / "model_0.pt"
    runner.save(str(checkpoint))
    write_artifact_inventory(
        log_dir / "artifacts.yaml",
        training_config=artifact,
        checkpoints=[checkpoint],
        selected_checkpoint=checkpoint,
    )
    sampling_std = 2.0 * (0.25 / (concentration + 1.0)) ** 0.5
    print(
        f"Behaviour-cloned actor written to {checkpoint} "
        f"(action mse={final_loss:.6f}, concentration={concentration:.1f}, "
        f"sampling std about {sampling_std:.3f} in action units)"
    )


if __name__ == "__main__":
    main()
