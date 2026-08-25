"""Evaluate an xArm6 + Gripper G2 pick-place checkpoint."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime
from importlib import metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from packaging.version import Version
from ufactory.robots.paths import robot_urdf
from ufactory.training import (
    PICK_PLACE_ACCEPTANCE_PROFILES,
    apply_pick_place_acceptance_profile,
    build_pick_place_task_configs,
    load_scenario_bank,
    scenario_bank_sha256,
    validate_and_load_rsl_checkpoint,
)

try:
    from ufactory.training.policy import enable_actor_output_tanh
except ImportError:
    enable_actor_output_tanh = None  # type: ignore[assignment]

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if Version(metadata.version("rsl-rl-lib")) < Version("5.3.0"):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib>=5.3.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from ufactory.config import load_runtime_config
from ufactory.simulation.compat import require_genesis_runtime

from .env import XArm6PickPlaceEnv
from .expert import scripted_pick_place_expert_from_env
from .trace_utils import (
    ACTION_AXIS_NAMES,
    HARD_EVENT_FIELDS,
    action_noise_axis_mask,
    apply_action_noise_bank,
    build_action_noise_bank,
    confidence_intervals_applicable,
    disable_training_action_noise_for_evaluation,
    env0_trace_row,
    hard_event_trace_row,
    save_rgb_frame,
    scenario_layout_key,
    task_phase_label,
    trace_fieldnames,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = EXAMPLE_DIR / "pretrained" / "model_299_g2stable.pt"
DEFAULT_SCENARIO_BANK = EXAMPLE_DIR / "scenarios" / "fixed_seed17000_n512.json"


def find_latest_checkpoint(log_dir: Path) -> Path:
    """Find the checkpoint with the highest iteration number."""
    pts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    return pts[-1]


def read_latest_metrics(metrics_path: Path) -> dict | None:
    """Read the most recent metrics row, if available."""
    if not metrics_path.exists():
        return None

    latest_row = None
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            latest_row = row
    return latest_row


def infer_eval_stage(metrics_path: Path) -> tuple[int, str]:
    """Infer the curriculum stage from the latest metrics row."""
    latest_metrics = read_latest_metrics(metrics_path)
    if latest_metrics is None:
        return 0, "metrics missing, fallback to stage 0"

    raw_stage = latest_metrics.get("curriculum_stage")
    if raw_stage is None:
        return 0, "curriculum_stage missing in metrics, fallback to stage 0"

    try:
        stage = int(round(float(raw_stage)))
    except ValueError:
        return 0, f"invalid curriculum_stage={raw_stage!r}, fallback to stage 0"

    return stage, f"latest metrics row ({metrics_path})"


def load_runner_checkpoint(
    runner: OnPolicyRunner,
    checkpoint: dict,
    load_optimizer: bool = False,
) -> dict:
    """Load an already integrity-checked RSL-RL checkpoint into a runner."""
    load_cfg = None if load_optimizer else {"actor": True, "critic": True, "optimizer": False}
    load_iteration = runner.alg.load(checkpoint, load_cfg, strict=True)
    if load_iteration:
        runner.current_learning_iteration = int(checkpoint["iter"])
    infos = checkpoint.get("infos")
    return infos if isinstance(infos, dict) else {}


def resolve_report_csv(report_csv: str | None) -> Path | None:
    if not report_csv:
        return None
    if report_csv == "auto":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("reports") / f"pick_place_eval_{stamp}.csv"
    return Path(report_csv)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a Bernoulli rate."""

    if total <= 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def percentile(values: list[float], q: float) -> float:
    """Deterministic linear percentile without a NumPy scalar in JSON output."""

    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _stamp(msg: str) -> None:
    """Print a wall-clock-timestamped progress line so startup phases stay visible."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _resolve_eval_performance_mode(
    args: argparse.Namespace,
    env_cfg: dict,
) -> tuple[bool, str]:
    """Choose Genesis storage mode without mutating the checkpoint environment contract."""
    cli_override = getattr(args, "performance_mode", None)
    if cli_override is not None:
        return bool(cli_override), "CLI override"
    if bool(getattr(args, "headless", False)):
        return (
            bool(env_cfg.get("genesis_performance_mode", False)),
            "saved training config (headless default)",
        )
    # Genesis >=1.3 dynamic arrays can reuse the Python-side kernel cache across
    # processes.  Static fields are worthwhile for long training/headless batches,
    # but add roughly 30 seconds of frontend compilation to every viewer process.
    return False, "viewer fast-start default"


def _stamp_eval_performance_mode(performance_mode: bool, source: str) -> None:
    array_mode = "static arrays / high throughput" if performance_mode else "dynamic arrays / fast startup"
    _stamp(f"Genesis runtime: {array_mode} (performance_mode={performance_mode}, {source})")


def _per_checkpoint_path(path: Path | None, ckpt_path: Path, multi_ckpt: bool) -> Path | None:
    """Suffix an output path with the checkpoint stem when evaluating several checkpoints."""
    if path is None or not multi_ckpt:
        return path
    return path.with_name(f"{path.stem}.{ckpt_path.parent.name}.{ckpt_path.stem}{path.suffix}")


def _load_eval_scenario_bank(args: argparse.Namespace, env_cfg: dict, runtime_config) -> dict | None:
    """Load, validate, and mount one deterministic scenario bank."""

    if args.scenario_bank is None:
        return None
    bank = load_scenario_bank(
        args.scenario_bank,
        expected_runtime_config_sha256=(None if args.allow_runtime_config_mismatch else runtime_config.sha256),
        expected_env=env_cfg,
    )
    env_cfg["evaluation_scenarios"] = bank["scenarios"]
    print(
        f"Scenario bank: {args.scenario_bank} "
        f"({bank['count']} {bank['mode']}, "
        f"sha256={scenario_bank_sha256(args.scenario_bank)[:12]}…)"
    )
    return bank


def _resolve_episode_count(args: argparse.Namespace, scenario_bank: dict | None, *, default: int) -> None:
    if args.episodes is None:
        args.episodes = int(scenario_bank["count"]) if scenario_bank is not None else int(default)
    if scenario_bank is not None and (args.episodes <= 0 or args.episodes > int(scenario_bank["count"])):
        raise ValueError("--episodes must be between 1 and the scenario-bank count")


def _pin_eval_curriculum_stage(env_cfg: dict, stage: int) -> int:
    """Freeze construction and every later reset to one evaluation envelope."""

    pinned_stage = int(stage)
    if not 0 <= pinned_stage <= 4:
        raise ValueError("evaluation curriculum stage must be in [0, 4]")
    env_cfg["curriculum_initial_stage"] = pinned_stage
    env_cfg["curriculum_max_stage"] = pinned_stage
    return pinned_stage


def _acceptance_target_results(stats: dict) -> dict[str, bool]:
    """Resolve immutable aggregate gates from one completed evaluation."""

    episode_count = int(stats["episode_count"])
    if episode_count <= 0:
        return {"standard": False, "robustness": False}
    p99_error = percentile(stats["final_xy_errors_m"], 99)
    p99_drag = percentile(stats["max_pre_lift_xy_values_m"], 99)
    p99_drift = percentile(stats["post_release_drift_values_m"], 99)
    clean_runtime = (
        stats["max_action_clip"] == 0.0 and stats["max_ik_failure"] == 0.0 and stats["max_ik_jump_reject"] == 0.0
    )
    return {
        "standard": (
            stats["success_count"] == episode_count and stats["quality_count"] == episode_count and clean_runtime
        ),
        "robustness": (
            stats["success_count"] / episode_count >= 0.99
            and stats["quality_count"] / episode_count >= 0.99
            and p99_error <= 0.010
            and p99_drag <= 0.005
            and p99_drift <= 0.003
            and clean_runtime
        ),
    }


def _apply_eval_env_overrides(args: argparse.Namespace, env_cfg: dict) -> None:
    """Force the evaluation-side environment contract onto a training env config."""
    env_cfg["num_envs"] = args.num_envs
    saved_train_noise_std = disable_training_action_noise_for_evaluation(env_cfg)
    args.source_train_action_noise_std = saved_train_noise_std
    # A checkpoint may have been trained with execution noise, but evaluation
    # owns the perturbation sequence.  Keeping the saved value here would add a
    # second, unseeded sample inside env.step(), making --action-noise-std=0
    # non-deterministic and --action-noise-std=0.02 effectively sqrt(2)*0.02.
    env_cfg["place_phase_reset_frac"] = float(args.place_phase_reset_frac)
    env_cfg["place_phase_table_reset_frac"] = float(args.place_phase_table_reset_frac)
    # Force BOTH the initial and annealed-final grasp-phase fractions so a fresh eval env
    # (total_env_steps=0 -> anneal returns the initial value) does not silently start
    # episodes at the grasp-bootstrap pose and inflate the grasp rate.
    env_cfg["grasp_phase_reset_frac"] = float(args.grasp_phase_reset_frac)
    env_cfg["grasp_phase_reset_frac_final"] = float(args.grasp_phase_reset_frac)
    env_cfg["carry_phase_reset_frac"] = float(args.carry_phase_reset_frac)
    if args.release_command_margin_m is not None:
        env_cfg["release_command_margin_m"] = float(args.release_command_margin_m)
    args.resolved_acceptance_profile = None
    if args.acceptance_profile is not None:
        args.resolved_acceptance_profile = apply_pick_place_acceptance_profile(
            env_cfg,
            args.acceptance_profile,
        )
        print(
            f"Acceptance profile {args.acceptance_profile}: "
            "xy<=10 mm, table-z<=5 mm, speed<=0.02 m/s, "
            "drag<=5 mm, post-release drift<=3 mm, "
            "landing region=20 mm, "
            f"stable_steps={env_cfg['success_hold_steps']}"
        )
    if args.event_frames_dir is not None:
        env_cfg["capture_camera"] = True
        env_cfg["capture_event_frames"] = True
    print(f"place_phase_reset_frac (eval override): {env_cfg['place_phase_reset_frac']}")
    print(f"place_phase_table_reset_frac (eval override): {env_cfg['place_phase_table_reset_frac']}")
    print(f"carry_phase_reset_frac (eval override): {env_cfg['carry_phase_reset_frac']}")
    print(f"grasp_phase_reset_frac (eval override): {env_cfg['grasp_phase_reset_frac']}")
    if args.release_command_margin_m is not None:
        print(f"release_command_margin_m (eval override): {env_cfg['release_command_margin_m']}")
    print(
        "train_action_noise_std (eval override): "
        f"{saved_train_noise_std:g} -> {env_cfg['train_action_noise_std']:g}; "
        "evaluation noise is applied exactly once"
    )


_EVALUATION_COMPATIBILITY_ENV_KEYS = (
    "runtime_config_sha256",
    "physics_profile",
    "num_obs",
    "num_actions",
    "include_commanded_gap",
    "include_previous_action",
    "include_normalized_layout_offsets",
    "include_scripted_action_hint",
    "include_quality_observations",
    "include_contact_observations",
    "include_ee_setpoint_residual",
    "use_contact_holding",
    "privileged_critic_obs",
    "strict_action_bounds",
    "ctrl_dt",
    "substeps",
    "constraint_solver",
    "solver_iterations",
    "noslip_iterations",
    "friction_cone",
    "contact_resolution",
    "constraint_time_constant_s",
    "use_gjk_collision",
    "action_scale",
    "action_clip",
    "action_response_exponent",
    "max_cartesian_delta_m",
    "max_ik_jump_rad",
    "ee_command_integration",
    "ee_setpoint_leash_m",
    "gripper_delta_mm",
    "gripper_min_command_gap_m",
    "contact_force_scale_n",
    "contact_force_threshold_n",
    "table_height",
    "obj_size",
    "obj_mass_kg",
    "fixed_demo_layout",
    "randomize_target",
    "fixed_obj_pos",
    "fixed_target_pos",
    "workspace_lower",
    "workspace_upper",
    "place_success_dist_m",
    "release_success_dist_m",
    "success_hold_steps",
    "success_table_z_tolerance_m",
    "success_max_obj_speed_m_s",
    "release_height_tolerance_m",
    "release_max_obj_speed_m_s",
    "pre_lift_max_drag_m",
    "post_release_max_drift_m",
    "landing_near_table_height_m",
    "landing_max_xy_speed_m_s",
    "landing_max_down_speed_m_s",
)


def _evaluation_compatibility_signature(args: argparse.Namespace, artifact: dict) -> dict:
    """Return fields that must match when checkpoints share one live scene."""

    env_cfg = deepcopy(artifact["env"])
    disable_training_action_noise_for_evaluation(env_cfg)
    env_cfg["place_phase_reset_frac"] = float(args.place_phase_reset_frac)
    env_cfg["place_phase_table_reset_frac"] = float(args.place_phase_table_reset_frac)
    env_cfg["grasp_phase_reset_frac"] = float(args.grasp_phase_reset_frac)
    env_cfg["grasp_phase_reset_frac_final"] = float(args.grasp_phase_reset_frac)
    env_cfg["carry_phase_reset_frac"] = float(args.carry_phase_reset_frac)
    if args.release_command_margin_m is not None:
        env_cfg["release_command_margin_m"] = float(args.release_command_margin_m)
    if args.acceptance_profile is not None:
        apply_pick_place_acceptance_profile(env_cfg, args.acceptance_profile)
    return {
        "env": {key: env_cfg.get(key) for key in _EVALUATION_COMPATIBILITY_ENV_KEYS},
        "actor": artifact["train"]["actor"],
        "critic": artifact["train"]["critic"],
        "obs_groups": artifact["train"]["obs_groups"],
    }


def _run_expert_eval(args: argparse.Namespace) -> None:
    """Score the scripted expert on the same strict contract used for checkpoints.

    This is a reachability probe: the expert has perfect state access, so a failure
    here is a property of the environment (action limits, gripper command range,
    contact physics) rather than of any learned policy.
    """
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs(
        "xarm6",
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )
    runtime_config = load_runtime_config("xarm6", task="pick_place", config_path=args.runtime_config)
    scenario_bank = _load_eval_scenario_bank(args, env_cfg, runtime_config)
    _resolve_episode_count(args, scenario_bank, default=8)
    pinned_stage = _pin_eval_curriculum_stage(env_cfg, args.stage if args.stage is not None else 0)
    _apply_eval_env_overrides(args, env_cfg)
    performance_mode, performance_mode_source = _resolve_eval_performance_mode(args, env_cfg)
    _stamp_eval_performance_mode(performance_mode, performance_mode_source)

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning" if args.headless else "info",
        seed=int(args.seed) if args.seed is not None else 1,
        performance_mode=performance_mode,
    )
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=not args.headless,
    )
    _stamp("Scene built (one-off kernel compilation done)")
    env.curriculum_stage = pinned_stage

    expert = scripted_pick_place_expert_from_env(env)
    print(
        "Scripted expert: "
        f"pre_grasp_gap={1000 * expert.pre_grasp_gap_m:.1f} mm, "
        f"close_gap={1000 * expert.close_gap_m:.1f} mm, "
        f"action_scale={env.action_scale}, max_delta={env.max_cartesian_delta_m}"
    )

    stats = _run_eval_loop(
        args,
        env,
        expert,
        env.get_observations(),
        scenario_bank,
        resolve_report_csv(args.report_csv),
        Path(args.trace_csv) if args.trace_csv else None,
        args.event_frames_dir,
    )
    _print_summary(args, scenario_bank, stats)
    print(f"  Expert phase distribution: {expert.phase_counts()}")
    if args.summary_json is not None:
        _write_summary_json(
            args.summary_json,
            args,
            {"seed": int(args.seed) if args.seed is not None else 1},
            None,
            runtime_config,
            scenario_bank,
            stats,
            Path("scripted-expert"),
            None,
        )
    if args.require_target is not None and not _acceptance_target_results(stats)[args.require_target]:
        raise SystemExit(2)


def _build_policy(args: argparse.Namespace, runner: OnPolicyRunner):
    """Build the inference policy for the currently loaded checkpoint."""
    policy = runner.get_inference_policy(device=gs.device)
    if args.beta_point_estimate == "mode":
        actor = runner.alg.actor
        distribution = actor.distribution
        if distribution is None or distribution.__class__.__name__ != "BetaDistribution":
            raise ValueError("--beta-point-estimate=mode requires a BetaDistribution actor")
        action_low, action_high = (float(value) for value in distribution.action_range)

        def beta_mode_policy(observations):
            latent = actor.get_latent(observations)
            base_raw_output = getattr(actor, "base_raw_output", None)
            raw_output = base_raw_output(latent) if callable(base_raw_output) else actor.mlp(latent)
            raw_alpha, raw_beta = torch.unbind(raw_output, dim=-2)
            alpha_excess = torch.nn.functional.softplus(raw_alpha)
            beta_excess = torch.nn.functional.softplus(raw_beta)
            concentration_excess = alpha_excess + beta_excess
            unit_mode = torch.where(
                concentration_excess > 1e-8,
                alpha_excess / concentration_excess.clamp_min(1e-8),
                torch.full_like(concentration_excess, 0.5),
            )
            base_action = (unit_mode * (action_high - action_low) + action_low).clamp(action_low, action_high)
            select_guided_action = getattr(actor, "select_guided_action", None)
            if callable(select_guided_action):
                return select_guided_action(latent, base_action)
            return base_action

        policy = beta_mode_policy
    return policy


def _run_eval_loop(
    args,
    env,
    policy,
    obs,
    scenario_bank,
    report_path,
    trace_path,
    event_frames_dir,
) -> dict:
    """Roll out episodes with ``policy`` and return per-checkpoint statistics."""
    extras = env.extras
    episode_count = 0
    total_reward = torch.zeros(args.num_envs, device=gs.device)
    episode_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=gs.device)
    episode_rewards = []

    grasp_count = 0
    lift_count = 0
    place_count = 0
    success_count = 0
    action_clip_fractions = []
    action_near_bound_fractions = []
    ik_failure_fractions = []
    ik_jump_reject_fractions = []
    completed_scenario_ids = []
    final_xy_errors_m = []
    max_pre_lift_xy_values_m = []
    post_release_drift_values_m = []
    release_xy_values_m = []
    release_height_values_m = []
    release_speed_values_m_s = []
    landing_xy_speed_values_m_s = []
    landing_down_speed_values_m_s = []
    hard_landing_first_steps = [-1] * args.num_envs
    hard_landing_first_phases = ["none"] * args.num_envs
    hard_landing_phase_counts: dict[str, int] = {}
    quality_count = 0
    post_release_recontact_count = 0
    unique_layouts: set[tuple[float, ...]] = set()
    action_noise_mask = action_noise_axis_mask(args.action_noise_axes)
    action_noise_bank = None
    noise_episode_ids = torch.arange(
        args.num_envs,
        dtype=torch.int64,
        device=gs.device,
    )
    next_noise_episode_id = int(args.num_envs)
    if args.action_noise_std > 0.0:
        bank_spec = str(args.action_noise_seed) if args.action_noise_bank is None else args.action_noise_bank
        action_noise_bank = build_action_noise_bank(
            bank_spec,
            episode_count=int(args.episodes) + int(args.num_envs),
            max_steps=int(env.max_episode_length) + 1,
            action_dim=int(env.num_actions),
        )
        print(f"Action-noise bank: {action_noise_bank.source}, sha256={action_noise_bank.sha256[:12]}…")
    captured_events: set[tuple[int, str]] = set()
    event_fields = (
        ("first_push_event", "first_push"),
        ("first_lift_event", "first_lift"),
        ("release_event", "release_start"),
        ("table_contact_event", "first_table_contact"),
        ("final_stable_event", "final_stable"),
    )

    report_file = None
    report_writer = None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_file = report_path.open("w", newline="")
        report_writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "episode",
                "action_noise_episode_id",
                "reward",
                "steps",
                "grasped",
                "lifted",
                "placed",
                "success",
                "curriculum_stage",
                "scenario_id",
                "obj_x",
                "obj_y",
                "obj_z",
                "target_x",
                "target_y",
                "target_z",
                "final_obj_x",
                "final_obj_y",
                "final_obj_z",
                "final_xy_error_m",
                "final_obj_speed_m_s",
                "final_gripper_gap_m",
                "max_pre_lift_xy_m",
                "release_started",
                "release_valid",
                "release_violation",
                "release_xy_dist_m",
                "release_height_error_m",
                "release_speed_m_s",
                "max_landing_xy_speed_m_s",
                "max_landing_down_speed_m_s",
                "hard_landing_violation",
                "hard_landing_first_step",
                "hard_landing_first_phase",
                "post_release_drift_m",
                "post_release_clearance_m",
                "post_release_recontact",
                "quality_ok",
                "action_near_bound_fraction",
                "action_clip_fraction",
                "ik_failure_fraction",
                "ik_jump_reject_fraction",
            ],
        )
        report_writer.writeheader()
        print(f"Report CSV: {report_path}")

    trace_file = None
    trace_writer = None
    trace_step = 0
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", newline="")
        trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fieldnames(env))
        trace_writer.writeheader()
        print(f"Trace CSV (env 0): {trace_path}")

    hard_event_path = (
        Path(args.hard_event_csv)
        if args.hard_event_csv is not None
        else (None if report_path is None else report_path.with_name(f"{report_path.stem}.hard_events.csv"))
    )
    hard_event_file = None
    hard_event_writer = None
    hard_event_count = 0
    if hard_event_path is not None:
        hard_event_path.parent.mkdir(parents=True, exist_ok=True)
        hard_event_file = hard_event_path.open("w", newline="")
        hard_event_writer = csv.DictWriter(
            hard_event_file,
            fieldnames=HARD_EVENT_FIELDS,
        )
        hard_event_writer.writeheader()
        print(f"Hard-event CSV: {hard_event_path}")

    try:
        while True:
            with torch.no_grad():
                policy_actions = policy(obs)
                actions = policy_actions
                if args.action_noise_std > 0.0:
                    if action_noise_bank is None:  # pragma: no cover - guarded above.
                        raise RuntimeError("action-noise bank was not initialized")
                    actions = apply_action_noise_bank(
                        actions,
                        std=float(args.action_noise_std),
                        bank=action_noise_bank,
                        episode_ids=noise_episode_ids,
                        step_indices=episode_steps.to(dtype=torch.int64),
                        action_clip=float(env.action_clip),
                        axis_mask=action_noise_mask,
                    )
            obs, reward, done, extras = env.step(actions)
            total_reward += reward
            episode_steps += 1
            snapshot = extras["step_snapshot"]
            new_hard_envs = snapshot["hard_landing_event"].nonzero(as_tuple=True)[0]
            for idx_t in new_hard_envs:
                idx = int(idx_t.item())
                if hard_landing_first_steps[idx] < 0:
                    hard_landing_first_steps[idx] = int(episode_steps[idx].item())
                    hard_landing_first_phases[idx] = task_phase_label(snapshot, idx)
                if hard_event_writer is not None:
                    hard_event_writer.writerow(
                        hard_event_trace_row(
                            snapshot,
                            index=idx,
                            step=int(episode_steps[idx].item()),
                            action_noise_episode_id=int(noise_episode_ids[idx].item()),
                            policy_action=policy_actions[idx],
                            executed_action=actions[idx],
                        )
                    )
                    hard_event_count += 1
            if hard_event_file is not None and len(new_hard_envs) > 0:
                hard_event_file.flush()

            if trace_writer is not None:
                trace_step += 1
                trace_writer.writerow(
                    env0_trace_row(
                        env,
                        episode=episode_count + 1,
                        step=trace_step,
                        reward_value=float(reward[0].item()),
                        done_flag=bool(done[0].item()),
                        action=actions[0],
                        policy_action=policy_actions[0],
                    )
                )
                trace_file.flush()
                if bool(done[0].item()):
                    trace_step = 0

            if event_frames_dir is not None:
                frame = extras.get("event_frame_rgb")
                current_episode = episode_count + 1
                for field, label in event_fields:
                    key = (current_episode, label)
                    if bool(snapshot[field][0].item()) and key not in captured_events:
                        if frame is None:
                            raise RuntimeError("environment did not freeze the pre-reset event frame")
                        frame_path = event_frames_dir / f"episode_{current_episode:04d}_{label}.png"
                        save_rgb_frame(frame, frame_path)
                        captured_events.add(key)

            done_envs = done.nonzero(as_tuple=True)[0]
            if len(done_envs) > 0:
                for idx_t in done_envs:
                    if args.episodes > 0 and episode_count >= args.episodes:
                        break
                    idx = int(idx_t.item())
                    action_noise_episode_id = int(noise_episode_ids[idx].item())
                    scenario_id = int(extras["episode_scenario_id"][idx].item())
                    if scenario_bank is not None and scenario_id < 0:
                        continue
                    ep_reward = total_reward[idx].item()
                    ep_steps = int(episode_steps[idx].item())
                    ep_grasped = bool(extras["episode_grasp_success"][idx].item())
                    ep_lifted = bool(extras["episode_lift_success"][idx].item())
                    ep_placed = bool(extras["episode_place_success"][idx].item())
                    ep_success = bool(extras["episode_success"][idx].item())
                    initial_obj = extras["episode_initial_obj_pos"][idx]
                    target = extras["episode_target_pos"][idx]
                    final_obj = extras["episode_final_obj_pos"][idx]
                    final_speed = float(torch.norm(extras["episode_final_obj_vel"][idx]).item())
                    final_gap = float(extras["episode_final_gripper_gap"][idx].item())
                    action_clip_fraction = float(extras["episode_action_clip_fraction"][idx].item())
                    action_near_bound_fraction = float(extras["episode_action_near_bound_fraction"][idx].item())
                    ik_failure_fraction = float(extras["episode_ik_failure_fraction"][idx].item())
                    ik_jump_reject_fraction = float(extras["episode_ik_jump_reject_fraction"][idx].item())
                    final_xy_error_m = float(torch.norm(final_obj[:2] - target[:2]).item())
                    max_pre_lift_xy_m = float(extras["episode_max_pre_lift_xy_m"][idx].item())
                    release_started = bool(extras["episode_release_started"][idx].item())
                    release_valid = bool(extras["episode_release_valid"][idx].item())
                    release_violation = bool(extras["episode_release_violation"][idx].item())
                    release_xy_dist_m = float(extras["episode_release_xy_dist_m"][idx].item())
                    release_height_error_m = float(extras["episode_release_height_error_m"][idx].item())
                    release_speed_m_s = float(extras["episode_release_speed_m_s"][idx].item())
                    max_landing_xy_speed_m_s = float(extras["episode_max_landing_xy_speed_m_s"][idx].item())
                    max_landing_down_speed_m_s = float(extras["episode_max_landing_down_speed_m_s"][idx].item())
                    hard_landing_violation = bool(extras["episode_hard_landing_violation"][idx].item())
                    hard_landing_first_step = hard_landing_first_steps[idx]
                    hard_landing_first_phase = hard_landing_first_phases[idx]
                    if hard_landing_violation:
                        if hard_landing_first_step < 0:
                            hard_landing_first_phase = "untracked"
                        hard_landing_phase_counts[hard_landing_first_phase] = (
                            hard_landing_phase_counts.get(hard_landing_first_phase, 0) + 1
                        )
                    post_release_drift_m = float(extras["episode_post_release_drift_m"][idx].item())
                    post_release_clearance_m = float(extras["episode_post_release_clearance_m"][idx].item())
                    post_release_recontact = bool(extras["episode_post_release_recontact"][idx].item())
                    quality_ok = bool(extras["episode_quality_ok"][idx].item())
                    episode_rewards.append(ep_reward)
                    if scenario_id >= 0:
                        completed_scenario_ids.append(scenario_id)
                    action_clip_fractions.append(action_clip_fraction)
                    action_near_bound_fractions.append(action_near_bound_fraction)
                    ik_failure_fractions.append(ik_failure_fraction)
                    ik_jump_reject_fractions.append(ik_jump_reject_fraction)
                    final_xy_errors_m.append(final_xy_error_m)
                    max_pre_lift_xy_values_m.append(max_pre_lift_xy_m)
                    post_release_drift_values_m.append(post_release_drift_m)
                    release_xy_values_m.append(release_xy_dist_m)
                    release_height_values_m.append(release_height_error_m)
                    release_speed_values_m_s.append(release_speed_m_s)
                    landing_xy_speed_values_m_s.append(max_landing_xy_speed_m_s)
                    landing_down_speed_values_m_s.append(max_landing_down_speed_m_s)
                    quality_count += int(quality_ok)
                    post_release_recontact_count += int(post_release_recontact)
                    unique_layouts.add(scenario_layout_key(initial_obj, target))
                    episode_count += 1
                    grasp_count += int(ep_grasped)
                    lift_count += int(ep_lifted)
                    place_count += int(ep_placed)
                    success_count += int(ep_success)

                    row = {
                        "episode": episode_count,
                        "action_noise_episode_id": action_noise_episode_id,
                        "reward": ep_reward,
                        "steps": ep_steps,
                        "grasped": int(ep_grasped),
                        "lifted": int(ep_lifted),
                        "placed": int(ep_placed),
                        "success": int(ep_success),
                        "curriculum_stage": env.curriculum_stage,
                        "scenario_id": scenario_id,
                        "obj_x": float(initial_obj[0].item()),
                        "obj_y": float(initial_obj[1].item()),
                        "obj_z": float(initial_obj[2].item()),
                        "target_x": float(target[0].item()),
                        "target_y": float(target[1].item()),
                        "target_z": float(target[2].item()),
                        "final_obj_x": float(final_obj[0].item()),
                        "final_obj_y": float(final_obj[1].item()),
                        "final_obj_z": float(final_obj[2].item()),
                        "final_xy_error_m": final_xy_error_m,
                        "final_obj_speed_m_s": final_speed,
                        "final_gripper_gap_m": final_gap,
                        "max_pre_lift_xy_m": max_pre_lift_xy_m,
                        "release_started": int(release_started),
                        "release_valid": int(release_valid),
                        "release_violation": int(release_violation),
                        "release_xy_dist_m": release_xy_dist_m,
                        "release_height_error_m": release_height_error_m,
                        "release_speed_m_s": release_speed_m_s,
                        "max_landing_xy_speed_m_s": max_landing_xy_speed_m_s,
                        "max_landing_down_speed_m_s": max_landing_down_speed_m_s,
                        "hard_landing_violation": int(hard_landing_violation),
                        "hard_landing_first_step": hard_landing_first_step,
                        "hard_landing_first_phase": hard_landing_first_phase,
                        "post_release_drift_m": post_release_drift_m,
                        "post_release_clearance_m": post_release_clearance_m,
                        "post_release_recontact": int(post_release_recontact),
                        "quality_ok": int(quality_ok),
                        "action_near_bound_fraction": action_near_bound_fraction,
                        "action_clip_fraction": action_clip_fraction,
                        "ik_failure_fraction": ik_failure_fraction,
                        "ik_jump_reject_fraction": ik_jump_reject_fraction,
                    }
                    if report_writer is not None:
                        report_writer.writerow(row)
                    if args.episode_log_interval > 0 and episode_count % args.episode_log_interval == 0:
                        print(
                            f"  Episode {episode_count}: "
                            f"scenario={scenario_id}, reward={ep_reward:.1f}, steps={ep_steps}, "
                            f"grasped={'Yes' if ep_grasped else 'No'}, "
                            f"lifted={'Yes' if ep_lifted else 'No'}, "
                            f"placed={'Yes' if ep_placed else 'No'}, "
                            f"quality={'Yes' if quality_ok else 'No'}, "
                            f"success={'Yes' if ep_success else 'No'}, "
                            f"error={1000 * final_xy_error_m:.1f} mm"
                        )

                total_reward[done_envs] = 0.0
                episode_steps[done_envs] = 0
                for idx_t in done_envs:
                    idx = int(idx_t.item())
                    hard_landing_first_steps[idx] = -1
                    hard_landing_first_phases[idx] = "none"
                    if next_noise_episode_id < int(args.episodes) + int(args.num_envs):
                        noise_episode_ids[idx] = next_noise_episode_id
                        next_noise_episode_id += 1

                if args.episodes > 0 and episode_count >= args.episodes:
                    break
    finally:
        if report_file is not None:
            report_file.close()
        if trace_file is not None:
            trace_file.close()
        if hard_event_file is not None:
            hard_event_file.close()

    return {
        "episode_count": episode_count,
        "episode_rewards": episode_rewards,
        "grasp_count": grasp_count,
        "lift_count": lift_count,
        "place_count": place_count,
        "success_count": success_count,
        "completed_scenario_ids": completed_scenario_ids,
        "quality_count": quality_count,
        "post_release_recontact_count": post_release_recontact_count,
        "unique_scenario_count": len(unique_layouts),
        "final_xy_errors_m": final_xy_errors_m,
        "max_pre_lift_xy_values_m": max_pre_lift_xy_values_m,
        "post_release_drift_values_m": post_release_drift_values_m,
        "release_xy_values_m": release_xy_values_m,
        "release_height_values_m": release_height_values_m,
        "release_speed_values_m_s": release_speed_values_m_s,
        "landing_xy_speed_values_m_s": landing_xy_speed_values_m_s,
        "landing_down_speed_values_m_s": landing_down_speed_values_m_s,
        "hard_landing_phase_counts": hard_landing_phase_counts,
        "captured_event_frame_count": len(captured_events),
        "hard_event_trace_count": hard_event_count,
        "max_action_clip": max(action_clip_fractions, default=0.0),
        "max_action_near_bound": max(action_near_bound_fractions, default=0.0),
        "max_ik_failure": max(ik_failure_fractions, default=0.0),
        "max_ik_jump_reject": max(ik_jump_reject_fractions, default=0.0),
        "action_noise_bank_source": (None if action_noise_bank is None else action_noise_bank.source),
        "action_noise_bank_sha256": (None if action_noise_bank is None else action_noise_bank.sha256),
    }


def _print_summary(args, scenario_bank, stats: dict) -> None:
    """Print the per-checkpoint evaluation summary."""
    episode_count = stats["episode_count"]
    episode_rewards = stats["episode_rewards"]
    if not episode_rewards:
        return
    if scenario_bank is not None:
        expected_ids = list(range(args.episodes))
        if sorted(stats["completed_scenario_ids"]) != expected_ids:
            raise RuntimeError(
                "scenario bank was not evaluated exactly once: "
                f"expected IDs 0..{args.episodes - 1}, got {sorted(stats['completed_scenario_ids'])}"
            )
    grasp_count = stats["grasp_count"]
    lift_count = stats["lift_count"]
    place_count = stats["place_count"]
    success_count = stats["success_count"]
    quality_count = stats["quality_count"]
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    unique_count = stats["unique_scenario_count"]
    show_ci = confidence_intervals_applicable(
        unique_scenario_count=unique_count,
        episode_count=episode_count,
        action_noise_std=float(args.action_noise_std),
    )

    def outcome_line(label: str, count: int) -> str:
        base = f"  {label:<18}{count}/{episode_count} ({100 * count / episode_count:.1f}%)"
        if not show_ci:
            return base
        ci = wilson_interval(count, episode_count)
        return f"{base}, 95% CI {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%"

    p99_error = percentile(stats["final_xy_errors_m"], 99)
    p99_drag = percentile(stats["max_pre_lift_xy_values_m"], 99)
    p99_drift = percentile(stats["post_release_drift_values_m"], 99)
    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary ({episode_count} episodes):")
    print(f"  Unique scenarios: {unique_count}")
    if not show_ci:
        print("  Confidence interval: omitted (deterministic repeats share one scenario)")
    print(f"  Avg reward:       {avg_reward:.1f}")
    print(outcome_line("Grasp success:", grasp_count))
    print(outcome_line("Lift success:", lift_count))
    print(outcome_line("Place success:", place_count))
    print(outcome_line("Quality pass:", quality_count))
    print(outcome_line("Full success:", success_count))
    print(f"  Post-release recontact: {stats['post_release_recontact_count']}/{episode_count}")
    print(f"  P99 final error:  {1000 * p99_error:.2f} mm")
    print(f"  P99 pre-lift drag:{1000 * p99_drag:.2f} mm")
    print(f"  P99 post-release: {1000 * p99_drift:.2f} mm")
    if stats["hard_landing_phase_counts"]:
        phase_counts = ", ".join(
            f"{phase}={count}" for phase, count in sorted(stats["hard_landing_phase_counts"].items())
        )
        print(f"  First hard-event phases: {phase_counts}")
    if args.action_noise_std > 0.0:
        print(
            "  Explicit noise:    "
            f"std={args.action_noise_std:g}, axes={','.join(args.action_noise_axes)}, "
            f"bank={stats['action_noise_bank_source']}, "
            f"sha256={stats['action_noise_bank_sha256'][:12]}…"
        )
    targets = _acceptance_target_results(stats)
    print(f"  Standard target:  {'PASS' if targets['standard'] else 'FAIL'}")
    print(f"  Robustness target:{'PASS' if targets['robustness'] else 'FAIL'}")
    if args.require_target is not None:
        print(f"  Required target:  {args.require_target} {'PASS' if targets[args.require_target] else 'FAIL'}")
    print(f"  Max near-bound action: {stats['max_action_near_bound']:.6f}")
    print(f"  Max action clip:  {stats['max_action_clip']:.6f}")
    print(f"  Max IK failure:   {stats['max_ik_failure']:.6f}")
    print(f"  Max IK jump reject: {stats['max_ik_jump_reject']:.6f}")
    print(f"{'=' * 60}")


def _write_summary_json(
    summary_path: Path,
    args,
    train_cfg: dict,
    artifact: dict,
    runtime_config,
    scenario_bank,
    stats: dict,
    ckpt_path: Path,
    manifest,
) -> None:
    """Write the machine-readable aggregate report for one checkpoint."""
    episode_count = stats["episode_count"]
    if episode_count <= 0:
        return
    grasp_count = stats["grasp_count"]
    lift_count = stats["lift_count"]
    place_count = stats["place_count"]
    success_count = stats["success_count"]
    quality_count = stats["quality_count"]
    unique_count = stats["unique_scenario_count"]
    include_wilson = confidence_intervals_applicable(
        unique_scenario_count=unique_count,
        episode_count=episode_count,
        action_noise_std=float(args.action_noise_std),
    )

    def outcome(count: int) -> dict:
        return {
            "count": count,
            "rate": count / episode_count,
            "wilson95": wilson_interval(count, episode_count) if include_wilson else None,
        }

    p99_error = percentile(stats["final_xy_errors_m"], 99)
    p99_drag = percentile(stats["max_pre_lift_xy_values_m"], 99)
    p99_drift = percentile(stats["post_release_drift_values_m"], 99)
    target_results = _acceptance_target_results(stats)
    summary = {
        "schema_version": 4,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": None if manifest is None else manifest.checkpoint_sha256,
        "training_config_sha256": None if artifact is None else artifact["config_sha256"],
        "runtime_config_sha256": None if runtime_config is None else runtime_config.sha256,
        "scenario_bank": (
            None
            if scenario_bank is None
            else {
                "path": str(args.scenario_bank),
                "sha256": scenario_bank_sha256(args.scenario_bank),
                "mode": scenario_bank["mode"],
                "count": scenario_bank["count"],
                "unique_count": unique_count,
            }
        ),
        "seed": int(train_cfg["seed"]),
        "num_envs": int(args.num_envs),
        "episodes": episode_count,
        "strict_success": args.acceptance_profile is not None,
        "acceptance_profile": args.acceptance_profile,
        "required_target": args.require_target,
        "required_target_pass": (None if args.require_target is None else target_results[args.require_target]),
        "resolved_acceptance_profile": getattr(
            args,
            "resolved_acceptance_profile",
            None,
        ),
        "beta_point_estimate": args.beta_point_estimate,
        "action_noise_std": float(args.action_noise_std),
        "action_noise_seed": int(args.action_noise_seed),
        "action_noise_bank": {
            "source": stats["action_noise_bank_source"],
            "sha256": stats["action_noise_bank_sha256"],
        },
        "action_noise_axes": list(args.action_noise_axes),
        "action_noise_std_by_axis": {
            axis: (float(args.action_noise_std) if axis in args.action_noise_axes else 0.0)
            for axis in ACTION_AXIS_NAMES
        },
        "source_training_action_noise_std": float(
            artifact["env"].get("train_action_noise_std", 0.0)
            if artifact is not None
            else getattr(args, "source_train_action_noise_std", 0.0)
        ),
        "environment_action_noise_std": 0.0,
        "unique_scenario_count": unique_count,
        "wilson_intervals_applicable": include_wilson,
        "average_reward": sum(stats["episode_rewards"]) / episode_count,
        "outcomes": {
            "grasp": outcome(grasp_count),
            "lift": outcome(lift_count),
            "place": outcome(place_count),
            "quality": outcome(quality_count),
            "success": outcome(success_count),
        },
        "quality": {
            "p99_final_xy_error_m": p99_error,
            "p99_pre_lift_drag_m": p99_drag,
            "p99_post_release_drift_m": p99_drift,
            "p99_release_xy_dist_m": percentile(stats["release_xy_values_m"], 99),
            "p99_release_height_error_m": percentile(
                stats["release_height_values_m"],
                99,
            ),
            "p99_release_speed_m_s": percentile(stats["release_speed_values_m_s"], 99),
            "p99_landing_xy_speed_m_s": percentile(
                stats["landing_xy_speed_values_m_s"],
                99,
            ),
            "p99_landing_down_speed_m_s": percentile(
                stats["landing_down_speed_values_m_s"],
                99,
            ),
            "standard_target_pass": target_results["standard"],
            "robustness_target_pass": target_results["robustness"],
        },
        "diagnostics": {
            "max_action_near_bound_fraction": stats["max_action_near_bound"],
            "max_action_clip_fraction": stats["max_action_clip"],
            "max_ik_failure_fraction": stats["max_ik_failure"],
            "max_ik_jump_reject_fraction": stats["max_ik_jump_reject"],
            "captured_event_frame_count": stats["captured_event_frame_count"],
            "hard_landing_first_phase_counts": stats["hard_landing_phase_counts"],
            "hard_event_trace_count": stats["hard_event_trace_count"],
            "post_release_recontact_count": stats["post_release_recontact_count"],
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary JSON: {summary_path}")


def _evaluate_checkpoint(
    args,
    *,
    env,
    runner: OnPolicyRunner,
    train_cfg: dict,
    artifact: dict,
    runtime_config,
    scenario_bank,
    ckpt_path: Path,
    checkpoint: dict,
    manifest,
    pinned_stage: int,
    is_first: bool,
    multi_ckpt: bool,
) -> bool:
    """Load one checkpoint into the shared runner and evaluate it without rebuilding the scene."""
    load_runner_checkpoint(runner, checkpoint, load_optimizer=False)
    _stamp(f"Checkpoint loaded: {ckpt_path}")
    policy = _build_policy(args, runner)
    if is_first:
        # XArm6PickPlaceEnv performs its initial reset in __init__. Resetting again
        # here would silently skip the first num_envs entries of a scenario bank.
        obs = env.get_observations()
    else:
        if scenario_bank is not None:
            # Each checkpoint replays the full bank: rewind the cursor so every scenario
            # is still assigned exactly once per checkpoint.
            env._scenario_cursor = 0
        obs, _ = env.reset()
        # reset_idx may advance the curriculum; keep the eval stage pinned.
        env.curriculum_stage = pinned_stage

    report_path = _per_checkpoint_path(resolve_report_csv(args.report_csv), ckpt_path, multi_ckpt)
    trace_path = _per_checkpoint_path(Path(args.trace_csv) if args.trace_csv else None, ckpt_path, multi_ckpt)
    summary_path = _per_checkpoint_path(args.summary_json, ckpt_path, multi_ckpt)
    event_frames_dir = args.event_frames_dir
    if event_frames_dir is not None and multi_ckpt:
        event_frames_dir = event_frames_dir / ckpt_path.stem

    stats = _run_eval_loop(
        args,
        env,
        policy,
        obs,
        scenario_bank,
        report_path,
        trace_path,
        event_frames_dir,
    )
    _print_summary(args, scenario_bank, stats)
    if summary_path is not None:
        _write_summary_json(
            summary_path,
            args,
            train_cfg,
            artifact,
            runtime_config,
            scenario_bank,
            stats,
            ckpt_path,
            manifest,
        )
    targets = _acceptance_target_results(stats)
    return args.require_target is None or targets[args.require_target]


def main():
    parser = argparse.ArgumentParser(description="xArm 6 Pick-Place Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        default=None,
        help=(
            "Path to a complete model checkpoint bundle (.pt + config + manifest). "
            f"Default: the bundled {DEFAULT_CHECKPOINT.name}. Repeat the flag to evaluate "
            "several checkpoints in one process: the scene is built (and kernels compiled) "
            "only once, later checkpoints swap in ~0.1 s"
        ),
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "-B",
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments to visualize",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run (default: scenario-bank size, otherwise 10; 0 = infinite)",
    )
    parser.add_argument("--seed", type=int, help="Override the artifact seed for evaluation")
    parser.add_argument(
        "--scenario-bank",
        type=Path,
        help=(
            "Deterministic JSON scenario bank; every scenario is assigned exactly once. "
            f"The published 512-episode bank is {DEFAULT_SCENARIO_BANK}"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional machine-readable aggregate report",
    )
    parser.add_argument(
        "--strict-success",
        action="store_true",
        help=("Deprecated alias for --acceptance-profile contact_v1"),
    )
    parser.add_argument(
        "--acceptance-profile",
        choices=PICK_PLACE_ACCEPTANCE_PROFILES,
        help=(
            "Apply an immutable evaluation profile instead of taking quality limits "
            "from the candidate's training recipe"
        ),
    )
    parser.add_argument(
        "--require-target",
        choices=("standard", "robustness"),
        help=(
            "Exit non-zero unless the selected aggregate gate passes. This also "
            "enables the immutable contact_v1 profile when none is specified."
        ),
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Add a reproducible Gaussian sequence to executed actions before clipping",
    )
    parser.add_argument(
        "--action-noise-seed",
        type=int,
        default=20260817,
        help="Compatibility seed used when --action-noise-bank is omitted",
    )
    parser.add_argument(
        "--action-noise-bank",
        help=(
            "Integer seed or .npz file containing 'noise'. Samples are indexed by "
            "episode and step, so batch size and checkpoint order cannot change them."
        ),
    )
    parser.add_argument(
        "--action-noise-axes",
        nargs="+",
        choices=ACTION_AXIS_NAMES,
        default=list(ACTION_AXIS_NAMES),
        metavar="AXIS",
        help=(
            "Action coordinates receiving --action-noise-std. Default: "
            "x y z gripper; subsets enable paired axis ablations."
        ),
    )
    parser.add_argument(
        "--event-frames-dir",
        type=Path,
        help=(
            "With -B 1, save first-push, first-lift, release-start, first-table-contact, and final-stable PNG frames"
        ),
    )
    parser.add_argument(
        "--beta-point-estimate",
        choices=("mean", "mode"),
        default="mean",
        help=("Deterministic Beta action for deployment evaluation: RSL-RL's mean or the distribution mode"),
    )
    parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print every Nth completed episode (use 0 to suppress per-episode output)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Curriculum stage for eval (default: infer from latest metrics.csv row)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run evaluation without opening the Genesis viewer",
    )
    parser.add_argument(
        "--performance-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override Genesis static-array performance mode. Viewer default: disabled "
            "for fast startup; headless default: value saved in the training config."
        ),
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Optional episode CSV path; use 'auto' for reports/pick_place_eval_<timestamp>.csv",
    )
    parser.add_argument(
        "--allow-runtime-config-mismatch",
        action="store_true",
        default=False,
        help=(
            "Skip runtime_config_sha256 check so older checkpoints can still be visualized "
            "after task YAML changes (e.g. grasp_gap_m). Policy metrics may not match new config."
        ),
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        default=False,
        help=(
            "Preview the pick_place scene pinned at home (no checkpoint). "
            "Opens the viewer; use --episodes 0 to hold until Ctrl+C."
        ),
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        default=False,
        help=(
            "Score the scripted state-machine expert instead of a checkpoint. "
            "Probes whether the strict targets are reachable in the current env."
        ),
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parent / "recipe.yaml",
        help="Recipe used with --hold/--expert (default: the published recipe.yaml)",
    )
    parser.add_argument(
        "--place-phase-reset-frac",
        type=float,
        default=0.0,
        help=(
            "Override env place_phase_reset_frac for evaluation. "
            "Default 0.0 = fair eval from home (no place-phase bootstrap)."
        ),
    )
    parser.add_argument(
        "--release-command-margin-m",
        type=float,
        default=None,
        help=(
            "Override env release_command_margin_m for evaluation: how far above the "
            "close gap the commanded gap must climb (together with an open-direction "
            "gripper action) before a release is latched. Use to validate release-intent "
            "hysteresis against checkpoints saved with a different margin."
        ),
    )
    parser.add_argument(
        "--place-phase-table-reset-frac",
        type=float,
        default=0.0,
        help=(
            "Within place-phase diagnostic resets, choose the fraction that starts "
            "table-ready. Default 0.0 forces the high set-down trajectory; use 1.0 "
            "to isolate release from a table-stable held state."
        ),
    )
    parser.add_argument(
        "--grasp-phase-reset-frac",
        type=float,
        default=0.0,
        help=(
            "Override env grasp_phase_reset_frac (and its annealed final) for evaluation. "
            "Default 0.0 = fair eval from home (no grasp-phase bootstrap). The saved training "
            "config ships a non-zero value for the reverse curriculum, so this MUST be forced "
            "to 0.0 to measure the honest learned-from-home grasp rate."
        ),
    )
    parser.add_argument(
        "--carry-phase-reset-frac",
        type=float,
        default=0.0,
        help=(
            "Override env carry_phase_reset_frac for evaluation. Default 0.0 = fair eval "
            "from home (no carry-phase bootstrap). Set to 1.0 to force every episode to "
            "start already holding the cube lifted above the pickup (pose/skill diagnostic)."
        ),
    )
    parser.add_argument(
        "--trace-csv",
        default=None,
        help=(
            "Per-step diagnostic trace of env 0, including clean/executed actions, "
            "noise, Cartesian setpoint residual, task phase, object motion and reward "
            "components. Use with -B 1 --episodes 1 for one complete trajectory."
        ),
    )
    parser.add_argument(
        "--hard-event-csv",
        help=("Event-level CSV for every hard landing. By default a sidecar is written next to --report-csv."),
    )
    args = parser.parse_args()
    if args.strict_success:
        if args.acceptance_profile not in (None, "contact_v1"):
            parser.error("--strict-success conflicts with the selected acceptance profile")
        args.acceptance_profile = "contact_v1"
    if args.require_target is not None and args.acceptance_profile is None:
        args.acceptance_profile = "contact_v1"
    if args.action_noise_std < 0.0:
        raise ValueError("--action-noise-std must be non-negative")
    if not 0.0 <= args.place_phase_table_reset_frac <= 1.0:
        raise ValueError("--place-phase-table-reset-frac must be in [0, 1]")
    if args.event_frames_dir is not None and args.num_envs != 1:
        raise ValueError("--event-frames-dir requires -B 1")

    if args.hold:
        if args.episodes is None:
            args.episodes = 10
        _run_hold_preview(args)
        return

    if args.expert:
        _run_expert_eval(args)
        return

    # Find checkpoints (repeatable --checkpoint; the scene is built once and each
    # additional checkpoint is swapped in without recompiling anything).
    if args.checkpoint:
        ckpt_paths = [Path(p) for p in args.checkpoint]
    elif args.log_dir is not None:
        ckpt_paths = [find_latest_checkpoint(args.log_dir)]
    else:
        ckpt_paths = [DEFAULT_CHECKPOINT]
    multi_ckpt = len(ckpt_paths) > 1
    if multi_ckpt and args.episodes is not None and args.episodes <= 0:
        raise ValueError("Evaluating multiple checkpoints requires a finite --episodes count")
    for path in ckpt_paths:
        print(f"Loading checkpoint: {path}")
    ckpt_path = ckpt_paths[0]

    # Load configs from training
    config_path = args.log_dir / "config.yaml" if args.log_dir is not None else ckpt_path.parent / "config.yaml"
    metrics_path = config_path.parent / "metrics.csv"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}; checkpoint evaluation requires its original hashed training config"
        )
    runtime_config = load_runtime_config("xarm6", task="pick_place", config_path=args.runtime_config)
    expected_hash = None if args.allow_runtime_config_mismatch else runtime_config.sha256
    expected_runtime_env = None
    if not args.allow_runtime_config_mismatch:
        expected_runtime_env, _expected_reward, _expected_robot = build_pick_place_task_configs(
            "xarm6",
            recipe_path=args.recipe,
            runtime_config_path=args.runtime_config,
        )
    if args.allow_runtime_config_mismatch:
        print(f"WARNING: skipping runtime config hash check (current={runtime_config.sha256[:12]}…); visual/debug only")
    loaded, artifact, _manifest = validate_and_load_rsl_checkpoint(
        ckpt_path,
        config_path,
        map_location="cpu",
        expected_task="pick_place",
        expected_robot_key=runtime_config.robot.key,
        expected_runtime_config_sha256=expected_hash,
        expected_runtime_env=expected_runtime_env,
    )
    artifacts = {ckpt_path: artifact}
    manifests = {ckpt_path: _manifest}
    loaded_checkpoints = {ckpt_path: loaded}
    for extra_ckpt in ckpt_paths[1:]:
        # Additional checkpoints may come from other experiments (different recipe), so
        # skip the recipe contract check but still verify hashes, task, robot and the
        # runtime config; each is validated against its own log dir's config.yaml.
        extra_config = extra_ckpt.parent / "config.yaml"
        extra_loaded, extra_artifact, extra_manifest = validate_and_load_rsl_checkpoint(
            extra_ckpt,
            extra_config,
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=expected_hash,
        )
        artifacts[extra_ckpt] = extra_artifact
        manifests[extra_ckpt] = extra_manifest
        loaded_checkpoints[extra_ckpt] = extra_loaded
    if multi_ckpt:
        reference_signature = _evaluation_compatibility_signature(args, artifact)
        incompatible = [
            str(path)
            for path in ckpt_paths[1:]
            if _evaluation_compatibility_signature(args, artifacts[path]) != reference_signature
        ]
        if incompatible:
            raise ValueError(
                "checkpoints with different policy or evaluation contracts cannot share "
                "one scene; evaluate these checkpoints in separate processes: " + ", ".join(incompatible)
            )
    _stamp("Checkpoint artifacts validated")
    # Remount machine-local asset paths so server-trained configs evaluate here.
    env_cfg = deepcopy(artifact["env"])
    reward_cfg = deepcopy(artifact["reward"])
    robot_cfg = deepcopy(artifact["robot"])
    train_cfg = deepcopy(artifact["train"])
    robot_cfg["urdf_path"] = robot_urdf(
        runtime_config.robot.key,
        runtime_config.robot.urdf,
    )

    scenario_bank = _load_eval_scenario_bank(args, env_cfg, runtime_config)
    _resolve_episode_count(args, scenario_bank, default=10)
    if args.action_noise_std > 0.0 and args.episodes <= 0:
        raise ValueError("action-noise bank evaluation requires a finite --episodes count")

    _apply_eval_env_overrides(args, env_cfg)
    performance_mode, performance_mode_source = _resolve_eval_performance_mode(args, env_cfg)
    _stamp_eval_performance_mode(performance_mode, performance_mode_source)
    train_cfg["runner"]["max_iterations"] = 0
    train_cfg["runner"]["resume"] = False
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)

    if args.stage is not None:
        pinned_stage = int(args.stage)
        stage_source = "CLI argument"
    else:
        pinned_stage, stage_source = infer_eval_stage(metrics_path)
    pinned_stage = _pin_eval_curriculum_stage(env_cfg, pinned_stage)

    # Init Genesis with viewer. Viewer mode uses info logging so the long one-off
    # kernel compilation ("Compiling simulation kernels...") is visible instead of a
    # silent multi-second freeze; headless runs stay quiet.
    require_genesis_runtime(gs)
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning" if args.headless else "info",
        seed=train_cfg["seed"],
        performance_mode=performance_mode,
    )

    # Create environment with viewer
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=not args.headless,
    )
    _stamp("Scene built (one-off kernel compilation done)")

    # Keep later resets pinned to the same stage used by the constructor's first reset.
    env.curriculum_stage = pinned_stage
    print(f"Curriculum stage: {env.curriculum_stage} ({stage_source})")

    latest_metrics = read_latest_metrics(metrics_path)
    if latest_metrics is not None:
        grasp_rate = latest_metrics.get("grasp_success_rate")
        place_rate = latest_metrics.get("place_success_rate")
        if grasp_rate is not None and place_rate is not None:
            print(
                "Latest training metrics: "
                f"grasp_success_rate={100 * float(grasp_rate):.1f}%, "
                f"place_success_rate={100 * float(place_rate):.1f}%"
            )

    # Create the runner once; checkpoints are swapped into it without rebuilding the scene.
    runner = OnPolicyRunner(env, deepcopy(train_cfg), None, device=gs.device)
    if bool(env_cfg.get("actor_output_tanh", False)):
        if enable_actor_output_tanh is None:
            raise ImportError("actor_output_tanh requires local ufactory.training.policy helper")
        enable_actor_output_tanh(runner.alg.actor)

    print(f"Beta deterministic point estimate: {args.beta_point_estimate}")
    mode = "headless" if args.headless else "viewer"
    print(f"Running evaluation ({mode} mode)...")

    required_results: list[bool] = []
    for ckpt_index, current_ckpt in enumerate(ckpt_paths):
        if multi_ckpt:
            print(f"\n{'#' * 60}")
            print(f"# Checkpoint {ckpt_index + 1}/{len(ckpt_paths)}: {current_ckpt}")
            print(f"{'#' * 60}")
        required_results.append(
            _evaluate_checkpoint(
                args,
                env=env,
                runner=runner,
                train_cfg=train_cfg,
                artifact=artifacts[current_ckpt],
                runtime_config=runtime_config,
                scenario_bank=scenario_bank,
                ckpt_path=current_ckpt,
                checkpoint=loaded_checkpoints[current_ckpt],
                manifest=manifests[current_ckpt],
                pinned_stage=pinned_stage,
                is_first=(ckpt_index == 0),
                multi_ckpt=multi_ckpt,
            )
        )
    if args.require_target is not None and not all(required_results):
        raise SystemExit(2)


def _run_hold_preview(args: argparse.Namespace) -> None:
    """Build the current recipe scene and pin the arm at home for visual comparison."""
    if args.headless:
        raise SystemExit("--hold requires a viewer; omit --headless")

    recipe = args.recipe
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs(
        "xarm6",
        recipe_path=recipe,
        runtime_config_path=args.runtime_config,
    )
    env_cfg["num_envs"] = int(args.num_envs)
    env_cfg["fixed_demo_layout"] = True
    pinned_stage = _pin_eval_curriculum_stage(env_cfg, args.stage if args.stage is not None else 0)
    performance_mode, performance_mode_source = _resolve_eval_performance_mode(args, env_cfg)
    _stamp_eval_performance_mode(performance_mode, performance_mode_source)

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=1,
        performance_mode=performance_mode,
    )

    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )
    env.curriculum_stage = pinned_stage
    _obs, _extras = env.reset()
    base = [float(v) for v in robot_cfg["base_pos"]]
    obj_base = [float(v) for v in env.obj_pos_base()[0].detach().cpu().tolist()]
    tgt_base = [float(v) for v in env.target_pos[0].detach().cpu().tolist()]
    obj_world = [float(v) for v in env.base_to_world(env.obj_pos_base())[0].detach().cpu().tolist()]
    tgt_world = [float(v) for v in env.base_to_world(env.target_pos)[0].detach().cpu().tolist()]
    print(
        "HOLD preview (pin home EE, gripper open). "
        f"fixed_demo_layout={env.fixed_demo_layout}, substeps={env.substeps}, "
        f"stage={env.curriculum_stage}."
    )
    print(f"  base_pos_world = {base}")
    print(f"  obj_base       = {obj_base}")
    print(f"  target_base    = {tgt_base}")
    print(f"  obj_world      = {obj_world}")
    print(f"  target_world   = {tgt_world}")
    print(f"  default_ee_pos = {list(env_cfg['default_ee_pos'])}")
    print("Close the viewer or Ctrl+C to exit.")
    steps = 0
    # episodes=0 → hold indefinitely; otherwise step until timeout episodes elapse.
    max_steps = None if args.episodes <= 0 else int(args.episodes) * int(env.max_episode_length)
    try:
        while max_steps is None or steps < max_steps:
            env.hold_home_step()
            steps += 1
    except KeyboardInterrupt:
        print(f"\nHOLD preview stopped after {steps} steps.")


if __name__ == "__main__":
    main()
