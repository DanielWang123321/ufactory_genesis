"""Train the xArm6 + Gripper G2 pick-place task with PPO."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import math
from pathlib import Path
import shutil

from packaging.version import Version

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if Version(metadata.version("rsl-rl-lib")) < Version("5.3.0"):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as exc:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib>=5.3.0'.") from exc

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from ufactory.config import load_runtime_config
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    load_training_config,
    load_training_recipe,
    build_pick_place_task_configs,
    build_train_config,
    PICK_PLACE_RL_ROBOTS,
    write_checkpoint_manifest,
    write_artifact_inventory,
    write_run_provenance,
    write_training_config,
    validate_and_load_rsl_checkpoint,
)
from ufactory.training.transfer import (
    constrain_actor_to_appended_observation_columns,
    constrain_actor_to_layout_residual,
    freeze_actor,
    initialize_guided_pick_place_actor,
    initialize_layout_residual_actor,
    project_actor_observation_expansion,
)

try:
    from ufactory.training.policy import enable_actor_output_tanh
except ImportError:
    enable_actor_output_tanh = None  # type: ignore[assignment]

from .env import XArm6PickPlaceEnv

DEFAULT_RECIPE = Path(__file__).with_name("recipe.yaml")
EXECUTOR_ACTION_CONTRACT = "normalized_cartesian_delta_xyz+normalized_gripper_gap_delta"
LAYOUT_RESIDUAL_PROJECTION_TYPES = {
    "layout_residual_actor_v1",
    "layout_phase_residual_actor_v2",
    "layout_phase_residual_actor_v3",
}
GUIDED_PROJECTION_TYPE = "scripted_action_hint_actor_v1"


class ArtifactOnPolicyRunner(OnPolicyRunner):
    """Write the integrity manifest immediately after every RSL checkpoint."""

    checkpoint_training_config = None
    checkpoint_env = None

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_infos = {} if infos is None else dict(infos)
        if self.checkpoint_env is not None:
            saved_infos["ufactory_env_state"] = self.checkpoint_env.training_state_dict()
        super().save(path, saved_infos)
        if self.checkpoint_training_config is not None:
            write_checkpoint_manifest(
                Path(path),
                training_config=self.checkpoint_training_config,
                executor_action_contract=EXECUTOR_ACTION_CONTRACT,
            )


def _prepare_log_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"training directory already exists: {path}; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _build_observation_projection(
    env_cfg: dict,
    train_cfg: dict,
    *,
    source_observation_dim: int,
    target_observation_dim: int,
) -> dict:
    """Resolve a versioned observation transfer recorded in run artifacts."""

    projection_type = str(
        train_cfg["runner"].get(
            "observation_projection_initializer",
            "zero_append_normalized_layout_offsets",
        )
    )
    projection = {
        "type": projection_type,
        "source_policy_observation_dim": int(source_observation_dim),
        "target_policy_observation_dim": int(target_observation_dim),
        "appended_observation_dim": int(target_observation_dim - source_observation_dim),
    }
    if projection_type == "zero_append_normalized_layout_offsets":
        return projection
    if projection_type in LAYOUT_RESIDUAL_PROJECTION_TYPES:
        return projection
    if projection_type == GUIDED_PROJECTION_TYPE:
        return projection
    if projection_type != "canonical_pick_place_layout_offsets_v1":
        raise ValueError(f"unsupported observation projection initializer: {projection_type!r}")

    def half_span(lower_key: str, upper_key: str) -> list[float]:
        lower = env_cfg.get(lower_key)
        upper = env_cfg.get(upper_key)
        if not isinstance(lower, (list, tuple)) or not isinstance(upper, (list, tuple)):
            raise ValueError(f"canonical layout projection requires {lower_key} and {upper_key}")
        if len(lower) < 2 or len(upper) < 2:
            raise ValueError(f"canonical layout projection requires xy bounds in {lower_key} and {upper_key}")
        result = [abs(float(upper[index]) - float(lower[index])) * 0.5 for index in range(2)]
        if any(value <= 0.0 or not math.isfinite(value) for value in result):
            raise ValueError(f"canonical layout projection requires positive finite xy spans for {lower_key}")
        return result

    projection["object_half_span_xy_m"] = half_span("obj_spawn_lower", "obj_spawn_upper")
    projection["target_half_span_xy_m"] = half_span("target_spawn_lower", "target_spawn_upper")
    return projection


def _install_fixed_learning_rate_guard(runner: OnPolicyRunner, train_cfg: dict) -> None:
    """Fail immediately if a nominally fixed-rate run changes LR during an update."""

    algorithm_cfg = train_cfg["algorithm"]
    if algorithm_cfg.get("schedule") != "fixed":
        return
    expected = float(algorithm_cfg["learning_rate"])
    original_update = runner.alg.update

    def assert_rate(phase: str) -> None:
        algorithm_rate = float(runner.alg.learning_rate)
        optimizer_rates = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
        if not math.isclose(algorithm_rate, expected, rel_tol=0.0, abs_tol=1e-12) or any(
            not math.isclose(rate, expected, rel_tol=0.0, abs_tol=1e-12) for rate in optimizer_rates
        ):
            raise RuntimeError(
                f"fixed learning rate changed {phase}: expected={expected:g}, "
                f"algorithm={algorithm_rate:g}, optimizer={optimizer_rates}"
            )

    def guarded_update(*args, **kwargs):
        assert_rate("before PPO update")
        result = original_update(*args, **kwargs)
        assert_rate("after PPO update")
        return result

    assert_rate("at runner initialization")
    runner.alg.update = guarded_update
    print(f"Fixed learning-rate guard active: {expected:g}")


def _install_observation_projection_training_guard(
    runner: OnPolicyRunner,
    train_cfg: dict,
    transfer_projection: dict | None,
    *,
    projection_was_applied: bool,
) -> None:
    """Protect inherited policy parameters under a declared transfer scope."""

    scope = str(train_cfg["runner"].get("observation_projection_train_scope", "full_actor"))
    if scope == "full_actor":
        return
    if scope not in {"appended_columns_only", "layout_residual_only", "frozen_guided_actor"}:
        raise ValueError(f"unsupported observation projection train scope: {scope!r}")
    if transfer_projection is None:
        raise ValueError(f"{scope} requires an actor observation projection")
    if scope == "layout_residual_only" and transfer_projection.get("type") not in LAYOUT_RESIDUAL_PROJECTION_TYPES:
        raise ValueError("layout_residual_only requires a supported layout-residual projection")
    if scope == "appended_columns_only" and transfer_projection.get("type") in LAYOUT_RESIDUAL_PROJECTION_TYPES:
        raise ValueError("layout residual projection requires layout_residual_only training scope")
    if scope == "frozen_guided_actor" and transfer_projection.get("type") != GUIDED_PROJECTION_TYPE:
        raise ValueError("frozen_guided_actor requires the scripted-action-hint projection")
    if scope != "frozen_guided_actor" and transfer_projection.get("type") == GUIDED_PROJECTION_TYPE:
        raise ValueError("scripted-action-hint projection requires frozen_guided_actor training scope")
    if scope == "frozen_guided_actor":
        guard = freeze_actor(runner.alg.actor)
        scope_label = "guided-actor"
    elif transfer_projection.get("type") in LAYOUT_RESIDUAL_PROJECTION_TYPES:
        guard = constrain_actor_to_layout_residual(runner.alg.actor)
        scope_label = "layout-residual"
    else:
        guard = constrain_actor_to_appended_observation_columns(
            runner.alg.actor,
            source_observation_dim=int(transfer_projection["source_policy_observation_dim"]),
            require_zero_appended=(
                projection_was_applied and transfer_projection.get("type") == "zero_append_normalized_layout_offsets"
            ),
        )
        scope_label = "appended-column"
    original_update = runner.alg.update

    def guarded_update(*args, **kwargs):
        guard.assert_preserved()
        result = original_update(*args, **kwargs)
        guard.assert_preserved()
        return result

    runner.alg.update = guarded_update
    print(f"Inherited actor guard active: only {guard.trainable_parameter_count} {scope_label} parameters may change")


def main() -> None:
    parser = argparse.ArgumentParser(description="xArm6 Gripper G2 pick-place RL training")
    parser.add_argument("--robot", default="xarm6", choices=PICK_PLACE_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-e", "--exp-name", default="xarm6-pick-place-cart-xyz-g2")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-B", "--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int, help="Override the training seed from the recipe")
    parser.add_argument("--gamma", type=float, help="Override PPO gamma for a controlled comparison")
    parser.add_argument("--lam", type=float, help="Override GAE lambda for a controlled comparison")
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the recipe learning rate for a controlled fine-tuning phase",
    )
    parser.add_argument(
        "--curriculum-edge-fraction",
        type=float,
        help="Override the fraction of training layouts sampled from stage boundary corners",
    )
    parser.add_argument(
        "--curriculum-initial-stage",
        type=int,
        choices=range(5),
        help="Override the recipe curriculum start stage for an independently gated phase",
    )
    parser.add_argument(
        "--curriculum-max-stage",
        type=int,
        choices=range(5),
        help="Cap curriculum expansion for an independently gated phase",
    )
    parser.add_argument("--place-phase-reset-frac", type=float, help="Override place-phase reset fraction")
    parser.add_argument("--carry-phase-reset-frac", type=float, help="Override carry-phase reset fraction")
    parser.add_argument("--grasp-phase-reset-frac", type=float, help="Override grasp-phase reset fraction")
    transfer = parser.add_mutually_exclusive_group()
    transfer.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume actor, critic, optimizer, noise, and iteration from an interrupted run",
    )
    transfer.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Transfer actor weights only; reset critic, optimizer, noise, and iteration",
    )
    transfer.add_argument(
        "--actor-critic-warm-start",
        type=Path,
        default=None,
        help=("Transfer architecture-compatible actor and critic weights; reset optimizer, noise, and iteration"),
    )
    parser.add_argument(
        "--actor-obs-normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override actor observation normalization from the published recipe",
    )
    parser.add_argument(
        "--performance-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Genesis static-array performance mode from the recipe",
    )
    parser.add_argument(
        "--randomize-initial-episode-length",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override initial episode-length randomization from the recipe",
    )
    parser.add_argument(
        "--actor-output-tanh",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override bounded actor-mean output from the recipe",
    )
    args = parser.parse_args()
    recipe = load_training_recipe(args.recipe)
    num_envs = int(args.num_envs if args.num_envs is not None else recipe["environment"]["num_envs"])
    max_iterations = int(
        args.max_iterations if args.max_iterations is not None else recipe["train"]["runner"]["max_iterations"]
    )
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs(
        args.robot,
        runtime_config_path=args.runtime_config,
        recipe_path=args.recipe,
    )
    env_cfg["num_envs"] = num_envs
    if args.curriculum_initial_stage is not None:
        env_cfg["curriculum_initial_stage"] = int(args.curriculum_initial_stage)
    if args.curriculum_max_stage is not None:
        env_cfg["curriculum_max_stage"] = int(args.curriculum_max_stage)
    if env_cfg["curriculum_initial_stage"] > env_cfg["curriculum_max_stage"]:
        parser.error("--curriculum-initial-stage must be <= --curriculum-max-stage")
    if args.curriculum_edge_fraction is not None:
        if not 0.0 <= args.curriculum_edge_fraction <= 1.0:
            parser.error("--curriculum-edge-fraction must be in [0, 1]")
        env_cfg["curriculum_edge_fraction"] = float(args.curriculum_edge_fraction)
    reset_overrides = {
        "place_phase_reset_frac": args.place_phase_reset_frac,
        "carry_phase_reset_frac": args.carry_phase_reset_frac,
        "grasp_phase_reset_frac": args.grasp_phase_reset_frac,
    }
    for key, value in reset_overrides.items():
        if value is not None:
            env_cfg[key] = float(value)
    reset_total = sum(float(env_cfg.get(key, 0.0)) for key in reset_overrides)
    if any(float(env_cfg.get(key, 0.0)) < 0.0 for key in reset_overrides) or reset_total > 1.0 + 1e-9:
        parser.error("phase reset fractions must be non-negative and sum to at most 1")
    performance_mode = (
        bool(args.performance_mode)
        if args.performance_mode is not None
        else bool(recipe["environment"].get("genesis_performance_mode", False))
    )
    randomize_initial_episode_length = (
        bool(args.randomize_initial_episode_length)
        if args.randomize_initial_episode_length is not None
        else bool(recipe["environment"].get("randomize_initial_episode_length", True))
    )
    actor_output_tanh = (
        bool(args.actor_output_tanh)
        if args.actor_output_tanh is not None
        else bool(recipe["environment"].get("actor_output_tanh", False))
    )
    env_cfg["genesis_performance_mode"] = performance_mode
    env_cfg["randomize_initial_episode_length"] = randomize_initial_episode_length
    env_cfg["actor_output_tanh"] = actor_output_tanh
    train_cfg = build_train_config(
        args.recipe,
        experiment_name=args.exp_name,
        max_iterations=max_iterations,
    )
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    if args.gamma is not None:
        train_cfg["algorithm"]["gamma"] = float(args.gamma)
    if args.lam is not None:
        train_cfg["algorithm"]["lam"] = float(args.lam)
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            parser.error("--learning-rate must be positive")
        train_cfg["algorithm"]["learning_rate"] = float(args.learning_rate)
    if args.actor_obs_normalization is not None:
        train_cfg["actor"]["obs_normalization"] = bool(args.actor_obs_normalization)
    env_cfg["training_steps_per_iteration"] = int(train_cfg["num_steps_per_env"])
    distribution_cfg = train_cfg["actor"].get("distribution_cfg") or {}
    distribution_name = str(distribution_cfg.get("class_name", ""))
    if distribution_name.endswith("BetaDistribution"):
        if actor_output_tanh:
            raise ValueError("BetaDistribution must not be combined with actor_output_tanh")
        action_range = tuple(float(value) for value in distribution_cfg.get("action_range", (-1.0, 1.0)))
        if action_range != (-env_cfg["action_clip"], env_cfg["action_clip"]):
            raise ValueError(
                f"Beta action_range={action_range} must match environment "
                f"[-action_clip, action_clip]=({-env_cfg['action_clip']}, {env_cfg['action_clip']})"
            )
        if not bool(env_cfg.get("strict_action_bounds", False)):
            raise ValueError("BetaDistribution runs must enable environment.strict_action_bounds")
    runtime_config = load_runtime_config(
        args.robot,
        task="pick_place",
        config_path=args.runtime_config,
    )
    transfer_path = args.resume or args.warm_start or args.actor_critic_warm_start
    transfer_state = None
    transfer_projection = None
    projection_was_applied = False
    projection_train_scope = str(train_cfg["runner"].get("observation_projection_train_scope", "full_actor"))
    if projection_train_scope not in {
        "full_actor",
        "appended_columns_only",
        "layout_residual_only",
        "frozen_guided_actor",
    }:
        raise ValueError(f"unsupported observation projection train scope: {projection_train_scope!r}")
    if transfer_path is None:
        train_cfg["runner"]["transfer_mode"] = "fresh"
    else:
        transfer_path = Path(transfer_path)
        if not transfer_path.is_file():
            raise FileNotFoundError(f"transfer checkpoint not found: {transfer_path}")
        with transfer_path.open("rb") as transfer_file:
            transfer_sha256 = hashlib.file_digest(transfer_file, "sha256").hexdigest()
        if args.resume is not None:
            transfer_mode = "full_resume"
        elif args.actor_critic_warm_start is not None:
            transfer_mode = "actor_critic"
        else:
            transfer_mode = "actor_only"
        train_cfg["runner"].update(
            {
                "transfer_mode": transfer_mode,
                "transfer_checkpoint": str(transfer_path),
                "transfer_checkpoint_sha256": transfer_sha256,
            }
        )
        transfer_state, transfer_config, transfer_manifest = validate_and_load_rsl_checkpoint(
            transfer_path,
            transfer_path.parent / "config.yaml",
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_runtime_env=env_cfg,
            allowed_runtime_env_mismatches=(() if args.resume is not None else ("fixed_demo_layout",)),
        )
        source_obs = int(transfer_manifest.observation_dim)
        target_obs = int(env_cfg["num_obs"])
        if source_obs != target_obs:
            if args.warm_start is None:
                raise ValueError("observation expansion supports actor-only --warm-start")
            projection_initializer = str(
                train_cfg["runner"].get(
                    "observation_projection_initializer",
                    "zero_append_normalized_layout_offsets",
                )
            )
            guided_projection = projection_initializer == GUIDED_PROJECTION_TYPE
            expected_append = 10 if guided_projection else 6
            if (
                not bool(env_cfg.get("include_normalized_layout_offsets", False))
                or target_obs - source_obs != expected_append
            ):
                suffix = "six layout plus four guide-action values" if guided_projection else "six layout values"
                raise ValueError(f"actor observation expansion requires exactly {suffix}")
            if guided_projection and not bool(env_cfg.get("include_scripted_action_hint", False)):
                raise ValueError("scripted-action-hint projection requires include_scripted_action_hint")
            if bool(train_cfg["actor"].get("obs_normalization", False)):
                raise ValueError("actor observation expansion requires actor observation normalization disabled")
            transfer_projection = _build_observation_projection(
                env_cfg,
                train_cfg,
                source_observation_dim=source_obs,
                target_observation_dim=target_obs,
            )
            train_cfg["runner"]["observation_projection"] = transfer_projection
            projection_was_applied = True
        elif projection_train_scope != "full_actor":
            saved_projection = transfer_config.get("train", {}).get("runner", {}).get("observation_projection")
            source_width = 10 if projection_train_scope == "frozen_guided_actor" else 6
            expected_projection = _build_observation_projection(
                env_cfg,
                train_cfg,
                source_observation_dim=target_obs - source_width,
                target_observation_dim=target_obs,
            )
            if saved_projection != expected_projection:
                raise ValueError("guarded same-dimension transfer requires its saved observation projection")
            transfer_projection = dict(expected_projection)
            train_cfg["runner"]["observation_projection"] = transfer_projection
    if projection_train_scope != "full_actor" and transfer_projection is None:
        raise ValueError(f"{projection_train_scope} requires an expanded actor warm start")
    log_dir = args.log_dir or Path("outputs") / "rl" / "pick_place" / args.exp_name
    _prepare_log_dir(log_dir, overwrite=args.overwrite)

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
        args.recipe,
        repo_root / "ufactory/training/logic/__init__.py",
        repo_root / "ufactory/training/logic/pick_place.py",
        repo_root / "ufactory/training/tasks.py",
        repo_root / "ufactory/training/artifacts.py",
        repo_root / "ufactory/training/models.py",
        repo_root / "ufactory/training/transfer.py",
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
        performance_mode=performance_mode,
    )
    env = XArm6PickPlaceEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=args.vis)
    env.csv_log_path = str(log_dir / "metrics.csv")
    env.write_metrics_snapshot()
    runner = ArtifactOnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.checkpoint_training_config = artifact
    runner.checkpoint_env = env
    if actor_output_tanh:
        if enable_actor_output_tanh is None:
            raise ImportError("actor_output_tanh requires local ufactory.training.policy helper")
        enable_actor_output_tanh(runner.alg.actor)
    if transfer_state is not None:
        if args.resume is not None:
            print(f"Resuming full training state from {args.resume}")
            load_iteration = runner.alg.load(transfer_state, None, strict=True)
            if load_iteration:
                runner.current_learning_iteration = int(transfer_state["iter"])
            resume_infos = transfer_state.get("infos")
            if not isinstance(resume_infos, dict) or "ufactory_env_state" not in resume_infos:
                raise ValueError(
                    "checkpoint predates faithful environment-state resume; use "
                    "--actor-critic-warm-start to create an explicit new branch"
                )
            env.load_training_state_dict(resume_infos["ufactory_env_state"])
            loaded_iteration = int(runner.current_learning_iteration)
            runner.current_learning_iteration = loaded_iteration + 1
            resolved_lr = float(train_cfg["algorithm"]["learning_rate"])
            runner.alg.learning_rate = resolved_lr
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = resolved_lr
            print(f"Applied resolved full-resume learning rate: {resolved_lr:g}")
            print(
                "Restored environment training progress: "
                f"source_iteration={loaded_iteration}, next_iteration="
                f"{runner.current_learning_iteration}, batch_step={env.total_env_steps}, "
                f"curriculum_stage={env.curriculum_stage}, "
                f"noise_std={env._current_action_noise_std():g}"
            )
        else:
            if projection_was_applied:
                if transfer_projection["type"] == GUIDED_PROJECTION_TYPE:
                    resolved_projection = initialize_guided_pick_place_actor(
                        runner.alg.actor,
                        transfer_state,
                        device=gs.device,
                    )
                elif transfer_projection["type"] in LAYOUT_RESIDUAL_PROJECTION_TYPES:
                    resolved_projection = initialize_layout_residual_actor(
                        runner.alg.actor,
                        transfer_state,
                        device=gs.device,
                    )
                else:
                    resolved_projection = project_actor_observation_expansion(
                        runner.alg.actor,
                        transfer_state,
                        device=gs.device,
                        appended_initializer=transfer_projection,
                    )
                expected_projection = {key: transfer_projection[key] for key in resolved_projection}
                if resolved_projection != expected_projection:
                    raise RuntimeError("resolved actor observation projection differs from training config")
                projection_label = transfer_projection["type"]
                print(
                    f"Initialized appended actor observation columns ({projection_label}): "
                    f"{resolved_projection['source_policy_observation_dim']} -> "
                    f"{resolved_projection['target_policy_observation_dim']}"
                )
            else:
                load_cfg = {
                    "actor": True,
                    "critic": args.actor_critic_warm_start is not None,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                }
                runner.alg.load(transfer_state, load_cfg, strict=True)
            source_iter = int(transfer_state.get("iter", -1))
            scope = "actor and critic" if args.actor_critic_warm_start is not None else "actor only"
            print(
                f"Warm-started {scope} from {transfer_path} "
                f"(source iteration {source_iter}); optimizer and iteration reset"
            )
    _install_observation_projection_training_guard(
        runner,
        train_cfg,
        transfer_projection,
        projection_was_applied=projection_was_applied,
    )
    _install_fixed_learning_rate_guard(runner, train_cfg)
    runner.learn(
        num_learning_iterations=max_iterations,
        init_at_random_ep_len=randomize_initial_episode_length,
    )

    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    for checkpoint in checkpoints:
        write_checkpoint_manifest(
            checkpoint,
            training_config=artifact,
            executor_action_contract=EXECUTOR_ACTION_CONTRACT,
        )
    if checkpoints:
        write_checkpoint_manifest(
            checkpoints[-1],
            training_config=artifact,
            executor_action_contract=EXECUTOR_ACTION_CONTRACT,
            output_path=log_dir / "checkpoint_manifest.json",
        )
    write_artifact_inventory(
        log_dir / "artifacts.yaml",
        training_config=artifact,
        checkpoints=checkpoints,
    )


if __name__ == "__main__":
    main()
