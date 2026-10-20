"""Train the xArm6 + Gripper G2 tabletop pick-and-place task with pure PPO."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from typing import Any

from .runtime import (
    DEFAULT_RECIPE,
    OUTPUT_ROOT,
    configure_runtime,
    require_backend,
    require_checkpoint,
    require_rsl_runtime,
)

configure_runtime()
require_rsl_runtime()

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import torch

from ufactory.robots.paths import robot_urdf
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    PICK_PLACE_RL_ROBOTS,
    build_pick_place_task_configs,
    build_train_config,
    load_pick_place_rl_runtime_config,
    load_training_config,
    load_training_recipe,
    validate_and_load_rsl_checkpoint,
    write_artifact_inventory,
    write_checkpoint_manifest,
    write_run_provenance,
    write_training_config,
)
from ufactory.training.pick_place_config import PICK_PLACE_ACTION_CONTRACT

from .env import XArm6PickPlaceEnv
from .evaluate import acceptance_passed, checkpoint_score, evaluate, save_results

EXECUTOR_ACTION_CONTRACT = PICK_PLACE_ACTION_CONTRACT


def _install_fixed_learning_rate_guard(runner: Any, train_cfg: dict[str, Any]) -> None:
    """Reject accidental learning-rate schedule changes when schedule == 'fixed'."""
    alg_cfg = train_cfg.get("algorithm", {})
    if str(alg_cfg.get("schedule", "")).lower() != "fixed":
        return
    expected_lr = float(alg_cfg["learning_rate"])
    original_update = runner.alg.update

    def guarded_update(*args: Any, **kwargs: Any) -> Any:
        result = original_update(*args, **kwargs)
        actual_lr = float(runner.alg.learning_rate)
        group_lrs = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
        if not math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12) or any(
            not math.isclose(lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12) for lr in group_lrs
        ):
            raise RuntimeError(
                f"fixed learning rate changed after update: expected {expected_lr:g}, got alg={actual_lr:g}, optimizer={group_lrs}"
            )
        return result

    runner.alg.update = guarded_update


class ArtifactOnPolicyRunner(OnPolicyRunner):
    """Write the integrity manifest immediately after every RSL-RL checkpoint."""

    checkpoint_training_config: dict[str, Any] | None = None
    checkpoint_env: XArm6PickPlaceEnv | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        original_log = self.logger.log

        def log_with_metrics(**metrics: Any) -> None:
            episodes = self.logger.ep_extras
            keys = {key for episode in episodes for key in episode}
            values = {
                key: torch.cat(
                    [torch.as_tensor(episode[key]).flatten().cpu() for episode in episodes if key in episode]
                )
                for key in keys
            }
            report = {
                "iteration": metrics["it"],
                "completed_episodes": values["success_rate"].numel() if "success_rate" in values else 0,
                "collection_s": metrics["collect_time"],
                "learning_s": metrics["learn_time"],
                "learning_rate": metrics["learning_rate"],
                "action_std": metrics["action_std"].mean().item(),
                "loss": metrics["loss_dict"],
                "episode": {key: value.float().mean().item() for key, value in values.items()},
            }
            original_log(**metrics)
            with (Path(self.logger.log_dir) / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")

        self.logger.log = log_with_metrics

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
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


def _provenance_sources(repo_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        Path(__file__).with_name("env.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("contacts.py"),
        Path(__file__).with_name("runtime.py"),
        Path(__file__).with_name("evaluate.py"),
        Path(__file__).with_name("recipe.yaml"),
        repo_root / "ufactory/simulation/__init__.py",
        repo_root / "ufactory/simulation/compat.py",
        repo_root / "ufactory/simulation/g2_adapter.py",
        repo_root / "ufactory/simulation/g2_controller.py",
        repo_root / "ufactory/simulation/g2_model.py",
        repo_root / "ufactory/simulation/physics.py",
        repo_root / "ufactory/training/logic/__init__.py",
        repo_root / "ufactory/training/artifacts.py",
        repo_root / "ufactory/training/models.py",
        repo_root / "ufactory/training/tasks.py",
        repo_root / "ufactory/training/pick_place_config.py",
        repo_root / "ufactory/training/transfer.py",
    ]


def _prepare_log_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"training directory already exists: {path}; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="xarm6", choices=PICK_PLACE_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-e", "--exp-name", default="pick_place_e2e_s1")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-B", "--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument(
        "--early-stop-evaluations", type=int, help="Consecutive passing evaluations before stopping; 0 disables"
    )
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.num_envs is not None and args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be positive")
    if args.early_stop_evaluations is not None and args.early_stop_evaluations < 0:
        parser.error("--early-stop-evaluations must be non-negative")

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")

    recipe = load_training_recipe(args.recipe)
    early_stop_evaluations = int(
        args.early_stop_evaluations
        if args.early_stop_evaluations is not None
        else recipe["train"]["runner"].get("early_stop_evaluations", 2)
    )
    num_envs = int(args.num_envs if args.num_envs is not None else recipe["environment"]["num_envs"])
    max_iterations = int(
        args.max_iterations if args.max_iterations is not None else recipe["train"]["runner"]["max_iterations"]
    )
    runtime_config = load_pick_place_rl_runtime_config(args.robot, config_path=args.runtime_config)
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs(
        args.robot,
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )
    env_cfg["num_envs"] = num_envs
    performance_mode = bool(recipe["environment"].get("genesis_performance_mode", False))
    env_cfg["genesis_performance_mode"] = performance_mode
    train_cfg = build_train_config(
        args.recipe,
        experiment_name=args.exp_name,
        max_iterations=max_iterations,
    )
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    train_cfg["runner"]["early_stop_evaluations"] = early_stop_evaluations
    train_cfg["runner"]["resume"] = args.resume is not None
    train_cfg["runner"]["resume_path"] = None if args.resume is None else str(args.resume)

    resume_state = None
    if args.resume is not None:
        args.resume = require_checkpoint(args.resume)
        resume_state, resume_config, _resume_manifest = validate_and_load_rsl_checkpoint(
            args.resume,
            args.resume.parent / "config.yaml",
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_runtime_env=env_cfg,
            full_resume=True,
        )
        if int(resume_config["env"]["num_envs"]) != num_envs:
            parser.error("resume requires the original --num-envs; episodes restart from home")

    log_dir = args.log_dir or (OUTPUT_ROOT / args.exp_name)
    _prepare_log_dir(log_dir, overwrite=args.overwrite)
    portable_robot_cfg = deepcopy(robot_cfg)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        portable_robot_cfg["urdf_path"] = Path(robot_cfg["urdf_path"]).resolve().relative_to(repo_root).as_posix()
    except ValueError:
        portable_robot_cfg["urdf_path"] = str(robot_cfg["urdf_path"])

    write_training_config(
        log_dir / "config.yaml",
        task="pick_place",
        robot_key=runtime_config.robot.key,
        env=env_cfg,
        reward=reward_cfg,
        robot=portable_robot_cfg,
        train=train_cfg,
    )
    training_config = load_training_config(log_dir / "config.yaml")
    write_run_provenance(
        log_dir / "provenance.json",
        training_config=training_config,
        source_paths=_provenance_sources(repo_root),
    )

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.cpu if args.backend == "cpu" else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(train_cfg["seed"]),
        performance_mode=performance_mode,
    )
    require_backend(gs, args.backend)
    robot_cfg["urdf_path"] = robot_urdf(runtime_config.robot.key, runtime_config.robot.urdf)
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    runner_cfg = deepcopy(train_cfg)
    runner_cfg["runner"]["resume"] = False
    runner_cfg["runner"]["resume_path"] = None
    runner = ArtifactOnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device=gs.device)
    runner.checkpoint_training_config = training_config
    runner.checkpoint_env = env
    _install_fixed_learning_rate_guard(runner, runner_cfg)

    if resume_state is not None:
        runner.alg.load(resume_state, None, strict=True)
        runner.current_learning_iteration = int(resume_state.get("iter", 0)) + 1
        infos = resume_state.get("infos") or {}
        if "ufactory_env_state" in infos:
            env.load_training_state_dict(infos["ufactory_env_state"])

    best_score = (-1.0, -1.0, -1.0, float("-inf"), float("-inf"))

    def validate(tag: str) -> dict[str, Any]:
        nonlocal best_score
        actor = runner.get_inference_policy(device=gs.device)

        def _policy(obs: Any) -> torch.Tensor:
            return actor(obs, stochastic_output=False)

        summary, rows = evaluate(env, _policy, episodes=args.eval_episodes, seed=17000)
        summary["passed"] = acceptance_passed(summary)
        save_results(log_dir / f"evaluation_{tag}.json", summary, rows)
        print(f"STRICT EVAL {tag}: {json.dumps(summary)}", flush=True)
        score = checkpoint_score(summary)
        if score > best_score:
            best_score = score
            runner.save(str(log_dir / "model_best.pt"), infos={"validation": summary, "stage": tag})
        env.fixed_spawns = None
        env.reset()
        if hasattr(runner.logger, "cur_reward_sum") and runner.logger.cur_reward_sum is not None:
            runner.logger.cur_reward_sum.zero_()
            runner.logger.cur_episode_length.zero_()
        return summary

    validate("initial")
    remaining = max_iterations
    completed = 0
    passing_evaluations = 0
    while remaining > 0:
        count = min(args.eval_interval, remaining)
        runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
        remaining -= count
        completed += count
        summary = validate(f"ppo_{completed}")
        passing_evaluations = passing_evaluations + 1 if summary["passed"] else 0
        if early_stop_evaluations and passing_evaluations >= early_stop_evaluations:
            break
        if remaining > 0:
            runner.current_learning_iteration += 1

    final_ckpt = log_dir / "model_final.pt"
    runner.save(str(final_ckpt), infos={"stage": "final"})
    checkpoints = sorted(log_dir.glob("model_*.pt"))
    selected = log_dir / "model_best.pt" if (log_dir / "model_best.pt").is_file() else final_ckpt
    write_artifact_inventory(
        log_dir / "artifacts.yaml",
        training_config=training_config,
        checkpoints=checkpoints,
        selected_checkpoint=selected,
    )
    (log_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "completed_iterations": completed,
                "max_iterations": max_iterations,
                "early_stopped": remaining > 0,
                "passing_evaluations": passing_evaluations,
                "best_success_rate": best_score[0],
                "best_press_mm": -best_score[3],
                "selected_checkpoint": selected.name,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    gs.destroy()
    print(f"Completed {completed} iterations: best strict success_rate={best_score[0]:.3f}; {selected}", flush=True)


if __name__ == "__main__":
    main()
