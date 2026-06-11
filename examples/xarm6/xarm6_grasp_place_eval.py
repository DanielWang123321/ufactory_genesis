"""
xArm 6 Grasp-Place - Evaluation / Visualization Script.

Load a trained checkpoint and visualize the policy in the Genesis viewer.

Usage:
    source ~/envs/py312/bin/activate

    # Visualize latest checkpoint
    python examples/xarm6/xarm6_grasp_place_eval.py

    # Visualize specific checkpoint
    python examples/xarm6/xarm6_grasp_place_eval.py --checkpoint logs/xarm6-grasp-place/model_2500.pt

    # Multiple envs side by side
    python examples/xarm6/xarm6_grasp_place_eval.py -B 4

    # Headless evaluation when no display is available
    python examples/xarm6/xarm6_grasp_place_eval.py --headless --episodes 1
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import torch

try:
    from importlib import metadata

    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xarm6_grasp_place_env import XArm6GraspPlaceEnv


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


def load_runner_checkpoint(runner: OnPolicyRunner, ckpt_path: Path, load_optimizer: bool = False) -> dict:
    """Load an rsl-rl checkpoint while respecting the current runtime device."""
    map_location = runner.device
    if not isinstance(map_location, torch.device):
        map_location = torch.device(map_location)

    loaded_dict = torch.load(ckpt_path, weights_only=False, map_location=map_location)
    runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    if runner.alg.rnd:
        runner.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
    if load_optimizer:
        runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if runner.alg.rnd:
            runner.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    runner.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict.get("infos", {})


def main():
    parser = argparse.ArgumentParser(description="xArm 6 Grasp-Place Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). Default: latest in logs/xarm6-grasp-place/",
    )
    parser.add_argument(
        "-B", "--num_envs", type=int, default=1,
        help="Number of parallel environments to visualize",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to run (0 = infinite)",
    )
    parser.add_argument(
        "-e", "--exp_name", type=str, default="xarm6-grasp-place",
        help="Experiment name (for finding log dir)",
    )
    parser.add_argument(
        "--stage", type=int, default=None,
        help="Curriculum stage for eval (default: infer from latest metrics.csv row)",
    )
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="Run evaluation without opening the Genesis viewer",
    )
    args = parser.parse_args()

    # Find checkpoint
    log_dir = Path("logs") / args.exp_name
    metrics_path = log_dir / "metrics.csv"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint(log_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    # Load configs from training
    cfgs_path = log_dir / "cfgs.pkl"
    if cfgs_path.exists():
        with open(cfgs_path, "rb") as f:
            env_cfg, reward_cfg, robot_cfg, train_cfg = pickle.load(f)
    else:
        # Fallback: import from train script
        from xarm6_grasp_place_train import get_task_cfgs, get_train_cfg
        env_cfg, reward_cfg, robot_cfg = get_task_cfgs()
        train_cfg = get_train_cfg(args.exp_name, 0)

    # Override for eval
    env_cfg["num_envs"] = args.num_envs
    train_cfg["runner"]["max_iterations"] = 0
    train_cfg["runner"]["resume"] = False

    # Init Genesis with viewer
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=train_cfg["seed"],
    )

    # Create environment with viewer
    env = XArm6GraspPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=not args.headless,
    )

    # Set curriculum stage for evaluation
    if args.stage is not None:
        env.curriculum_stage = args.stage
        stage_source = "CLI argument"
    else:
        env.curriculum_stage, stage_source = infer_eval_stage(metrics_path)
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

    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    load_runner_checkpoint(runner, ckpt_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=gs.device)
    mode = "headless" if args.headless else "viewer"
    print(f"Model loaded. Running evaluation ({mode} mode)...")

    # Run evaluation loop
    obs, extras = env.reset()
    episode_count = 0
    step_count = 0
    total_reward = torch.zeros(args.num_envs, device=gs.device)
    episode_rewards = []

    grasp_count = 0
    place_count = 0

    while True:
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, done, extras = env.step(actions)
        total_reward += reward
        step_count += 1

        # Check for finished episodes
        done_envs = done.nonzero(as_tuple=True)[0]
        if len(done_envs) > 0:
            for idx in done_envs:
                ep_reward = total_reward[idx].item()
                ep_grasped = extras["episode_grasp_success"][idx].item()
                ep_placed = extras["episode_place_success"][idx].item()
                episode_rewards.append(ep_reward)
                episode_count += 1
                grasp_count += int(ep_grasped)
                place_count += int(ep_placed)

                print(
                    f"  Episode {episode_count}: "
                    f"reward={ep_reward:.1f}, "
                    f"grasped={'Yes' if ep_grasped else 'No'}, "
                    f"placed={'Yes' if ep_placed else 'No'}"
                )

            total_reward[done_envs] = 0.0

            # Check episode limit
            if args.episodes > 0 and episode_count >= args.episodes:
                break

    # Summary
    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({episode_count} episodes):")
        print(f"  Avg reward:       {avg_reward:.1f}")
        print(f"  Grasp success:    {grasp_count}/{episode_count} ({100*grasp_count/episode_count:.1f}%)")
        print(f"  Place success:    {place_count}/{episode_count} ({100*place_count/episode_count:.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
