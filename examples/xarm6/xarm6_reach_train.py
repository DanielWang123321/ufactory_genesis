"""
xArm 6 Reach Task - RL Training Script using PPO (rsl-rl-lib).

Usage:
    source ~/envs/py312/bin/activate

    # Step 1 quick smoke test (1 env, few iterations, with viewer)
    python examples/xarm6/xarm6_reach_train.py -B 1 --max_iterations 10 -v

    # Step 2: Full training (2048 parallel envs)
    python examples/xarm6/xarm6_reach_train.py -B 2048 --max_iterations 300

    # Step 3: 10-arm parallel training
    python examples/xarm6/xarm6_reach_train.py -B 10 --max_iterations 300 -v
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata
from pathlib import Path

# Validate rsl-rl-lib version
try:
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
from ufactory.paths import robot_urdf
from ufactory.robot_params import get_robot_runtime_profile, robot_runtime_cli_choices

# Allow importing from same directory
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xarm6_reach_env import XArm6ReachEnv


def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_task_cfgs(robot: str = "xarm6"):
    runtime = get_robot_runtime_profile(robot)
    env_cfg = {"num_envs": 10, **runtime.task.reach_env_defaults}
    reward_cfg = {
        "reach": 1.0,
        "action_penalty": 0.001,
    }
    robot_cfg = {
        "urdf_path": robot_urdf(runtime.model.key),
        "ee_link_name": runtime.arm.ee_link,
        "joint_names": list(runtime.arm.joint_names),
        "default_qpos": list(runtime.arm.default_qpos),
        "kp": list(runtime.arm.kp),
        "kv": list(runtime.arm.kv),
        "force_lower": list(runtime.arm.force_lower),
        "force_upper": list(runtime.arm.force_upper),
    }
    return env_cfg, reward_cfg, robot_cfg


def main():
    parser = argparse.ArgumentParser(description="xArm 6 Reach Task RL Training")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("-e", "--exp_name", type=str, default="xarm6-reach")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()

    # === Configs ===
    env_cfg, reward_cfg, robot_cfg = get_task_cfgs(args.robot)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # === Log dir ===
    log_dir = Path("logs") / args.exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, reward_cfg, robot_cfg, train_cfg], f)

    # === Init Genesis ===
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=train_cfg["seed"],
    )

    # === Create environment ===
    env_cfg["num_envs"] = args.num_envs
    env = XArm6ReachEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === Train with PPO ===
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
