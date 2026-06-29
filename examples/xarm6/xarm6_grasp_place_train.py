"""
xArm 6 Grasp-Place Task - RL Training Script using PPO (rsl-rl-lib).

Usage:
    source ~/envs/py312/bin/activate

    # Smoke test (1 env, with viewer)
    python examples/xarm6/xarm6_grasp_place_train.py -B 1 --max_iterations 5 -v

    # Small-scale training
    python examples/xarm6/xarm6_grasp_place_train.py -B 10 --max_iterations 50

    # Full training (2048 parallel envs)
    python examples/xarm6/xarm6_grasp_place_train.py -B 2048 --max_iterations 1000
"""

import argparse
import os
import pickle
import shutil
import sys
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xarm6_grasp_place_env import XArm6GraspPlaceEnv


def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
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
            "init_noise_std": 0.5,
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
        "num_steps_per_env": 200,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_task_cfgs(robot: str = "xarm6"):
    runtime = get_robot_runtime_profile(robot)
    if not runtime.task.grasp_place_supported or runtime.gripper_g2 is None:
        raise ValueError(f"{runtime.model.key} has no grasp-place task profile")
    env_cfg = {"num_envs": 10, **runtime.task.grasp_place_env_defaults}
    reward_cfg = {
        "reach": 4.0,
        "align": 3.0,
        "close_gripper": 3.0,
        "lift": 8.0,
        "grasp": 15.0,
        "place": 4.0,
        "release": 15.0,
        "success": 10.0,
        "action_penalty": 0.0005,
        "table_collision": 5.0,
    }
    gripper = runtime.gripper_g2
    robot_cfg = {
        "urdf_path": robot_urdf(runtime.model.key, "xarm6_with_gripper.urdf"),
        "ik_link_name": runtime.arm.ee_link,
        "gripper_link_names": list(gripper.finger_link_names),
        "arm_joint_names": list(runtime.arm.joint_names),
        "gripper_joint_name": gripper.drive_joint,
        "default_qpos": list(runtime.arm.default_qpos),
        "default_gripper_pos": gripper.open_pos,
        "kp": list(runtime.arm.kp),
        "kv": list(runtime.arm.kv),
        "force_lower": list(runtime.arm.force_lower),
        "force_upper": list(runtime.arm.force_upper),
        "gripper_kp": gripper.kp,
        "gripper_kv": gripper.kv,
        "gripper_force_lower": gripper.force_lower,
        "gripper_force_upper": gripper.force_upper,
        "all_gripper_joint_names": list(gripper.all_joint_names),
        "gripper_damping": gripper.damping,
        "gripper_frictionloss": gripper.frictionloss,
        "collision_monitor_links": ["link3", "link4", "link5"],
    }
    return env_cfg, reward_cfg, robot_cfg


def main():
    parser = argparse.ArgumentParser(description="xArm 6 Grasp-Place RL Training")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("-e", "--exp_name", type=str, default="xarm6-grasp-place")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=3000)
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
    env = XArm6GraspPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === CSV logging ===
    env.csv_log_path = str(log_dir / "metrics.csv")

    # === Train with PPO ===
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    print(f"\n=== Training complete ===")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"CSV log: {log_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
