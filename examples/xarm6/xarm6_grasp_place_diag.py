"""
xArm 6 Grasp-Place - Diagnostic Script.

Records per-step trajectory data for analysis.
Outputs key metrics and detects anomalous behaviors.

Usage:
    python examples/xarm6/xarm6_grasp_place_diag.py
    python examples/xarm6/xarm6_grasp_place_diag.py --checkpoint logs/xarm6-grasp-place/model_2500.pt
"""

import argparse
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
    pts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    return pts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("-e", "--exp_name", type=str, default="xarm6-grasp-place")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    log_dir = Path("logs") / args.exp_name
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(log_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, train_cfg = pickle.load(f)

    env_cfg["num_envs"] = 1
    train_cfg["runner"]["max_iterations"] = 0

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)

    env = XArm6GraspPlaceEnv(
        env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=False,
    )
    env.curriculum_stage = 4

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(str(ckpt_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=gs.device)

    obs, extras = env.reset()
    episode_count = 0

    # Per-step recording for current episode
    traj = {
        "ee_pos": [], "obj_pos": [], "target_pos": [], "gripper_pos": [],
        "joint_qpos": [], "grasped": [], "actions": [],
    }

    while episode_count < args.episodes:
        with torch.no_grad():
            actions = policy(obs)

        # Record BEFORE step
        ee_pos = env._finger_center_pos()[0].cpu().tolist()
        obj_pos = env.obj.get_pos()[0].cpu().tolist()
        target_pos = env.target_pos[0].cpu().tolist()
        gripper_pos = env.robot.get_dofs_position(env.gripper_dof_idx)[0].item()
        joint_qpos = env.robot.get_dofs_position(env.arm_dof_idx)[0].cpu().tolist()
        grasped = env.grasped[0].item()

        traj["ee_pos"].append(ee_pos)
        traj["obj_pos"].append(obj_pos)
        traj["target_pos"].append(target_pos)
        traj["gripper_pos"].append(gripper_pos)
        traj["joint_qpos"].append(joint_qpos)
        traj["grasped"].append(grasped)
        traj["actions"].append(actions[0].cpu().tolist())

        obs, reward, done, extras = env.step(actions)

        if done[0]:
            episode_count += 1
            analyze_episode(traj, episode_count, env)
            traj = {k: [] for k in traj}

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


def analyze_episode(traj, ep_num, env):
    import numpy as np
    n = len(traj["ee_pos"])
    ee = np.array(traj["ee_pos"])
    obj = np.array(traj["obj_pos"])
    tgt = np.array(traj["target_pos"])
    grip = np.array(traj["gripper_pos"])
    qpos = np.array(traj["joint_qpos"])
    grasped = np.array(traj["grasped"])
    actions = np.array(traj["actions"])

    table_h = env.table_height
    obj_half_z = env.obj_size[2] / 2

    print(f"\n{'='*70}")
    print(f"EPISODE {ep_num} ANALYSIS ({n} steps)")
    print(f"{'='*70}")

    # --- Positions ---
    print(f"\n--- Initial State (step 0) ---")
    print(f"  EE pos:     [{ee[0,0]:.4f}, {ee[0,1]:.4f}, {ee[0,2]:.4f}]")
    print(f"  Object pos: [{obj[0,0]:.4f}, {obj[0,1]:.4f}, {obj[0,2]:.4f}]")
    print(f"  Target pos: [{tgt[0,0]:.4f}, {tgt[0,1]:.4f}, {tgt[0,2]:.4f}]")
    print(f"  Table height: {table_h}")

    # --- EE trajectory analysis ---
    print(f"\n--- End-Effector Trajectory ---")
    print(f"  Z range:  [{ee[:,2].min():.4f}, {ee[:,2].max():.4f}]")
    print(f"  X range:  [{ee[:,0].min():.4f}, {ee[:,0].max():.4f}]")
    print(f"  Y range:  [{ee[:,1].min():.4f}, {ee[:,1].max():.4f}]")
    below_table = (ee[:, 2] < table_h).sum()
    print(f"  Steps EE below table: {below_table}/{n} ({100*below_table/n:.1f}%)")

    # --- Object trajectory ---
    print(f"\n--- Object Trajectory ---")
    print(f"  Z range:  [{obj[:,2].min():.4f}, {obj[:,2].max():.4f}]")
    print(f"  Initial Z (should be ~{table_h + obj_half_z:.4f}): {obj[0,2]:.4f}")
    obj_moved = np.linalg.norm(obj[-1] - obj[0])
    print(f"  Total displacement: {obj_moved:.4f} m")
    obj_lifted = (obj[:, 2] > table_h + obj_half_z + 0.005).sum()
    print(f"  Steps object lifted: {obj_lifted}/{n} ({100*obj_lifted/n:.1f}%)")
    max_lift = obj[:, 2].max() - (table_h + obj_half_z)
    print(f"  Max lift height: {max_lift:.4f} m")

    # --- Gripper ---
    print(f"\n--- Gripper ---")
    print(f"  Range: [{grip.min():.4f}, {grip.max():.4f}] (0=closed, 0.85=open)")
    closed_steps = (grip < 0.5).sum()
    print(f"  Steps closed (<0.5): {closed_steps}/{n} ({100*closed_steps/n:.1f}%)")

    # --- Grasp detection ---
    print(f"\n--- Grasp Detection ---")
    grasped_steps = grasped.sum()
    print(f"  Steps grasped: {int(grasped_steps)}/{n} ({100*grasped_steps/n:.1f}%)")
    if grasped_steps > 0:
        first_grasp = np.argmax(grasped)
        print(f"  First grasp at step: {first_grasp}")
        print(f"    EE pos at grasp: [{ee[first_grasp,0]:.4f}, {ee[first_grasp,1]:.4f}, {ee[first_grasp,2]:.4f}]")
        print(f"    Obj pos at grasp: [{obj[first_grasp,0]:.4f}, {obj[first_grasp,1]:.4f}, {obj[first_grasp,2]:.4f}]")

    # --- Joint positions (check for extreme values) ---
    print(f"\n--- Joint Positions ---")
    for j in range(qpos.shape[1]):
        print(f"  Joint {j+1}: [{qpos[:,j].min():.3f}, {qpos[:,j].max():.3f}]")

    # --- Distance analysis ---
    ee_obj_dist = np.linalg.norm(ee - obj, axis=1)
    obj_tgt_dist = np.linalg.norm(obj - tgt, axis=1)
    print(f"\n--- Distances ---")
    print(f"  EE-to-Object:  min={ee_obj_dist.min():.4f}, final={ee_obj_dist[-1]:.4f}")
    print(f"  Obj-to-Target: start={obj_tgt_dist[0]:.4f}, min={obj_tgt_dist.min():.4f}, final={obj_tgt_dist[-1]:.4f}")

    # --- Key frames (sample 5 evenly spaced) ---
    print(f"\n--- Key Frames ---")
    key_steps = [0, n//4, n//2, 3*n//4, n-1]
    print(f"  {'Step':>5} | {'EE_X':>7} {'EE_Y':>7} {'EE_Z':>7} | {'Obj_X':>7} {'Obj_Y':>7} {'Obj_Z':>7} | {'Grip':>5} | {'Grasp':>5}")
    print(f"  {'-'*5}-+-{'-'*23}-+-{'-'*23}-+-{'-'*5}-+-{'-'*5}")
    for s in key_steps:
        print(
            f"  {s:5d} | {ee[s,0]:7.4f} {ee[s,1]:7.4f} {ee[s,2]:7.4f} | "
            f"{obj[s,0]:7.4f} {obj[s,1]:7.4f} {obj[s,2]:7.4f} | "
            f"{grip[s]:5.3f} | {'YES' if grasped[s] else 'no':>5}"
        )

    # --- Anomaly detection ---
    print(f"\n--- Anomaly Check ---")
    anomalies = []
    if below_table > n * 0.1:
        anomalies.append(f"EE spends {100*below_table/n:.0f}% of time below table!")
    if max_lift < 0.005 and grasped_steps > 0:
        anomalies.append(f"Grasp detected but object barely lifted (max {max_lift:.4f}m)")
    if ee[:, 2].min() < 0.1:
        anomalies.append(f"EE reaches dangerously low Z={ee[:,2].min():.4f}")
    if obj[:, 2].min() < table_h + obj_half_z - 0.01:
        anomalies.append(f"Object falls below table surface (Z={obj[:,2].min():.4f})")
    if grip.max() - grip.min() < 0.1:
        anomalies.append(f"Gripper barely moves (range {grip.min():.3f}-{grip.max():.3f})")

    if anomalies:
        for a in anomalies:
            print(f"  [!] {a}")
    else:
        print(f"  No anomalies detected.")


if __name__ == "__main__":
    main()
