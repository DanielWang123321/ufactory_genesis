#!/usr/bin/env python3
"""Record Genesis xArm6+G2 pick-place demos to LeRobotDataset v3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LEROBOT_DIR = Path(__file__).resolve().parent
EXAMPLES_XARM6 = LEROBOT_DIR.parent
sys.path.insert(0, str(EXAMPLES_XARM6))
sys.path.insert(0, str(LEROBOT_DIR))
sys.path.insert(0, str(EXAMPLES_XARM6.parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs

import camera_mount
import constants
import dataset_utils
import pick_place_script as pps
from xarm6_g2_il_env import XArm6G2ILEnv
from xarm6_lerobot_features import pack_ee_state


def record_episode(env: XArm6G2ILEnv, dataset, cfg: pps.PickPlaceConfig) -> int:
    finger_z = pps.measure_finger_z_offset(env.ik_link, env.left_finger, env.right_finger)
    cfg.finger_z_offset = finger_z
    traj = pps.build_pick_place_trajectory(cfg)
    n = 0
    for step in traj:
        state = env.get_ee_state_base()
        pps.execute_trajectory_step(
            env.robot,
            env.scene,
            env.ik_link,
            env.arm_dof_idx,
            env.gripper_dof_idx,
            env.down_quat,
            step,
        )
        rgb = env.render_wrist_rgb()
        drive = step.drive_joint
        openness = 1.0 - drive / constants.DRIVE_CLOSED
        pos_base = np.array(step.link6_pos, dtype=np.float64) - env.base_pos
        rotvec = env.get_ee_state_base()[3:6].copy()
        action = pack_ee_state(pos_base, rotvec, openness)
        dataset_utils.add_frame_to_dataset(dataset, rgb, state, action)
        n += 1
    dataset.save_episode()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sim LeRobot dataset for xArm6+G2")
    parser.add_argument("--repo-id", type=str, default="local/xarm6_g2_sim_pickplace")
    parser.add_argument("--root", type=Path, default=Path("data/lerobot_datasets"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--randomize", action="store_true", help="Domain randomization (Phase 2)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera-height", type=float, default=constants.CAMERA_HEIGHT_M)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    mount = camera_mount.CameraMountConfig(
        offset_T=camera_mount.make_offset_T(height_m=args.camera_height),
    )
    root = args.root / args.repo_id.replace("/", "_")
    resume = root.exists() and (root / "meta" / "info.json").is_file()
    dataset = dataset_utils.create_lerobot_dataset(args.repo_id, root=root, resume=resume)

    rng = np.random.default_rng(args.seed)
    env = XArm6G2ILEnv(show_viewer=False, camera_mount=mount)
    total_frames = 0
    try:
        for ep in range(args.episodes):
            cfg = pps.PickPlaceConfig(table_height=env.table_height)
            if args.randomize:
                rnd = XArm6G2ILEnv.sample_randomization(int(rng.integers(0, 2**31)))
                ox, px = env.apply_randomization(rnd)
                cfg.obj_xy = ox
                cfg.place_xy = px
            else:
                env.reset_object()
            frames = record_episode(env, dataset, cfg)
            total_frames += frames
            print(f"Episode {ep + 1}/{args.episodes}: {frames} frames")
    finally:
        gs.destroy()

    print(f"Done. {args.episodes} episodes, {total_frames} frames → {root}")


if __name__ == "__main__":
    main()
