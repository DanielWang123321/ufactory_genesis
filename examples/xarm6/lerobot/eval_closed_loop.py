#!/usr/bin/env python3
"""Closed-loop replay of dataset actions in Genesis (sanity check before ACT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LEROBOT_DIR = Path(__file__).resolve().parent
EXAMPLES_XARM6 = LEROBOT_DIR.parent
sys.path.insert(0, str(EXAMPLES_XARM6))
sys.path.insert(0, str(LEROBOT_DIR))
sys.path.insert(0, str(EXAMPLES_XARM6.parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs

import camera_mount
import constants
from xarm6_g2_il_env import XArm6G2ILEnv
from xarm6_lerobot_features import drive_from_gripper_openness, rotvec_to_quat_wxyz, unpack_ee_state


def load_episode_actions(root: Path, episode: int) -> np.ndarray:
    meta = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    row = ep_df.iloc[episode]
    start = int(row["dataset_from_index"])
    end = int(row["dataset_to_index"])

    data_files = sorted((root / "data").rglob("*.parquet"))
    data_df = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    actions = np.stack(data_df.iloc[start:end]["action"].to_numpy())
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay LeRobot actions in Genesis")
    parser.add_argument("--repo-id", type=str, default="local/xarm6_g2_sim_pickplace")
    parser.add_argument("--root", type=Path, default=Path("data/lerobot_datasets"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--camera-height", type=float, default=constants.CAMERA_HEIGHT_M)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)
    root = args.root / args.repo_id.replace("/", "_")
    actions = load_episode_actions(root, args.episode)[: args.max_steps]

    mount = camera_mount.CameraMountConfig(
        offset_T=camera_mount.make_offset_T(height_m=args.camera_height),
    )
    env = XArm6G2ILEnv(show_viewer=False, camera_mount=mount)

    for action in actions:
        pos, rotvec, openness = unpack_ee_state(action)
        world_pos = pos + env.base_pos
        quat_wxyz = rotvec_to_quat_wxyz(rotvec)
        quat_t = torch.tensor([quat_wxyz], device=gs.device, dtype=gs.tc_float)
        pos_t = torch.tensor([world_pos], device=gs.device, dtype=gs.tc_float)
        qpos = env.robot.inverse_kinematics(
            link=env.ik_link,
            pos=pos_t,
            quat=quat_t,
            dofs_idx_local=env.arm_dof_idx,
        )
        drive = drive_from_gripper_openness(openness)
        env.robot.control_dofs_position(qpos[:, env.arm_dof_idx], env.arm_dof_idx)
        env.robot.control_dofs_position(
            torch.tensor([[drive]], device=gs.device, dtype=gs.tc_float),
            env.gripper_dof_idx,
        )
        env.scene.step()

    obj = env.obj.get_pos()[0].detach().cpu().numpy()
    print(f"Replayed {len(actions)} steps from episode {args.episode}")
    print(f"Object position: {obj}")
    gs.destroy()


if __name__ == "__main__":
    main()
