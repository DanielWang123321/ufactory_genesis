#!/usr/bin/env python3
"""Convert FastUMI Pro export sessions to LeRobotDataset (RGB + EE only, no ToF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LEROBOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEROBOT_DIR))

import dataset_utils
from xarm6_lerobot_features import (
    gripper_openness_from_width_mm,
    pack_ee_state,
    quat_wxyz_to_rotvec,
)


def _load_trajectory(path: Path) -> np.ndarray:
    """Parse merged_trajectory.txt: timestamp x y z qx qy qz qw."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append([float(x) for x in parts[:8]])
    return np.asarray(rows, dtype=np.float64)


def _load_clamp(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append([float(parts[0]), float(parts[1])])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 2))


def convert_session(session_dir: Path, dataset, task: str) -> int:
    import imageio.v2 as imageio

    traj_files = list(session_dir.rglob("Merged_Trajectory/merged_trajectory.txt"))
    if not traj_files:
        return 0
    traj = _load_trajectory(traj_files[0])
    clamp_path = next(session_dir.rglob("Clamp_Data/clamp_data_tum.txt"), None)
    clamp = _load_clamp(clamp_path) if clamp_path else np.zeros((0, 2))

    video_path = next(session_dir.rglob("RGB_Images/video.mp4"), None)
    if video_path is None:
        video_path = next(session_dir.rglob("RGB_Images/Frames"), None)
    if video_path is None:
        return 0

    if video_path.is_dir():
        frames = sorted(video_path.glob("*.jpg")) + sorted(video_path.glob("*.png"))
        images = [imageio.imread(p) for p in frames]
    else:
        reader = imageio.get_reader(video_path)
        images = [im for im in reader]
        reader.close()

    n = min(len(images), len(traj))
    for i in range(n):
        _, x, y, z, qx, qy, qz, qw = traj[i]
        rotvec = quat_wxyz_to_rotvec(np.array([qw, qx, qy, qz]))
        if len(clamp) > 0:
            # nearest clamp width by index (sync refinement optional)
            ci = min(i, len(clamp) - 1)
            openness = gripper_openness_from_width_mm(clamp[ci, 1])
        else:
            openness = 0.5
        state = pack_ee_state(np.array([x, y, z]), rotvec, openness)
        action = state.copy()
        rgb = images[i]
        if rgb.shape[0] != 1280 or rgb.shape[1] != 1280:
            import cv2

            rgb = cv2.resize(rgb, (1280, 1280), interpolation=cv2.INTER_AREA)
        dataset_utils.add_frame_to_dataset(dataset, rgb, state, action, task=task)
    dataset.save_episode()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="FastUMI Pro → LeRobot (no ToF)")
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default="local/xarm6_g2_real_pickplace")
    parser.add_argument("--root", type=Path, default=Path("data/lerobot_datasets"))
    parser.add_argument("--task", type=str, default="pick cube and place at target")
    args = parser.parse_args()

    dataset = dataset_utils.create_lerobot_dataset(
        args.repo_id, root=args.root / args.repo_id.replace("/", "_")
    )
    total = 0
    for session in sorted(args.session_root.rglob("session_*")):
        if not session.is_dir():
            continue
        n = convert_session(session, dataset, args.task)
        if n:
            print(f"Converted {session.name}: {n} frames")
            total += n
    print(f"Total frames: {total}")
    print("Note: ToF_PointClouds intentionally skipped (pose-only use on device).")


if __name__ == "__main__":
    main()
