#!/usr/bin/env python3
"""Validate recorded LeRobot dataset (stats + sample frames)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

LEROBOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEROBOT_DIR))

import constants
from xarm6_lerobot_features import assert_allowed_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify xArm6 G2 LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="local/xarm6_g2_sim_pickplace")
    parser.add_argument("--root", type=Path, default=Path("data/lerobot_datasets"))
    parser.add_argument("--out", type=Path, default=Path("data/lerobot_debug/dataset_preview.png"))
    args = parser.parse_args()

    from hf_lerobot import get_lerobot_dataset_class

    LeRobotDataset = get_lerobot_dataset_class()
    root = args.root / args.repo_id.replace("/", "_")
    ds = LeRobotDataset(args.repo_id, root=root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature_keys = set(info.get("features", {}).keys())
    assert_allowed_features({k: {} for k in feature_keys if k.startswith("observation") or k == "action"})

    idx = 0
    sample = ds[idx]
    state = np.asarray(sample["observation.state"])
    action = np.asarray(sample["action"])
    img = sample[constants.CAMERA_FEATURE_KEY]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    print(f"Dataset: {args.repo_id}")
    print(f"Episodes: {ds.num_episodes}, frames: {len(ds)}, fps: {info.get('fps')}")
    print(f"state shape={state.shape}, action shape={action.shape}, image shape={img.shape}")
    bad = [k for k in feature_keys if any(x in k.lower() for x in ("tof", "depth", "pointcloud"))]
    if bad:
        raise SystemExit(f"ToF/depth features present (not allowed): {bad}")
    print("ToF/depth: not in features (OK)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(args.out, img)
    print(f"Wrote preview {args.out}")


if __name__ == "__main__":
    main()
