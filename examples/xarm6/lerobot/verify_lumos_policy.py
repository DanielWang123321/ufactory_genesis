#!/usr/bin/env python3
"""Phase 0: confirm ToF/depth are excluded from LeRobot training features."""

from __future__ import annotations

import sys
from pathlib import Path

LEROBOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEROBOT_DIR))

import dataset_utils
from xarm6_lerobot_features import assert_allowed_features

FORBIDDEN_SUBSTRINGS = ("tof", "depth", "pointcloud", "pcd", "lidar")


def main() -> None:
    features = dataset_utils.build_dataset_features()
    assert_allowed_features(features)

    for key in features:
        low = key.lower()
        if any(s in low for s in FORBIDDEN_SUBSTRINGS):
            raise SystemExit(f"Forbidden modality in features: {key}")

    print("LeRobot feature whitelist OK (RGB + 7D EE only).")
    print("ToF / depth / pointcloud: excluded from policy I/O.")
    print("Real pose: Vive preferred; ToF+SLAM only when Vive absent (labels only).")


if __name__ == "__main__":
    main()
