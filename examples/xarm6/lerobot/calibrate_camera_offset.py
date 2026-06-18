#!/usr/bin/env python3
"""Sweep camera mount offsets to land G2 fingers in bottom 20% band."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

EXAMPLES_XARM6 = Path(__file__).resolve().parents[1]
LEROBOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_XARM6))
sys.path.insert(0, str(LEROBOT_DIR))
sys.path.insert(0, str(EXAMPLES_XARM6.parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs

import constants
from camera_mount import CameraMountConfig, fingers_in_bottom_band, project_world_points_to_image
from verify_camera_framing import _camera_pose_world, check_framing
from xarm6_g2_il_env import XArm6G2ILEnv


def make_offset(rx_deg: float, rz_deg: float, height: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = ScipyR.from_euler("x", rx_deg, degrees=True).as_matrix()
    if abs(rz_deg) > 1e-6:
        T[:3, :3] = T[:3, :3] @ ScipyR.from_euler("z", rz_deg, degrees=True).as_matrix()
    T[:3, 3] = [0.0, 0.0, height]
    return T


def main() -> None:
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)
    out_dir = Path("data/lerobot_debug/calib")
    out_dir.mkdir(parents=True, exist_ok=True)

    best = None
    for rx in (-90, -120, -60, 90, 180):
        for height in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
            for rz in (0, 90, -90, 180):
                offset = make_offset(rx, rz, height)
                env = XArm6G2ILEnv(
                    show_viewer=False,
                    camera_mount=CameraMountConfig(offset_T=offset),
                )
                metrics = check_framing(env, offset)
                score = metrics["left_y"] + (metrics["right_y"] if not np.isnan(metrics["right_y"]) else metrics["left_y"])
                key = (metrics["ok"], score)
                print(
                    f"rx={rx:4d} rz={rz:4d} h={height:.2f} "
                    f"ly={metrics['left_y']:.0f} ry={metrics['right_y']:.0f} ok={metrics['ok']}"
                )
                if best is None or (key[0] and not best[0][0]) or (key[0] == best[0][0] and key[1] > best[0][1]):
                    best = (key, rx, rz, height, metrics)
                gs.destroy()

    if best:
        _, rx, rz, height, metrics = best
        print(f"BEST rx={rx} rz={rz} h={height} metrics={metrics}")
        offset = make_offset(rx, rz, height)
        env = XArm6G2ILEnv(show_viewer=False, camera_mount=CameraMountConfig(offset_T=offset))
        rgb = env.render_wrist_rgb()
        imageio.imwrite(out_dir / "best.png", rgb)
        gs.destroy()


if __name__ == "__main__":
    main()
