"""Phase 0: validate G2 wrist fisheye framing (bottom 20% shows fingers)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import torch

EXAMPLES_XARM6 = Path(__file__).resolve().parents[1]
LEROBOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_XARM6))
sys.path.insert(0, str(LEROBOT_DIR))
sys.path.insert(0, str(EXAMPLES_XARM6.parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs

import camera_mount  # noqa: E402
import constants  # noqa: E402
from camera_mount import CameraMountConfig, check_finger_framing, make_offset_T
from xarm6_g2_il_env import XArm6G2ILEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify wrist camera framing on G2")
    parser.add_argument("--out", type=Path, default=Path("data/lerobot_debug/camera_framing.png"))
    parser.add_argument("--height", type=float, default=constants.CAMERA_HEIGHT_M)
    parser.add_argument(
        "--drive",
        type=float,
        default=None,
        help="single drive_joint; default checks open (0) and closed (0.85)",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)
    offset_T = make_offset_T(height_m=args.height)

    env = XArm6G2ILEnv(show_viewer=False, camera_mount=CameraMountConfig(offset_T=offset_T))
    drives = [args.drive] if args.drive is not None else [constants.DRIVE_OPEN, constants.DRIVE_CLOSED]

    all_ok = True
    for drive in drives:
        env.robot.control_dofs_position(
            torch.tensor([[drive]], device=gs.device, dtype=gs.tc_float),
            env.gripper_dof_idx,
        )
        for _ in range(15):
            env.scene.step()

        metrics = check_finger_framing(env, offset_T)
        label = "open" if drive < 0.1 else "closed"
        print(
            f"[{label}] left_y={metrics['left_y']:.1f} right_y={metrics['right_y']:.1f} "
            f"proj={metrics['proj_ok']} render={metrics['render_ok']} ok={metrics['ok']}"
        )
        all_ok = all_ok and metrics["ok"]

    rgb = env.render_wrist_rgb()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, rgb)

    print(f"Wrote {args.out} shape={rgb.shape}")
    print(f"Finger band y >= {constants.FINGER_BAND_Y_START}")
    print(f"Framing OK (open+closed): {all_ok}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
