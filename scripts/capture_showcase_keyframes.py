#!/usr/bin/env python3
"""Headless keyframe capture for packaging showcase startup pose debugging."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES_ROOT))
sys.path.insert(0, str(ROOT / "examples" / "xarm6"))

import _bootstrap  # noqa: F401
import genesis as gs
from _packaging_scene import build_packaging_scene, make_layout, packaging_camera
from ufactory.glb_visual import enable_glb_pbr_surfaces
from xarm6_g2_showcase import (
  SIM_DT,
  ShowcaseRobotCtx,
  _reset_block,
  _scale_steps,
  hold_robot_home,
  init_showcase_robot,
  run_pick_place_cycle,
)

DEFAULT_OUT = ROOT / "debug" / "showcase_keyframes"


@dataclass
class KeyframeMeta:
  name: str
  file: str
  link6_pos: list[float]
  arm_qpos_deg: list[float] | None
  gripper_q: float | None


def _render_frame(cam) -> np.ndarray:
  rgb, _, _, _ = cam.render()
  frame = rgb[0] if isinstance(rgb, np.ndarray) and rgb.ndim == 4 else rgb
  return np.asarray(frame)


def _read_robot_state(robot, ctx: ShowcaseRobotCtx | None) -> tuple[list[float], list[float] | None, float | None]:
  link6 = robot.get_link("link6").get_pos()[0].cpu().numpy().tolist()
  if ctx is None or not ctx.arm_dof_idx:
    return link6, None, None
  arm_q = robot.get_dofs_position(ctx.arm_dof_idx)[0].cpu().numpy()
  arm_deg = np.degrees(arm_q).round(2).tolist()
  grip = robot.get_dofs_position(ctx.gripper_dof_idx)[0].item()
  return link6, arm_deg, grip


def _snapshot(
  robot,
  ctx: ShowcaseRobotCtx | None,
  name: str,
  out_dir: Path,
  cam,
) -> KeyframeMeta:
  link6, arm_deg, grip = _read_robot_state(robot, ctx)
  frame = _render_frame(cam)
  path = out_dir / f"{name}.png"
  imageio.imwrite(path, frame)
  meta = KeyframeMeta(name, str(path.relative_to(ROOT)), link6, arm_deg, grip)
  print(f"  [{name}] link6={link6} arm_deg={arm_deg} grip={grip} -> {path}")
  return meta


def capture_keyframes(
  *,
  out_dir: Path,
  settle_steps: int = 40,
  speed: float = 1.0,
  run_phase0: bool = True,
) -> list[KeyframeMeta]:
  out_dir.mkdir(parents=True, exist_ok=True)
  enable_glb_pbr_surfaces()
  gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)

  layout = make_layout()
  cam_pos, cam_lookat = packaging_camera(layout.table_top_z)
  scene, robot, block, layout = build_packaging_scene(
    layout.table_top_z, sim_dt=SIM_DT, show_viewer=False, build_scene=False
  )
  cam = scene.add_camera(
    res=(1280, 720),
    pos=cam_pos,
    lookat=cam_lookat,
    fov=35,
  )
  scene.build(n_envs=1)

  metas: list[KeyframeMeta] = []
  metas.append(_snapshot(robot, None, "01_post_build", out_dir, cam))

  _reset_block(block, layout)
  for _ in range(_scale_steps(settle_steps, speed)):
    scene.step()
  metas.append(_snapshot(robot, None, "02_post_settle_no_home", out_dir, cam))

  ctx = init_showcase_robot(robot, layout, scene)
  metas.append(_snapshot(robot, ctx, "03_post_init_home", out_dir, cam))

  hold_robot_home(robot, scene, ctx, steps=_scale_steps(40, speed))
  metas.append(_snapshot(robot, ctx, "04_after_phase0_hold", out_dir, cam))

  if run_phase0:
    run_pick_place_cycle(scene, robot, block, layout, speed=speed, ctx=ctx, stop_after_phase0=True)
    metas.append(_snapshot(robot, ctx, "05_cycle_phase0_done", out_dir, cam))

  meta_path = out_dir / "keyframes.json"
  payload = {
    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "settle_steps": settle_steps,
    "note": "02_post_settle_no_home shows URDF zero-qpos bug if init is delayed",
    "keyframes": [asdict(m) for m in metas],
  }
  meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(f"\nWrote {len(metas)} keyframes to {out_dir}")
  print(f"Metadata: {meta_path}")
  return metas


def main() -> None:
  parser = argparse.ArgumentParser(description="Capture packaging showcase startup keyframes")
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--settle-steps", type=int, default=40)
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--no-phase0", action="store_true", help="Skip run_pick_place_cycle phase 0")
  args = parser.parse_args()
  capture_keyframes(
    out_dir=args.out_dir.resolve(),
    settle_steps=args.settle_steps,
    speed=args.speed,
    run_phase0=not args.no_phase0,
  )


if __name__ == "__main__":
  main()
