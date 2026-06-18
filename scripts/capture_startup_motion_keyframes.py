#!/usr/bin/env python3
"""Capture startup keyframes for diagnosing visible arm motion in GLB viewer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
if str(EXAMPLES_ROOT) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_ROOT))
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import _bootstrap  # noqa: F401,E402
import genesis as gs  # noqa: E402
from _bio_gripper_demo import (  # noqa: E402
  BIO_GRIPPER_OPEN,
  bio_gripper_demo_target,
  bio_gripper_dof_indices,
  control_bio_gripper_pose,
  set_bio_gripper_pose,
  setup_bio_gripper_pd,
)
from _gripper_demo import (  # noqa: E402
  GRIPPER_OPEN,
  control_gripper_pose,
  gripper_demo_target,
  gripper_dof_indices,
  set_gripper_pose,
  setup_gripper_pd,
)
from _lite6_gripper_demo import (  # noqa: E402
  LITE6_GRIPPER_OPEN,
  control_lite6_gripper_pose,
  lite6_gripper_demo_target,
  lite6_gripper_dof_indices,
  set_lite6_gripper_pose,
  setup_lite6_gripper_pd,
)
from _robot_viewer import _apply_kinematic_hold, _disable_robot_pd, _kinematic_step  # noqa: E402
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface  # noqa: E402
from ufactory.paths import robot_visual_glb_urdf  # noqa: E402
from ufactory.robot_registry import arm_link_names, get_robot_profile, joint_names  # noqa: E402

OUT_ROOT = PROJECT_ROOT / ".cursor" / "startup_motion_keyframes"
CAMERA_POS = (1.5, -1.5, 1.5)
CAMERA_LOOKAT = (0.0, 0.0, 0.4)
CAMERA_FOV = 40


def _to_numpy(value) -> np.ndarray:
  if hasattr(value, "detach"):
    value = value.detach()
  if hasattr(value, "cpu"):
    value = value.cpu()
  if hasattr(value, "numpy"):
    value = value.numpy()
  return np.asarray(value)


def _sync_visual_state(scene) -> None:
  context = getattr(scene.visualizer, "_context", None)
  if context is not None:
    context._t = -1
    context.update(force_render=True)


def _render(scene, cam, out: Path) -> np.ndarray:
  _sync_visual_state(scene)
  rgb, _, _, _ = cam.render()
  frame = rgb[0] if isinstance(rgb, np.ndarray) and rgb.ndim == 4 else rgb
  frame = np.asarray(frame)
  imageio.imwrite(out, frame)
  return frame


def _image_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
  aa = a.astype(np.float32)
  bb = b.astype(np.float32)
  diff = np.abs(aa - bb)
  gray = diff.max(axis=-1) if diff.ndim == 3 else diff
  return {
    "mean_abs": float(diff.mean()),
    "max_abs": float(diff.max()),
    "changed_px_gt_8": int((gray > 8.0).sum()),
    "changed_px_gt_24": int((gray > 24.0).sum()),
  }


def _state(robot, profile, arm_dof_idx: list[int]) -> dict:
  qpos = _to_numpy(robot.get_dofs_position()).reshape(-1)
  links = {link.name.split("/")[-1]: link for link in robot.links}
  link_pos = {}
  for name in arm_link_names(profile):
    if name not in links:
      continue
    link_pos[name] = _to_numpy(links[name].get_pos()).reshape(-1)[:3]
  return {
    "arm_q": qpos[arm_dof_idx].copy() if arm_dof_idx else np.zeros(0),
    "link_pos": link_pos,
  }


def _state_delta(a: dict, b: dict) -> dict:
  q_delta = b["arm_q"] - a["arm_q"]
  per_link = {}
  max_norm = 0.0
  max_abs_z = 0.0
  for name, apos in a["link_pos"].items():
    if name not in b["link_pos"]:
      continue
    delta = b["link_pos"][name] - apos
    norm = float(np.linalg.norm(delta) * 1000.0)
    dz = float(delta[2] * 1000.0)
    max_norm = max(max_norm, norm)
    max_abs_z = max(max_abs_z, abs(dz))
    per_link[name] = {
      "dx_mm": round(float(delta[0] * 1000.0), 4),
      "dy_mm": round(float(delta[1] * 1000.0), 4),
      "dz_mm": round(dz, 4),
      "norm_mm": round(norm, 4),
    }
  return {
    "max_abs_q_rad": float(np.max(np.abs(q_delta))) if len(q_delta) else 0.0,
    "max_link_norm_mm": max_norm,
    "max_abs_link_z_mm": max_abs_z,
    "links": per_link,
  }


def _setup_gripper_demo(
  robot,
  *,
  gripper_demo: bool,
  gripper_dof_idx: list[int],
  all_gripper_dof_idx: list[int],
  lite6_gripper_dof_idx: list[int],
  all_lite6_gripper_dof_idx: list[int],
  bio_gripper_dof_idx: list[int],
  all_bio_gripper_dof_idx: list[int],
) -> str:
  if not gripper_demo:
    return "none"
  if gripper_dof_idx:
    setup_gripper_pd(robot, gripper_dof_idx, all_gripper_dof_idx)
    set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)
    return "g2"
  if lite6_gripper_dof_idx:
    setup_lite6_gripper_pd(robot, lite6_gripper_dof_idx, all_lite6_gripper_dof_idx)
    set_lite6_gripper_pose(
      robot,
      lite6_gripper_dof_idx,
      all_lite6_gripper_dof_idx,
      LITE6_GRIPPER_OPEN,
    )
    return "lite6"
  if bio_gripper_dof_idx:
    setup_bio_gripper_pd(robot, bio_gripper_dof_idx, all_bio_gripper_dof_idx)
    set_bio_gripper_pose(robot, bio_gripper_dof_idx, all_bio_gripper_dof_idx, BIO_GRIPPER_OPEN)
    return "bio"
  return "missing"


def _control_gripper_demo(
  robot,
  kind: str,
  step: int,
  gripper_dof_idx: list[int],
  all_gripper_dof_idx: list[int],
  lite6_gripper_dof_idx: list[int],
  all_lite6_gripper_dof_idx: list[int],
  bio_gripper_dof_idx: list[int],
  all_bio_gripper_dof_idx: list[int],
) -> None:
  if kind == "g2":
    control_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, gripper_demo_target(step))
  elif kind == "lite6":
    control_lite6_gripper_pose(
      robot,
      lite6_gripper_dof_idx,
      all_lite6_gripper_dof_idx,
      lite6_gripper_demo_target(step),
    )
  elif kind == "bio":
    control_bio_gripper_pose(
      robot,
      bio_gripper_dof_idx,
      all_bio_gripper_dof_idx,
      bio_gripper_demo_target(step),
    )


def _apply_hold(
  robot,
  arm_dof_idx: list[int],
  home: np.ndarray,
  *,
  arm_kinematic_hold: bool,
  idle_gripper_kinematic_hold: bool,
  all_gripper_dof_idx: list[int],
  all_lite6_gripper_dof_idx: list[int],
  all_bio_gripper_dof_idx: list[int],
) -> None:
  _apply_kinematic_hold(
    robot,
    arm_dof_idx,
    home,
    hold_arm=arm_kinematic_hold,
    hold_gripper=idle_gripper_kinematic_hold,
    all_gripper_dof_idx=all_gripper_dof_idx,
    all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
    all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
  )


def _capture_phase(
  scene,
  cam,
  robot,
  profile,
  arm_dof_idx: list[int],
  out_dir: Path,
  idx: int,
  name: str,
  frames: dict[str, np.ndarray],
  states: dict[str, dict],
) -> None:
  path = out_dir / f"{idx:02d}_{name}.png"
  frames[name] = _render(scene, cam, path)
  states[name] = _state(robot, profile, arm_dof_idx)
  print(f"wrote {path}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--robot", default="xarm5_1305")
  parser.add_argument("--bio-gripper-g2", action="store_true")
  parser.add_argument("--gripper-g2", action="store_true")
  parser.add_argument("--lite6-gripper", action="store_true")
  parser.add_argument("--movable", action="store_true")
  parser.add_argument("--gripper-demo", action="store_true")
  parser.add_argument("--steps", type=int, default=100)
  args = parser.parse_args()

  profile = get_robot_profile(args.robot)
  out_dir = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
  out_dir.mkdir(parents=True, exist_ok=True)
  report_path = out_dir / "report.json"

  enable_glb_pbr_surfaces()
  gs.init(backend=gs.gpu, logging_level="error")
  urdf_path = robot_visual_glb_urdf(
    args.robot,
    with_bio_gripper_g2=args.bio_gripper_g2,
    with_gripper_g2=args.gripper_g2,
    with_lite6_gripper=args.lite6_gripper,
    movable=args.movable,
  )
  scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
      camera_pos=CAMERA_POS,
      camera_lookat=CAMERA_LOOKAT,
      camera_fov=CAMERA_FOV,
    ),
  )
  scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
  robot = scene.add_entity(
    gs.morphs.URDF(file=urdf_path, fixed=True, requires_jac_and_IK=True),
    surface=glb_view_surface(),
  )
  cam = scene.add_camera(res=(960, 720), pos=CAMERA_POS, lookat=CAMERA_LOOKAT, fov=CAMERA_FOV)
  scene.build()

  jnames = joint_names(profile)
  home = np.zeros(profile.dof)
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  arm_dof_idx = [joint_map[name].dofs_idx_local[0] for name in jnames if name in joint_map]
  gripper_dof_idx, all_gripper_dof_idx = gripper_dof_indices(robot)
  lite6_gripper_dof_idx, all_lite6_gripper_dof_idx = lite6_gripper_dof_indices(robot)
  bio_gripper_dof_idx, all_bio_gripper_dof_idx = bio_gripper_dof_indices(robot)
  arm_kinematic_hold = True
  idle_gripper_kinematic_hold = not args.gripper_demo

  frames: dict[str, np.ndarray] = {}
  states: dict[str, dict] = {}
  _capture_phase(scene, cam, robot, profile, arm_dof_idx, out_dir, 0, "post_build", frames, states)

  held_dof_idx = list(arm_dof_idx)
  if idle_gripper_kinematic_hold:
    held_dof_idx.extend(all_gripper_dof_idx or all_lite6_gripper_dof_idx or all_bio_gripper_dof_idx)
  _disable_robot_pd(robot, sorted(set(held_dof_idx)))
  _apply_hold(
    robot,
    arm_dof_idx,
    home,
    arm_kinematic_hold=arm_kinematic_hold,
    idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
    all_gripper_dof_idx=all_gripper_dof_idx,
    all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
    all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
  )
  gripper_kind = _setup_gripper_demo(
    robot,
    gripper_demo=args.gripper_demo,
    gripper_dof_idx=gripper_dof_idx,
    all_gripper_dof_idx=all_gripper_dof_idx,
    lite6_gripper_dof_idx=lite6_gripper_dof_idx,
    all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
    bio_gripper_dof_idx=bio_gripper_dof_idx,
    all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
  )
  _capture_phase(scene, cam, robot, profile, arm_dof_idx, out_dir, 1, "after_setup_hold", frames, states)

  for _ in range(3):
    _apply_hold(
      robot,
      arm_dof_idx,
      home,
      arm_kinematic_hold=arm_kinematic_hold,
      idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
    )
    scene.step(update_visualizer=False)
    _apply_hold(
      robot,
      arm_dof_idx,
      home,
      arm_kinematic_hold=arm_kinematic_hold,
      idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
    )
  _capture_phase(scene, cam, robot, profile, arm_dof_idx, out_dir, 2, "after_hidden_warmup", frames, states)

  _control_gripper_demo(
    robot,
    gripper_kind,
    0,
    gripper_dof_idx,
    all_gripper_dof_idx,
    lite6_gripper_dof_idx,
    all_lite6_gripper_dof_idx,
    bio_gripper_dof_idx,
    all_bio_gripper_dof_idx,
  )
  _apply_hold(
    robot,
    arm_dof_idx,
    home,
    arm_kinematic_hold=arm_kinematic_hold,
    idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
    all_gripper_dof_idx=all_gripper_dof_idx,
    all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
    all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
  )
  _capture_phase(scene, cam, robot, profile, arm_dof_idx, out_dir, 3, "visible_pre_step_hold", frames, states)

  scene.step(update_visualizer=True)
  _capture_phase(
    scene,
    cam,
    robot,
    profile,
    arm_dof_idx,
    out_dir,
    4,
    "old_visible_after_scene_step_before_rehold",
    frames,
    states,
  )
  _apply_hold(
    robot,
    arm_dof_idx,
    home,
    arm_kinematic_hold=arm_kinematic_hold,
    idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
    all_gripper_dof_idx=all_gripper_dof_idx,
    all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
    all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
  )
  _capture_phase(scene, cam, robot, profile, arm_dof_idx, out_dir, 5, "after_rehold", frames, states)

  _control_gripper_demo(
    robot,
    gripper_kind,
    1,
    gripper_dof_idx,
    all_gripper_dof_idx,
    lite6_gripper_dof_idx,
    all_lite6_gripper_dof_idx,
    bio_gripper_dof_idx,
    all_bio_gripper_dof_idx,
  )
  _kinematic_step(
    scene,
    robot,
    arm_kinematic_hold=arm_kinematic_hold,
    idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
    arm_dof_idx=arm_dof_idx,
    home=home,
    all_gripper_dof_idx=all_gripper_dof_idx,
    all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
    all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
  )
  _capture_phase(
    scene,
    cam,
    robot,
    profile,
    arm_dof_idx,
    out_dir,
    6,
    "actual_helper_step_after_rehold",
    frames,
    states,
  )

  max_visible_delta = {"step": -1, "max_link_norm_mm": 0.0, "max_abs_link_z_mm": 0.0, "delta": {}}
  for step in range(args.steps):
    _control_gripper_demo(
      robot,
      gripper_kind,
      step,
      gripper_dof_idx,
      all_gripper_dof_idx,
      lite6_gripper_dof_idx,
      all_lite6_gripper_dof_idx,
      bio_gripper_dof_idx,
      all_bio_gripper_dof_idx,
    )
    _apply_hold(
      robot,
      arm_dof_idx,
      home,
      arm_kinematic_hold=arm_kinematic_hold,
      idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
    )
    pre = _state(robot, profile, arm_dof_idx)
    scene.step(update_visualizer=True)
    visible = _state(robot, profile, arm_dof_idx)
    delta = _state_delta(pre, visible)
    if delta["max_link_norm_mm"] > max_visible_delta["max_link_norm_mm"]:
      max_visible_delta = {
        "step": step,
        "max_link_norm_mm": delta["max_link_norm_mm"],
        "max_abs_link_z_mm": delta["max_abs_link_z_mm"],
        "delta": delta,
      }
    _apply_hold(
      robot,
      arm_dof_idx,
      home,
      arm_kinematic_hold=arm_kinematic_hold,
      idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_dof_idx=all_bio_gripper_dof_idx,
    )

  report = {
    "robot": args.robot,
    "urdf": str(urdf_path),
    "gripper_kind": gripper_kind,
    "out_dir": str(out_dir),
    "state_deltas": {
      "post_build_to_after_setup_hold": _state_delta(states["post_build"], states["after_setup_hold"]),
      "after_warmup_to_visible_pre_step": _state_delta(
        states["after_hidden_warmup"], states["visible_pre_step_hold"]
      ),
      "visible_pre_step_to_old_visible_before_rehold": _state_delta(
        states["visible_pre_step_hold"], states["old_visible_after_scene_step_before_rehold"]
      ),
      "old_visible_before_rehold_to_after_rehold": _state_delta(
        states["old_visible_after_scene_step_before_rehold"], states["after_rehold"]
      ),
      "after_rehold_to_actual_helper_step": _state_delta(
        states["after_rehold"], states["actual_helper_step_after_rehold"]
      ),
    },
    "image_diffs": {
      "visible_pre_step_to_old_visible_before_rehold": _image_diff(
        frames["visible_pre_step_hold"],
        frames["old_visible_after_scene_step_before_rehold"],
      ),
      "old_visible_before_rehold_to_after_rehold": _image_diff(
        frames["old_visible_after_scene_step_before_rehold"],
        frames["after_rehold"],
      ),
      "after_rehold_to_actual_helper_step": _image_diff(
        frames["after_rehold"],
        frames["actual_helper_step_after_rehold"],
      ),
    },
    "max_old_visible_delta_first_second": max_visible_delta,
  }
  report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps(report, indent=2))
  print(f"keyframes: {out_dir}")
  print(f"report: {report_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
