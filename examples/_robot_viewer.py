"""Shared Genesis GLB viewer helpers for multi-robot examples."""

from __future__ import annotations

import time

import numpy as np

import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.robot_registry import RobotModelSpec, joint_names

from _gripper_demo import (
  GRIPPER_OPEN,
  control_gripper_pose,
  gripper_demo_target,
  gripper_dof_indices,
  set_gripper_pose,
  setup_gripper_pd,
)


def setup_arm_pd(robot, dof_idx: list[int], dof: int) -> None:
  kp = [3000, 3000, 2000, 2000, 1000, 1000, 800][:dof]
  kv = [300, 300, 200, 200, 100, 100, 80][:dof]
  force_lo = [-50, -50, -32, -32, -32, -20, -15][:dof]
  force_hi = [50, 50, 32, 32, 32, 20, 15][:dof]
  robot.set_dofs_kp(np.array(kp), dof_idx)
  robot.set_dofs_kv(np.array(kv), dof_idx)
  robot.set_dofs_force_range(np.array(force_lo), np.array(force_hi), dof_idx)


def run_glb_viewer(
  profile: RobotModelSpec,
  urdf_path: str,
  *,
  headless: bool = False,
  pd_demo: bool = False,
  gripper_demo: bool = False,
) -> None:
  enable_glb_pbr_surfaces()
  gs.init(backend=gs.gpu)
  scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
      camera_pos=(1.5, -1.5, 1.5),
      camera_lookat=(0.0, 0.0, 0.4),
      camera_fov=40,
      max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=not headless,
  )
  scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
  robot = scene.add_entity(
    gs.morphs.URDF(
      file=urdf_path,
      pos=(0.0, 0.0, 0.0),
      fixed=True,
      requires_jac_and_IK=True,
    ),
    surface=glb_view_surface(),
  )
  scene.build()

  jnames = joint_names(profile)
  home = np.zeros(profile.dof)
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in jnames if n in joint_map]
  gripper_dof_idx, all_gripper_dof_idx = gripper_dof_indices(robot)

  if arm_dof_idx:
    setup_arm_pd(robot, arm_dof_idx, profile.dof)
    robot.set_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
    robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
  if gripper_demo and gripper_dof_idx:
    setup_gripper_pd(robot, gripper_dof_idx, all_gripper_dof_idx)
    set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)

  for _ in range(100):
    scene.step()

  if headless:
    return

  if gripper_demo:
    print(f"Viewer: {profile.key} — gripper open/close demo")
  else:
    print(f"Viewer: {profile.key} ({profile.dof} DOF). Close window or Ctrl+C to exit.")

  poses = _demo_poses(profile.dof)
  step = 0
  pose_idx = 0
  hold_steps = 300
  last_gripper_phase = -1
  while True:
    if pd_demo and step % hold_steps == 0:
      target = poses[pose_idx % len(poses)]
      if arm_dof_idx:
        robot.control_dofs_position(target[: len(arm_dof_idx)], arm_dof_idx)
      pose_idx += 1
    elif not gripper_demo and arm_dof_idx:
      robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
    if gripper_demo and gripper_dof_idx:
      grip_phase = (step // 200) % 2
      if grip_phase != last_gripper_phase:
        label = "closed" if grip_phase else "open"
        print(f"  Gripper target: {label}")
        last_gripper_phase = grip_phase
      control_gripper_pose(
        robot,
        gripper_dof_idx,
        all_gripper_dof_idx,
        gripper_demo_target(step),
      )
    scene.step()
    step += 1
    time.sleep(0.01)


def _demo_poses(dof: int) -> list[np.ndarray]:
  base = [
    np.zeros(dof),
    np.linspace(0.3, -0.3, dof),
    np.linspace(-0.2, 0.2, dof),
  ]
  if dof >= 6:
    base.append(np.array([0.5, -0.3, -0.1, 0.5, 0.3, 0.0] + [0.0] * (dof - 6)))
  return base
