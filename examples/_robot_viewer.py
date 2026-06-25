"""Shared Genesis GLB viewer helpers for multi-robot examples."""

from __future__ import annotations

import sys
import time

import numpy as np

import genesis as gs
from ufactory.bio_gripper_g2 import BioGripperG2
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.robot_registry import RobotModelSpec, joint_names

from _bio_gripper_g2_demo import (
  BIO_GRIPPER_G2_OPEN,
  bio_gripper_g2_demo_target,
)
from _gripper_demo import (
  GRIPPER_OPEN,
  control_gripper_pose,
  gripper_demo_target,
  gripper_dof_indices,
  set_gripper_pose,
  setup_gripper_pd,
)
from _lite6_gripper_demo import (
  LITE6_GRIPPER_OPEN,
  control_lite6_gripper_pose,
  lite6_gripper_demo_target,
  lite6_gripper_dof_indices,
  set_lite6_gripper_pose,
  setup_lite6_gripper_pd,
)

TCP_MARKER_RADIUS = 0.008


def resolve_robot_link(robot, name: str):
  available = {link.name.split("/")[-1]: link for link in robot.links}
  if name not in available:
    raise KeyError(f"Link not found: {name}. Available: {sorted(available)}")
  return available[name]


def add_tcp_marker(scene):
  """Red sphere at DH TCP (EE flange); visual only, no collision."""
  return scene.add_entity(
    gs.morphs.Sphere(
      radius=TCP_MARKER_RADIUS,
      fixed=True,
      collision=False,
    ),
    surface=gs.surfaces.Rough(
      diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
    ),
  )


def update_tcp_marker(marker, ee_link) -> None:
  marker.set_pos(ee_link.get_pos())


def start_deferred_viewer(scene) -> None:
  """Open the interactive viewer after the scene has been initialized and warmed up."""
  visualizer = scene.visualizer
  if visualizer.viewer is not None:
    return

  try:
    from genesis.vis.viewer import Viewer
    from genesis.vis.visualizer import VIEWER_DEFAULT_ASPECT_RATIO, VIEWER_DEFAULT_HEIGHT_RATIO
  except Exception as exc:
    gs.raise_exception_from("Rendering not working on this machine.", exc)

  live_other_scenes = [
    scene_ref() for scene_ref in gs._scene_registry if scene_ref() is not None and scene_ref() is not scene
  ]
  if live_other_scenes:
    gs.raise_exception(
      "Interactive viewer not supported when managing multiple scenes. Please set `show_viewer=False` "
      "or call `del scene`."
    )

  viewer_options = scene.viewer_options
  if viewer_options.res is None:
    try:
      screen_height, _screen_width, screen_scale = gs.utils.try_get_display_size()
    except Exception as exc:
      gs.raise_exception_from("No display detected. Use `show_viewer=False` for headless mode.", exc)
    viewer_height = (screen_height * screen_scale) * VIEWER_DEFAULT_HEIGHT_RATIO
    viewer_width = viewer_height / VIEWER_DEFAULT_ASPECT_RATIO
    viewer_options.res = (int(viewer_width), int(viewer_height))
  if viewer_options.run_in_thread is None:
    if sys.platform == "linux":
      viewer_options.run_in_thread = True
    elif sys.platform == "darwin":
      viewer_options.run_in_thread = False
    elif sys.platform == "win32":
      viewer_options.run_in_thread = True
  if sys.platform == "darwin" and viewer_options.run_in_thread:
    gs.raise_exception("Running viewer in background thread is not supported on MacOS.")

  viewer = Viewer(viewer_options, visualizer._context)
  visualizer._viewer = viewer
  if getattr(visualizer, "_rasterizer", None) is not None:
    visualizer._rasterizer._viewer = viewer
    visualizer._rasterizer._offscreen = False
  viewer.build(scene)
  visualizer.viewer_lock = viewer.lock
  visualizer.reset()


def setup_arm_pd(robot, dof_idx: list[int], dof: int) -> None:
  kp = [3000, 3000, 2000, 2000, 1000, 1000, 800][:dof]
  kv = [300, 300, 200, 200, 100, 100, 80][:dof]
  force_lo = [-50, -50, -32, -32, -32, -20, -15][:dof]
  force_hi = [50, 50, 32, 32, 32, 20, 15][:dof]
  robot.set_dofs_kp(np.array(kp), dof_idx)
  robot.set_dofs_kv(np.array(kv), dof_idx)
  robot.set_dofs_force_range(np.array(force_lo), np.array(force_hi), dof_idx)


def _set_gripper_kinematic(robot, all_gripper_dof_idx: list[int], value: float) -> None:
  if not all_gripper_dof_idx:
    return
  robot.set_dofs_position(
    np.full(len(all_gripper_dof_idx), value),
    all_gripper_dof_idx,
    zero_velocity=True,
  )


def _disable_robot_pd(robot, dof_idx: list[int]) -> None:
  """Zero PD gains so scene.step() does not apply position control forces."""
  if not dof_idx:
    return
  n = len(dof_idx)
  zeros = np.zeros(n)
  robot.set_dofs_kp(zeros, dof_idx)
  robot.set_dofs_kv(zeros, dof_idx)
  robot.set_dofs_force_range(zeros, zeros, dof_idx)


def _apply_kinematic_hold(
  robot,
  arm_dof_idx: list[int],
  arm_q: np.ndarray,
  *,
  hold_arm: bool,
  hold_gripper: bool,
  all_gripper_dof_idx: list[int],
  all_lite6_gripper_dof_idx: list[int],
  all_bio_gripper_g2_dof_idx: list[int],
  bio_gripper: BioGripperG2 | None = None,
) -> None:
  """Visual-only hold: teleport joints after physics step, no PD."""
  if hold_arm and arm_dof_idx:
    robot.set_dofs_position(arm_q[: len(arm_dof_idx)], arm_dof_idx, zero_velocity=True)
  if not hold_gripper:
    return
  if all_gripper_dof_idx:
    _set_gripper_kinematic(robot, all_gripper_dof_idx, GRIPPER_OPEN)
  elif all_lite6_gripper_dof_idx:
    _set_gripper_kinematic(robot, all_lite6_gripper_dof_idx, LITE6_GRIPPER_OPEN)
  elif all_bio_gripper_g2_dof_idx and bio_gripper is not None:
    # Bio Gripper G2 mirrors the left finger (-1), so hold the open pose through the
    # controller rather than writing the same value to both finger DOFs.
    bio_gripper.set_pose(BIO_GRIPPER_G2_OPEN)


def _kinematic_step(
  scene,
  robot,
  *,
  arm_kinematic_hold: bool,
  idle_gripper_kinematic_hold: bool,
  arm_dof_idx: list[int],
  home: np.ndarray,
  all_gripper_dof_idx: list[int],
  all_lite6_gripper_dof_idx: list[int],
  all_bio_gripper_g2_dof_idx: list[int],
  bio_gripper: BioGripperG2 | None = None,
  arm_target: np.ndarray | None = None,
) -> None:
  """Advance sim one step; held DOFs are teleported immediately before and after."""
  hold_needed = arm_kinematic_hold or idle_gripper_kinematic_hold
  if hold_needed:
    _apply_kinematic_hold(
      robot,
      arm_dof_idx,
      home if arm_target is None else arm_target,
      hold_arm=arm_kinematic_hold,
      hold_gripper=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
      bio_gripper=bio_gripper,
    )
    scene.step(update_visualizer=False)
    _apply_kinematic_hold(
      robot,
      arm_dof_idx,
      home if arm_target is None else arm_target,
      hold_arm=arm_kinematic_hold,
      hold_gripper=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
      bio_gripper=bio_gripper,
    )
    visualizer = getattr(scene, "visualizer", None)
    if getattr(visualizer, "viewer", None) is not None:
      visualizer.update(force=False)
  else:
    scene.step()


# Match xArm-Python-SDK 2001-move_joint.py default speed=50 (deg/s).
_ARM_DEMO_SPEED_DEG_S = 50.0
_ARM_DEMO_DT = 0.01


def _step_joint_toward(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
  delta = goal - current
  max_abs = float(np.max(np.abs(delta)))
  if max_abs <= max_step:
    return goal.copy()
  return current + delta * (max_step / max_abs)


class _ArmJointDemo:
  """Smooth joint-space waypoint playback (kinematic), not stiff PD tracking."""

  def __init__(self, poses: list[np.ndarray], n_dof: int) -> None:
    self._poses = poses
    self._n = n_dof
    self._idx = 0
    self._max_step = np.radians(_ARM_DEMO_SPEED_DEG_S) * _ARM_DEMO_DT
    self.current = poses[0][:n_dof].copy()
    self.goal = poses[0][:n_dof].copy()
    self.finished = len(poses) <= 1

  def step(self) -> tuple[np.ndarray, np.ndarray, bool]:
    if self.finished:
      return self.current.copy(), self.current.copy(), True
    prev = self.current.copy()
    self.current = _step_joint_toward(self.current, self.goal, self._max_step)
    if np.allclose(self.current, self.goal, atol=1e-4):
      if self._idx >= len(self._poses) - 1:
        self.finished = True
        return prev, self.current.copy(), True
      self._idx += 1
      self.goal = self._poses[self._idx][: self._n].copy()
    return prev, self.current.copy(), False


def run_glb_viewer(
  profile: RobotModelSpec,
  urdf_path: str,
  *,
  headless: bool = False,
  pd_demo: bool = False,
  gripper_demo: bool = False,
  show_tcp: bool = False,
) -> None:
  enable_glb_pbr_surfaces()
  gs.init(backend=gs.gpu)
  scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
      camera_pos=(1.5, -1.5, 1.5),
      camera_lookat=(0.0, 0.0, 0.4),
      camera_fov=40,
      refresh_rate=60,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
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
  tcp_marker = None
  if show_tcp and not headless:
    tcp_marker = add_tcp_marker(scene)
  scene.build()

  jnames = joint_names(profile)
  home = np.zeros(profile.dof)
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in jnames if n in joint_map]
  gripper_dof_idx, all_gripper_dof_idx = gripper_dof_indices(robot)
  lite6_gripper_dof_idx, all_lite6_gripper_dof_idx = lite6_gripper_dof_indices(robot)
  # Use BioGripperG2 class for discovery and control (preferred API).
  bio_gripper = BioGripperG2(robot)
  bio_gripper_g2_dof_idx = bio_gripper.drive_dof_idx
  all_bio_gripper_g2_dof_idx = bio_gripper.all_dof_idx
  arm_kinematic_hold = not pd_demo
  idle_gripper_kinematic_hold = not gripper_demo

  ee_link = None
  if tcp_marker is not None:
    ee_link = resolve_robot_link(robot, profile.ee_link)
    print(f"TCP marker: {profile.ee_link} (DH flange, no tool)")

  held_dof_idx: list[int] = []
  if arm_kinematic_hold:
    held_dof_idx.extend(arm_dof_idx)
  if idle_gripper_kinematic_hold:
    held_dof_idx.extend(all_gripper_dof_idx or all_lite6_gripper_dof_idx or all_bio_gripper_g2_dof_idx)
  if held_dof_idx:
    _disable_robot_pd(robot, sorted(set(held_dof_idx)))
  if arm_kinematic_hold or idle_gripper_kinematic_hold:
    _apply_kinematic_hold(
      robot,
      arm_dof_idx,
      home,
      hold_arm=arm_kinematic_hold,
      hold_gripper=idle_gripper_kinematic_hold,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
      bio_gripper=bio_gripper,
    )
  if pd_demo and arm_dof_idx:
    setup_arm_pd(robot, arm_dof_idx, profile.dof)
    robot.set_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
    robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
  if gripper_demo and gripper_dof_idx:
    setup_gripper_pd(robot, gripper_dof_idx, all_gripper_dof_idx)
    set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)
  elif gripper_demo and lite6_gripper_dof_idx:
    setup_lite6_gripper_pd(robot, lite6_gripper_dof_idx, all_lite6_gripper_dof_idx)
    set_lite6_gripper_pose(
      robot,
      lite6_gripper_dof_idx,
      all_lite6_gripper_dof_idx,
      LITE6_GRIPPER_OPEN,
    )
  elif gripper_demo and bio_gripper_g2_dof_idx:
    bio_gripper.setup_pd()
    bio_gripper.set_pose(BIO_GRIPPER_G2_OPEN)

  warmup_steps = 3 if arm_kinematic_hold else 100
  for _ in range(warmup_steps):
    if pd_demo and arm_dof_idx:
      robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
    if tcp_marker is not None and ee_link is not None:
      update_tcp_marker(tcp_marker, ee_link)
    _kinematic_step(
      scene,
      robot,
      arm_kinematic_hold=arm_kinematic_hold,
      idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
      arm_dof_idx=arm_dof_idx,
      home=home,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
      bio_gripper=bio_gripper,
    )

  if headless:
    return
  start_deferred_viewer(scene)

  if gripper_demo:
    print(f"Viewer: {profile.key} — gripper open/close demo")
  elif pd_demo:
    print(f"Viewer: {profile.key} — joint motion demo ({_ARM_DEMO_SPEED_DEG_S:.0f} deg/s, once)")
  else:
    print(f"Viewer: {profile.key} ({profile.dof} DOF). Close window or Ctrl+C to exit.")

  joint_demo = _ArmJointDemo(_demo_poses(profile), len(arm_dof_idx)) if pd_demo and arm_dof_idx else None
  demo_done_announced = False
  step = 0
  last_gripper_phase = -1
  while True:
    arm_target = None
    if joint_demo is not None and not joint_demo.finished:
      _, q, _ = joint_demo.step()
      robot.set_dofs_position(q, arm_dof_idx, zero_velocity=True)
      arm_target = q
    elif joint_demo is not None:
      if not demo_done_announced:
        print("  Joint motion demo complete.")
        demo_done_announced = True
      robot.set_dofs_position(joint_demo.current, arm_dof_idx, zero_velocity=True)
      arm_target = joint_demo.current
    elif pd_demo and arm_dof_idx:
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
    elif gripper_demo and lite6_gripper_dof_idx:
      grip_phase = (step // 200) % 2
      if grip_phase != last_gripper_phase:
        label = "closed" if grip_phase else "open"
        print(f"  Lite6 gripper target: {label}")
        last_gripper_phase = grip_phase
      control_lite6_gripper_pose(
        robot,
        lite6_gripper_dof_idx,
        all_lite6_gripper_dof_idx,
        lite6_gripper_demo_target(step),
      )
    elif gripper_demo and bio_gripper_g2_dof_idx:
      grip_phase = (step // 200) % 2
      if grip_phase != last_gripper_phase:
        label = "closed" if grip_phase else "open"
        print(f"  Bio Gripper G2 target: {label}")
        last_gripper_phase = grip_phase
      bio_gripper.control_pose(bio_gripper_g2_demo_target(step))
    if tcp_marker is not None and ee_link is not None:
      update_tcp_marker(tcp_marker, ee_link)
    _kinematic_step(
      scene,
      robot,
      arm_kinematic_hold=arm_kinematic_hold,
      idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
      arm_dof_idx=arm_dof_idx,
      home=home,
      all_gripper_dof_idx=all_gripper_dof_idx,
      all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
      all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
      bio_gripper=bio_gripper,
      arm_target=arm_target,
    )
    step += 1
    time.sleep(0.01)


def _lite6_demo_poses_deg() -> list[list[float]]:
  """Lite6 PD preview: J1 cycles 0→-90→0°, then J3 cycles 0→90→0° (other joints at 0)."""
  home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  return [
    home,
    [-90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    home,
    [0.0, 0.0, 90.0, 0.0, 0.0, 0.0],
    home,
  ]


def _sdk_move_joint_poses_deg(dof: int) -> list[list[float]]:
  """Joint targets from xArm-Python-SDK example/wrapper/*/2001-move_joint.py (degrees)."""
  if dof == 5:
    return [
      [90, 0, 0, 0, 0],
      [90, 0, -60, 0, 0],
      [90, -30, -60, 0, 0],
      [0, -30, -60, 0, 0],
      [0, 0, -60, 0, 0],
      [0, 0, 0, 0, 0],
    ]
  if dof == 7:
    return [
      [90, 0, 0, 0, 0, 0, 0],
      [90, -60, 0, 0, 0, 0, 0],
      [90, -60, -30, 0, 0, 0, 0],
      [0, -60, -30, 0, 0, 0, 0],
      [0, -60, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ]
  return [
    [90, 0, 0, 0, 0, 0],
    [90, 0, -60, 0, 0, 0],
    [90, -30, -60, 0, 0, 0],
    [0, -30, -60, 0, 0, 0],
    [0, 0, -60, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
  ]


def _demo_poses(profile: RobotModelSpec) -> list[np.ndarray]:
  """PD preview joint targets (radians). Lite6 uses J1/J3 sweep; others use SDK 2001-move_joint."""
  if profile.key == "lite6":
    rows = _lite6_demo_poses_deg()
  else:
    rows = _sdk_move_joint_poses_deg(profile.dof)
  return [np.radians(row[: profile.dof], dtype=float) for row in rows]
