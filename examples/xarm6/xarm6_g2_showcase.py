"""
xArm 6 + Gripper G2 packaging showcase — physics pick-and-place.

Solid yellow table, xArm6 on the table long edge, physics grasp of a red block, place into an open cardboard shipping box.

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    python scripts/generate_showcase_textures.py   # first time only
    python examples/xarm6/xarm6_g2_showcase.py
    python examples/xarm6/xarm6_g2_showcase.py --no-loop --speed 1.5
    python examples/xarm6/xarm6_g2_showcase.py --capture-keyframes
    python scripts/capture_showcase_keyframes.py   # headless keyframes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
import genesis as gs
from genesis.utils.geom import transform_by_quat, transform_quat_by_quat, xyz_to_quat
from scipy.spatial.transform import Rotation as R

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_ROOT))

from _packaging_scene import (
  HOME_RPY_DEG,
  HOME_XY,
  HOME_Z,
  ROBOT_BASE_YAW_DEG,
  box_top_z,
  build_packaging_scene,
  make_layout,
)
from _robot_viewer import start_deferred_viewer

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.85
GRIPPER_OPEN_GAP_M = 0.084  # drive=0 → ~84 mm two-finger gap
GRIPPER_GAP_CALIBRATION_OFFSET_M = 0.0053  # linear model under-closes vs G2 pad kinematics
GRASP_SQUEEZE_GAP_MARGIN = 0.0  # flush with block width; calibration offset handles pad error
GRIPPER_SPEED_FACTOR = 5.0  # open/close 5× faster than original step counts
GRIPPER_OPEN_STEPS = 150
GRASP_CLOSE_STEPS = 50
GRASP_CLOSE_MIN_STEPS = 47
GRASP_SQUEEZE_STEPS = 80

FINGER_PAD_BELOW_FC = 0.061
FINGER_CLOSE_DESCENT = 0.015
GRASP_TABLE_CLEARANCE = 0.010

SIM_DT = 0.02
SETTLE_STEPS = 40
SHOWCASE_CARTESIAN_SPEED_MMS = 100.0
SHOWCASE_CARTESIAN_ACCEL_MMS2 = 1000.0
TRANSIT_JOINT_BLEND_STEPS = 1  # kinematic cruise: 1 step per planned waypoint

ALL_GRIPPER_JOINTS = (
  "drive_joint",
  "left_finger_joint",
  "left_inner_knuckle_joint",
  "right_outer_knuckle_joint",
  "right_finger_joint",
  "right_inner_knuckle_joint",
)


@dataclass
class ShowcaseRobotCtx:
  ik_link: object
  left_finger: object
  right_finger: object
  arm_dof_idx: list[int]
  gripper_dof_idx: list[int]
  all_gripper_dof_idx: list[int]
  down_quat: torch.Tensor
  home_pos: list[float]
  home_qpos_saved: torch.Tensor
  finger_z_offset: float = 0.0
  grasp_drive: float = GRIPPER_CLOSE


def gripper_drive_for_gap(gap_m: float) -> float:
  """Map desired two-finger gap (m) to drive_joint command."""
  gap_m = max(0.0, min(GRIPPER_OPEN_GAP_M, gap_m))
  return GRIPPER_CLOSE * (1.0 - gap_m / GRIPPER_OPEN_GAP_M)


def grasp_gripper_drive(obj_size: tuple[float, float, float]) -> float:
  """Partial close target for a block grasped with gripper pointing down (Y axis)."""
  obj_width = obj_size[1]
  target_gap = max(0.0, obj_width + GRASP_SQUEEZE_GAP_MARGIN - GRIPPER_GAP_CALIBRATION_OFFSET_M)
  return gripper_drive_for_gap(target_gap)


def stiffen_gripper_mimic_constraints(robot) -> None:
  """Tighten mimic joint equality constraints so gripper linkages stay rigid."""
  import numpy as np

  stiff_sol_params = np.array([0.01, 0.1, 0.0001, 0.001, 0.001, 0.5, 2.0])
  mimic_keywords = ("finger", "knuckle")
  for eq in robot.equalities:
    if any(kw in eq.name for kw in mimic_keywords):
      eq.set_sol_params(stiff_sol_params)


def collect_gripper_snapshot(robot) -> dict:
  """All gripper joint q/qd plus knuckle link poses — linkage whip diagnostics."""
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  joints: dict[str, dict[str, float]] = {}
  drive_q: float | None = None
  max_qd = 0.0
  max_mimic_err = 0.0
  for name in ALL_GRIPPER_JOINTS:
    joint = joint_map.get(name)
    if joint is None:
      continue
    idx = joint.dofs_idx_local[0]
    q = robot.get_dofs_position()[0, idx].item()
    qd = robot.get_dofs_velocity()[0, idx].item()
    joints[name] = {"q": q, "qd": qd}
    max_qd = max(max_qd, abs(qd))
    if name == "drive_joint":
      drive_q = q
  if drive_q is not None:
    for name, state in joints.items():
      if name != "drive_joint":
        max_mimic_err = max(max_mimic_err, abs(state["q"] - drive_q))
  knuckles: dict[str, list[float]] = {}
  for link_name in (
    "left_outer_knuckle",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_inner_knuckle",
  ):
    pos = robot.get_link(link_name).get_pos()[0].cpu().tolist()
    knuckles[link_name] = pos
  return {
    "joints": joints,
    "max_qd": max_qd,
    "max_mimic_err": max_mimic_err,
    "knuckles": knuckles,
  }


def collect_pose_snapshot(robot, layout) -> dict:
  """Return link6 / finger poses and arm qpos for keyframe metadata."""
  ik_link = robot.get_link("link6")
  left_finger = robot.get_link("left_finger")
  right_finger = robot.get_link("right_finger")
  arm_dof_idx = [robot.get_joint(f"joint{i + 1}").dofs_idx_local[0] for i in range(6)]
  gripper_dof_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]

  link6 = ik_link.get_pos()[0].cpu().tolist()
  left = left_finger.get_pos()[0].cpu().tolist()
  right = right_finger.get_pos()[0].cpu().tolist()
  finger_center_z = (left[2] + right[2]) / 2
  finger_y_gap_mm = abs(left[1] - right[1]) * 1000.0
  arm_q = robot.get_dofs_position()[0, arm_dof_idx].cpu().tolist()
  grip_q = robot.get_dofs_position()[0, gripper_dof_idx[0]].item()
  table_z = layout.table_top_z

  return {
    "link6_pos": link6,
    "left_finger_pos": left,
    "right_finger_pos": right,
    "finger_center_z": finger_center_z,
    "finger_y_gap_mm": finger_y_gap_mm,
    "table_top_z": table_z,
    "link6_above_table_mm": (link6[2] - table_z) * 1000.0,
    "finger_above_table_mm": (finger_center_z - table_z) * 1000.0,
    "arm_qpos_deg": [math.degrees(q) for q in arm_q],
    "gripper_q": grip_q,
    "gripper_detail": collect_gripper_snapshot(robot),
  }


def _scale_steps(steps: int, speed: float) -> int:
  return max(1, int(round(steps / max(0.25, speed))))


def _scaled_cartesian_kinematics(speed_mult: float) -> tuple[float, float]:
  mult = max(0.25, speed_mult)
  return SHOWCASE_CARTESIAN_SPEED_MMS * mult, SHOWCASE_CARTESIAN_ACCEL_MMS2 * mult


def _trapezoid_duration(dist_mm: float, speed_mms: float, accel_mms2: float) -> float:
  """Total move time (s) for symmetric trapezoid/triangle velocity profile."""
  if dist_mm <= 0:
    return 0.0
  t_a = speed_mms / accel_mms2
  d_a = 0.5 * accel_mms2 * t_a * t_a
  if dist_mm >= 2.0 * d_a:
    return 2.0 * t_a + (dist_mm - 2.0 * d_a) / speed_mms
  return 2.0 * math.sqrt(dist_mm / accel_mms2)


def _trapezoid_arclength(t: float, dist_mm: float, speed_mms: float, accel_mms2: float) -> float:
  """Arc length (mm) at time t along a symmetric trapezoid/triangle profile."""
  if dist_mm <= 0:
    return 0.0
  t = max(0.0, t)
  t_a = speed_mms / accel_mms2
  d_a = 0.5 * accel_mms2 * t_a * t_a
  if dist_mm >= 2.0 * d_a:
    t_cruise = (dist_mm - 2.0 * d_a) / speed_mms
    t_total = 2.0 * t_a + t_cruise
    t = min(t, t_total)
    if t <= t_a:
      return 0.5 * accel_mms2 * t * t
    if t <= t_a + t_cruise:
      return d_a + speed_mms * (t - t_a)
    t_d = t - t_a - t_cruise
    return dist_mm - 0.5 * accel_mms2 * (t_a - t_d) ** 2
  t_peak = math.sqrt(dist_mm / accel_mms2)
  t_total = 2.0 * t_peak
  t = min(t, t_total)
  if t <= t_peak:
    return 0.5 * accel_mms2 * t * t
  t_d = t - t_peak
  return dist_mm - 0.5 * accel_mms2 * t_d * t_d


def _trapezoid_step_count(
  dist_mm: float, speed_mult: float, dt: float = SIM_DT,
) -> int:
  if dist_mm <= 0:
    return 1
  speed_mms, accel_mms2 = _scaled_cartesian_kinematics(speed_mult)
  duration = _trapezoid_duration(dist_mm, speed_mms, accel_mms2)
  return max(1, int(math.ceil(duration / dt)))


def _trapezoid_alphas(
  dist_mm: float, speed_mult: float, dt: float = SIM_DT,
) -> list[float]:
  """Normalized path fractions [0, 1] at each sim step for trapezoid motion."""
  if dist_mm <= 0:
    return [1.0]
  speed_mms, accel_mms2 = _scaled_cartesian_kinematics(speed_mult)
  n = _trapezoid_step_count(dist_mm, speed_mult, dt)
  alphas: list[float] = []
  for i in range(1, n + 1):
    s = _trapezoid_arclength(i * dt, dist_mm, speed_mms, accel_mms2)
    alphas.append(min(1.0, s / dist_mm))
  return alphas


def _control_gripper(
  robot,
  grip_val: float,
  all_gripper_dof_idx: list[int],
) -> None:
  """Drive all gripper DOFs to the same angle (movable visual demo pattern)."""
  target = torch.full((1, len(all_gripper_dof_idx)), grip_val, device=gs.device, dtype=gs.tc_float)
  robot.control_dofs_position(target, all_gripper_dof_idx)


def _scale_gripper_steps(steps: int, speed: float) -> int:
  return max(1, int(round(steps / (GRIPPER_SPEED_FACTOR * max(0.25, speed)))))


def _scale_grasp_close_steps(steps: int, speed: float) -> int:
  """Grasp close: arm-speed only (skip GRIPPER_SPEED_FACTOR to avoid linkage whip)."""
  return max(40, int(round(steps / max(0.25, speed))))


def _world_home(layout) -> list[float]:
  rx, ry = layout.robot_xy
  yaw = math.radians(ROBOT_BASE_YAW_DEG)
  c, s = math.cos(yaw), math.sin(yaw)
  wx = HOME_XY[0] * c - HOME_XY[1] * s
  wy = HOME_XY[0] * s + HOME_XY[1] * c
  return [rx + wx, ry + wy, layout.table_top_z + HOME_Z]


def _world_down_quat() -> torch.Tensor:
  """TCP RPY in base frame → world quat (accounts for ROBOT_BASE_YAW_DEG)."""
  base_yaw_quat = xyz_to_quat(
    torch.tensor([[0.0, 0.0, ROBOT_BASE_YAW_DEG]], device=gs.device, dtype=gs.tc_float),
    rpy=True,
    degrees=True,
  )
  tcp_base_quat = xyz_to_quat(
    torch.tensor([list(HOME_RPY_DEG)], device=gs.device, dtype=gs.tc_float),
    rpy=True,
    degrees=True,
  )
  return transform_quat_by_quat(tcp_base_quat, base_yaw_quat)


def _setup_robot(robot, scene):
  ik_link = robot.get_link("link6")
  left_finger = robot.get_link("left_finger")
  right_finger = robot.get_link("right_finger")

  arm_dof_idx = [robot.get_joint(f"joint{i + 1}").dofs_idx_local[0] for i in range(6)]
  gripper_dof_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]
  all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in ALL_GRIPPER_JOINTS]

  robot.set_dofs_kp(
    torch.tensor([3000, 3000, 2000, 2000, 1000, 1000], device=gs.device, dtype=gs.tc_float),
    arm_dof_idx,
  )
  robot.set_dofs_kv(
    torch.tensor([300, 300, 200, 200, 100, 100], device=gs.device, dtype=gs.tc_float),
    arm_dof_idx,
  )
  robot.set_dofs_force_range(
    torch.tensor([-50, -50, -32, -32, -32, -20], device=gs.device, dtype=gs.tc_float),
    torch.tensor([50, 50, 32, 32, 32, 20], device=gs.device, dtype=gs.tc_float),
    arm_dof_idx,
  )
  robot.set_dofs_kp(torch.tensor([30.0], device=gs.device, dtype=gs.tc_float), gripper_dof_idx)
  robot.set_dofs_kv(torch.tensor([6.0], device=gs.device, dtype=gs.tc_float), gripper_dof_idx)
  robot.set_dofs_force_range(
    torch.tensor([-20.0], device=gs.device, dtype=gs.tc_float),
    torch.tensor([20.0], device=gs.device, dtype=gs.tc_float),
    gripper_dof_idx,
  )
  n_grip = len(all_gripper_dof_idx)
  robot.set_dofs_damping(
    torch.full((n_grip,), 0.05, device=gs.device, dtype=gs.tc_float),
    all_gripper_dof_idx,
  )
  robot.set_dofs_frictionloss(
    torch.zeros(n_grip, device=gs.device, dtype=gs.tc_float),
    all_gripper_dof_idx,
  )

  down_quat = _world_down_quat()

  return ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, all_gripper_dof_idx, down_quat


def _init_home_qpos(robot, ik_link, arm_dof_idx, all_gripper_dof_idx, down_quat, home_pos):
  home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
  init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
  home_qpos_result = robot.inverse_kinematics(
    link=ik_link,
    pos=home_link6_pos,
    quat=down_quat,
    dofs_idx_local=arm_dof_idx,
    init_qpos=init_qpos,
  )
  for i, idx in enumerate(arm_dof_idx):
    init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
  for idx in all_gripper_dof_idx:
    init_qpos[:, idx] = GRIPPER_OPEN
  robot.set_qpos(init_qpos)
  return init_qpos.clone()


def _measure_finger_offset(ik_link, left_finger, right_finger):
  link6_pos = ik_link.get_pos()[0]
  fc_pos = ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]
  return (link6_pos[2] - fc_pos[2]).item()


def init_showcase_robot(robot, layout, scene) -> ShowcaseRobotCtx:
  """Apply PD gains, set home qpos via IK, and prime finger geometry."""
  ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, all_gripper_dof_idx, down_quat = _setup_robot(
    robot, None
  )
  home_pos = _world_home(layout)
  home_qpos_saved = _init_home_qpos(
    robot, ik_link, arm_dof_idx, all_gripper_dof_idx, down_quat, home_pos
  )
  scene.step()
  finger_z_offset = _measure_finger_offset(ik_link, left_finger, right_finger)
  grasp_drive = grasp_gripper_drive(layout.obj_size)
  return ShowcaseRobotCtx(
    ik_link=ik_link,
    left_finger=left_finger,
    right_finger=right_finger,
    arm_dof_idx=arm_dof_idx,
    gripper_dof_idx=gripper_dof_idx,
    all_gripper_dof_idx=all_gripper_dof_idx,
    down_quat=down_quat,
    home_pos=home_pos,
    home_qpos_saved=home_qpos_saved,
    finger_z_offset=finger_z_offset,
    grasp_drive=grasp_drive,
  )


def hold_robot_home(robot, scene, ctx: ShowcaseRobotCtx, *, steps: int = 1) -> None:
  target_qpos = ctx.home_qpos_saved
  for _ in range(steps):
    robot.control_dofs_position(target_qpos[:, ctx.arm_dof_idx], ctx.arm_dof_idx)
    _control_gripper(robot, GRIPPER_OPEN, ctx.all_gripper_dof_idx)
    scene.step()


def _grasp_link6_z(surface_z: float, finger_z_offset: float) -> float:
  return (
    surface_z
    + GRASP_TABLE_CLEARANCE
    + FINGER_CLOSE_DESCENT
    + FINGER_PAD_BELOW_FC
    + finger_z_offset
  )


def _reset_block(block, layout) -> None:
  half_z = layout.obj_size[2] / 2
  pos = torch.tensor(
    [[layout.obj_spawn_xy[0], layout.obj_spawn_xy[1], layout.table_top_z + half_z]],
    device=gs.device,
    dtype=gs.tc_float,
  )
  block.set_pos(pos, zero_velocity=True)


def run_pick_place_cycle(
  scene,
  robot,
  block,
  layout,
  *,
  speed: float = 1.0,
  ctx: ShowcaseRobotCtx | None = None,
  stop_after_phase0: bool = False,
  capture_hook: callable | None = None,
) -> ShowcaseRobotCtx:
  if ctx is None:
    ctx = init_showcase_robot(robot, layout, scene)

  ik_link = ctx.ik_link
  left_finger = ctx.left_finger
  right_finger = ctx.right_finger
  arm_dof_idx = ctx.arm_dof_idx
  gripper_dof_idx = ctx.gripper_dof_idx
  all_gripper_dof_idx = ctx.all_gripper_dof_idx
  down_quat = ctx.down_quat
  home_pos = ctx.home_pos
  home_qpos_saved = ctx.home_qpos_saved

  finger_z_offset = ctx.finger_z_offset
  grasp_drive = grasp_gripper_drive(layout.obj_size)
  ctx.grasp_drive = grasp_drive
  grasp_z = _grasp_link6_z(layout.table_top_z, finger_z_offset)
  btop = box_top_z(layout)
  release_z = btop + 0.10 + finger_z_offset  # finger center 100 mm above box top rim
  transfer_lift_z = release_z  # transit and release at same height

  obj_xy = list(layout.obj_spawn_xy)
  place_xy = list(layout.place_xy)

  def finger_center():
    return ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]

  carry_active = False
  carry_offset = torch.zeros(3, device=gs.device, dtype=gs.tc_float)
  carry_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device, dtype=gs.tc_float)

  _debug_last_t = 0.0
  _debug_interval = 0.2  # 5 Hz

  _home_rpy_base = R.from_euler("xyz", list(HOME_RPY_DEG), degrees=True)

  def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    # Genesis quats are scalar-first [w, x, y, z]; inverse = conjugate for unit quats.
    return torch.stack([q[0], -q[1], -q[2], -q[3]], dim=-1)

  def _gs_to_scipy_quat(q: torch.Tensor) -> list[float]:
    """[w, x, y, z] (Genesis) -> [x, y, z, w] (scipy)."""
    qn = q.cpu().numpy()
    return [qn[1], qn[2], qn[3], qn[0]]

  def _robot_base_pose() -> tuple[torch.Tensor, torch.Tensor]:
    base_pos = robot.get_pos()[0]
    if base_pos.ndim == 2:
      base_pos = base_pos[0]
    base_quat = robot.get_quat()[0]
    if base_quat.ndim == 2:
      base_quat = base_quat[0]
    return base_pos, base_quat

  def _world_to_base_pos(world_pos: torch.Tensor) -> torch.Tensor:
    base_pos, base_quat = _robot_base_pose()
    return transform_by_quat(world_pos - base_pos, _quat_inv(base_quat))

  def _base_to_world_pos(base_rel: torch.Tensor) -> torch.Tensor:
    base_pos, base_quat = _robot_base_pose()
    return base_pos + transform_by_quat(base_rel, base_quat)

  def _ee_base_pose() -> tuple[torch.Tensor, R]:
    """Return (pos_mm, rot) of link6 expressed in the robot base frame."""
    base_pos = robot.get_pos()[0]
    if base_pos.ndim == 2:
      base_pos = base_pos[0]
    base_quat = robot.get_quat()[0]
    if base_quat.ndim == 2:
      base_quat = base_quat[0]
    l6_pos = ik_link.get_pos()[0]
    l6_quat = ik_link.get_quat()[0]
    base_quat_inv = _quat_inv(base_quat)
    rel_pos = transform_by_quat(l6_pos - base_pos, base_quat_inv)
    rel_quat = transform_quat_by_quat(l6_quat, base_quat_inv)
    return rel_pos * 1000.0, R.from_quat(_gs_to_scipy_quat(rel_quat))

  def _sync_carried_block() -> None:
    if not carry_active:
      return
    fc = finger_center()
    block.set_pos((fc + carry_offset).unsqueeze(0), zero_velocity=True)
    block.set_quat(carry_quat.unsqueeze(0), zero_velocity=True)

  def _sim_step() -> None:
    nonlocal _debug_last_t
    scene.step()
    _sync_carried_block()
    now = time.time()
    if now - _debug_last_t >= _debug_interval:
      _debug_last_t = now
      pos_mm, rot = _ee_base_pose()
      err_rot = rot.inv() * _home_rpy_base
      err_q = err_rot.as_quat()
      if err_q[3] < 0.0:
        err_q = -err_q
      err_short = R.from_quat(err_q)
      rv = err_short.as_rotvec(degrees=True)
      dpos = pos_mm - torch.tensor(
        [HOME_XY[0] * 1000, HOME_XY[1] * 1000, HOME_Z * 1000],
        device=gs.device, dtype=gs.tc_float,
      )
      print(
        f"  [EE/base] pos=[{pos_mm[0]:+7.1f}, {pos_mm[1]:+7.1f}, {pos_mm[2]:+7.1f}] mm  "
        f"aa=[{rv[0]:+6.1f}, {rv[1]:+6.1f}, {rv[2]:+6.1f}]°  "
        f"Δpos=[{dpos[0]:+5.1f}, {dpos[1]:+5.1f}, {dpos[2]:+5.1f}] mm  "
        f"|Δrot|={torch.tensor(rv).norm().item():.1f}°"
      )

  def _latch_carry() -> None:
    nonlocal carry_active, carry_offset, carry_quat
    fc = finger_center()
    carry_offset = block.get_pos()[0] - fc
    carry_quat = block.get_quat()[0].clone()
    carry_active = True

  def _release_carry() -> None:
    nonlocal carry_active
    carry_active = False

  def print_state(label: str) -> None:
    fc = finger_center()
    grip_val = robot.get_dofs_position(gripper_dof_idx)[0].item()
    obj_pos = block.get_pos()[0]
    print(
      f"  [{label:15s}] FC: [{fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f}]  "
      f"Grip: {grip_val:.3f}  "
      f"Block: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]"
    )

  def move_to(target_link6_pos, gripper_val, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    start_pos = ik_link.get_pos().clone()
    dist_mm = torch.norm(target_t - start_pos).item() * 1000.0
    for alpha in _trapezoid_alphas(dist_mm, speed):
      interp = start_pos + alpha * (target_t - start_pos)
      qpos = robot.inverse_kinematics(
        link=ik_link, pos=interp, quat=down_quat, dofs_idx_local=arm_dof_idx,
      )
      robot.control_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx)
      _control_gripper(robot, gripper_val, all_gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)
      _maybe_capture(label)

  def move_xy(xy, gripper_val, label=""):
    """Move link6 in XY only; keep current Z."""
    start_pos = ik_link.get_pos().clone()
    target_t = torch.tensor(
      [[xy[0], xy[1], start_pos[0, 2].item()]],
      device=gs.device,
      dtype=gs.tc_float,
    )
    delta = target_t - start_pos
    dist_mm = torch.norm(delta[:, :2]).item() * 1000.0
    for alpha in _trapezoid_alphas(dist_mm, speed):
      interp = start_pos + alpha * delta
      qpos = robot.inverse_kinematics(
        link=ik_link, pos=interp, quat=down_quat, dofs_idx_local=arm_dof_idx,
      )
      robot.control_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx)
      _control_gripper(robot, gripper_val, all_gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)
      _maybe_capture(label)

  def _ik_qpos_at_base_pose(
    base_x: float, base_y: float, base_z_m: float, init_qpos,
  ):
    base_target = torch.tensor(
      [base_x, base_y, base_z_m], device=gs.device, dtype=gs.tc_float,
    )
    world_target = _base_to_world_pos(base_target).unsqueeze(0)
    result = robot.inverse_kinematics(
      link=ik_link,
      pos=world_target,
      quat=down_quat,
      dofs_idx_local=arm_dof_idx,
      init_qpos=init_qpos,
      return_error=True,
      max_solver_iters=20,
    )
    if result is None:
      return robot.inverse_kinematics(
        link=ik_link,
        pos=world_target,
        quat=down_quat,
        dofs_idx_local=arm_dof_idx,
        init_qpos=init_qpos,
      )
    qpos, _ = result
    return qpos

  def _refine_base_z_at(world_xy, base_z_m: float, gripper_val, max_iters: int = 5) -> float:
    """Endpoint-only Z correction (kinematic, no PD snap)."""
    base_xy = _world_to_base_pos(
      torch.tensor([world_xy[0], world_xy[1], 0.0], device=gs.device, dtype=gs.tc_float),
    )[:2]
    init_qpos = robot.get_dofs_position()
    z_cmd = base_z_m
    residual = 0.0
    for _ in range(max_iters):
      qpos = _ik_qpos_at_base_pose(base_xy[0].item(), base_xy[1].item(), z_cmd, init_qpos)
      _move_arm_kinematic(qpos, gripper_val)
      _sim_step()
      init_qpos = robot.get_dofs_position()
      actual_z = _ee_base_pose()[0][2].item() / 1000.0
      residual = base_z_m - actual_z
      if abs(residual) < 0.002:
        return residual
      z_cmd += residual
    return residual

  def _set_arm_kinematic(qpos, gripper_val) -> None:
    """Kinematic arm pose; gripper follows caller (PD or kinematic)."""
    robot.set_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx, zero_velocity=True)
    grip_target = torch.full(
      (1, len(all_gripper_dof_idx)), gripper_val, device=gs.device, dtype=gs.tc_float,
    )
    robot.set_dofs_position(grip_target, all_gripper_dof_idx, zero_velocity=True)

  def _move_arm_kinematic(interp_q, gripper_val) -> None:
    """Kinematic arm/gripper pose — avoids PD overshoot jitter during scripted transit."""
    _set_arm_kinematic(interp_q, gripper_val)

  def move_xy_at_base_z(
    world_xy, base_z_m: float, gripper_val, label="", refine_endpoint: bool = False,
  ):
    """Move link6 in XY at constant base-frame Z; kinematic trapezoid cruise."""
    start_base_mm, _ = _ee_base_pose()
    start_base = start_base_mm / 1000.0
    dest_base_xy = _world_to_base_pos(
      torch.tensor([world_xy[0], world_xy[1], 0.0], device=gs.device, dtype=gs.tc_float),
    )[:2]
    dx = dest_base_xy[0].item() - start_base[0].item()
    dy = dest_base_xy[1].item() - start_base[1].item()
    dist_mm = math.hypot(dx, dy) * 1000.0

    prev_q = robot.get_dofs_position()
    for alpha in _trapezoid_alphas(dist_mm, speed):
      bx = start_base[0].item() + alpha * dx
      by = start_base[1].item() + alpha * dy
      q_tgt = _ik_qpos_at_base_pose(bx, by, base_z_m, prev_q)
      blend = TRANSIT_JOINT_BLEND_STEPS
      q0 = prev_q
      for m in range(blend):
        alpha_b = (m + 1) / blend
        interp_q = q0 + alpha_b * (q_tgt - q0)
        _move_arm_kinematic(interp_q, gripper_val)
        _sim_step()
      prev_q = q_tgt

    if refine_endpoint:
      _refine_base_z_at(world_xy, base_z_m, gripper_val)
    if label:
      print_state(label)
      if capture_hook is not None:
        q_snap = robot.get_dofs_position()
        capture_hook(label)
        _move_arm_kinematic(q_snap, gripper_val)

  def _hold_kinematic_at_base_z(
    world_xy, base_z_m: float, gripper_val, steps: int,
  ) -> None:
    """Kinematic hold at fixed base-frame pose — avoids PD sag under payload."""
    _refine_base_z_at(world_xy, base_z_m, gripper_val, max_iters=5)
    base_xy = _world_to_base_pos(
      torch.tensor([world_xy[0], world_xy[1], 0.0], device=gs.device, dtype=gs.tc_float),
    )[:2]
    q_hold = _ik_qpos_at_base_pose(
      base_xy[0].item(), base_xy[1].item(), base_z_m, robot.get_dofs_position(),
    )
    for _ in range(steps):
      _move_arm_kinematic(q_hold, gripper_val)
      _sim_step()

  def hold_at_base_z(world_xy, base_z_m: float, gripper_val, steps=50, label=""):
    n = _scale_steps(steps, speed)
    _hold_kinematic_at_base_z(world_xy, base_z_m, gripper_val, n)
    if label:
      print_state(label)
      _maybe_capture(label)

  def open_gripper_at_base_z(world_xy, base_z_m: float, open_from: float, steps=GRIPPER_OPEN_STEPS, label=""):
    _refine_base_z_at(world_xy, base_z_m, open_from, max_iters=5)
    base_xy = _world_to_base_pos(
      torch.tensor([world_xy[0], world_xy[1], 0.0], device=gs.device, dtype=gs.tc_float),
    )[:2]
    q_hold = _ik_qpos_at_base_pose(
      base_xy[0].item(), base_xy[1].item(), base_z_m, robot.get_dofs_position(),
    )
    n = _scale_gripper_steps(steps, speed)
    for s in range(n):
      alpha = (s + 1) / n
      grip_val = open_from + alpha * (GRIPPER_OPEN - open_from)
      _set_arm_kinematic(q_hold, grip_val)
      _control_gripper(robot, grip_val, all_gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)

  def move_z(z, xy, gripper_val, label=""):
    """Move link6 in Z only; XY fixed at xy."""
    start_pos = ik_link.get_pos().clone()
    target_t = torch.tensor([[xy[0], xy[1], z]], device=gs.device, dtype=gs.tc_float)
    dist_mm = abs(target_t[0, 2].item() - start_pos[0, 2].item()) * 1000.0
    delta = target_t - start_pos
    for alpha in _trapezoid_alphas(dist_mm, speed):
      interp = start_pos + alpha * delta
      qpos = robot.inverse_kinematics(
        link=ik_link, pos=interp, quat=down_quat, dofs_idx_local=arm_dof_idx,
      )
      robot.control_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx)
      _control_gripper(robot, gripper_val, all_gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)
      _maybe_capture(label)

  def hold(target_link6_pos, gripper_val, steps=50, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    target_qpos = robot.inverse_kinematics(
      link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
    )
    n = _scale_steps(steps, speed)
    for _ in range(n):
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      _control_gripper(robot, gripper_val, all_gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)
      _maybe_capture(label)

  def _maybe_capture(tag: str) -> None:
    if capture_hook is not None:
      capture_hook(tag)

  def grasp_close(target_link6_pos, close_target: float, steps=GRASP_CLOSE_STEPS, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    target_qpos = robot.inverse_kinematics(
      link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
    )
    n = max(
      _scale_grasp_close_steps(GRASP_CLOSE_MIN_STEPS, speed),
      _scale_grasp_close_steps(steps, speed),
    )
    sample_idxs = {0, n // 4, n // 2, 3 * n // 4, n - 1}
    for s in range(n):
      alpha = (s + 1) / n
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      grip_val = GRIPPER_OPEN + alpha * (close_target - GRIPPER_OPEN)
      _control_gripper(robot, grip_val, all_gripper_dof_idx)
      _sim_step()
      if s in sample_idxs:
        _maybe_capture(f"grasp_close_step_{s}")
    if label:
      print_state(label)
      _maybe_capture(label)

  def grasp_open(target_link6_pos, open_from: float, steps=GRIPPER_OPEN_STEPS, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    target_qpos = robot.inverse_kinematics(
      link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
    )
    n = _scale_gripper_steps(steps, speed)
    for s in range(n):
      alpha = (s + 1) / n
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      grip_val = open_from + alpha * (GRIPPER_OPEN - open_from)
      _control_gripper(robot, grip_val, all_gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)

  def restore_home(steps=150):
    start_qpos = robot.get_dofs_position().clone()
    target_qpos = home_qpos_saved.clone()
    n = _scale_steps(steps, speed)
    for s in range(n):
      alpha = (s + 1) / n
      interp = start_qpos + alpha * (target_qpos - start_qpos)
      robot.control_dofs_position(interp[:, arm_dof_idx], arm_dof_idx)
      robot.control_dofs_position(interp[:, all_gripper_dof_idx], all_gripper_dof_idx)
      _sim_step()
    settle = _scale_steps(50, speed)
    for _ in range(settle):
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      robot.control_dofs_position(target_qpos[:, all_gripper_dof_idx], all_gripper_dof_idx)
      _sim_step()

  print("\n[Phase 0] Idle")
  hold(home_pos, GRIPPER_OPEN, steps=40, label="Home")
  if stop_after_phase0:
    return ctx

  print("\n[Phase 1] Transit XY above block")
  move_xy(obj_xy, GRIPPER_OPEN, label="Above block XY")

  print("\n[Phase 2] Descend to grasp height")
  move_z(grasp_z, obj_xy, GRIPPER_OPEN, label="At block")

  print("\n[Phase 3] Settle before close")
  hold([obj_xy[0], obj_xy[1], grasp_z], GRIPPER_OPEN, steps=30, label="Settled")

  print("\n[Phase 4] Close gripper (physics grasp)")
  target_gap_mm = max(0.0, (layout.obj_size[1] + GRASP_SQUEEZE_GAP_MARGIN - GRIPPER_GAP_CALIBRATION_OFFSET_M) * 1000.0)
  print(
    f"  Grasp drive: {grasp_drive:.3f} "
    f"(target gap {target_gap_mm:.1f} mm for {layout.obj_size[1] * 1000:.0f} mm block)"
  )
  grasp_close([obj_xy[0], obj_xy[1], grasp_z], grasp_drive, label="Grasped")

  print("\n[Phase 4b] Squeeze hold")
  hold([obj_xy[0], obj_xy[1], grasp_z], grasp_drive, steps=GRASP_SQUEEZE_STEPS, label="Squeezed")

  # Latch carry BEFORE lift so block follows fingers from the first upward motion
  _latch_carry()
  print("  Carry latched")

  print("\n[Phase 5] Lift to release height (100 mm above box)")
  move_z(release_z, obj_xy, grasp_drive, label="Lifted")
  hold([obj_xy[0], obj_xy[1], release_z], grasp_drive, steps=20, label="Lift hold")
  cruise_base_z_m = _ee_base_pose()[0][2].item() / 1000.0

  print("\n[Phase 6] Transit to above box at release height")
  move_xy_at_base_z(
    place_xy, cruise_base_z_m, grasp_drive, label="Above box",
    refine_endpoint=True,
  )

  print("\n[Phase 7] Hold above box before release")
  hold_at_base_z(place_xy, cruise_base_z_m, grasp_drive, steps=30, label="Pre-release")

  print("\n[Phase 8] Release block above box")
  _release_carry()
  open_gripper_at_base_z(place_xy, cruise_base_z_m, grasp_drive, label="Released")

  print("\n[Phase 9] Brief hold after release")
  hold_at_base_z(place_xy, cruise_base_z_m, GRIPPER_OPEN, steps=30, label="Post-release")

  print("\n[Phase 10] Return transit at release height")
  move_xy_at_base_z(obj_xy, cruise_base_z_m, GRIPPER_OPEN, label="Above block return")

  print("\n[Phase 11] Transit XY above home at cruise height")
  move_xy_at_base_z([home_pos[0], home_pos[1]], cruise_base_z_m, GRIPPER_OPEN, label="Above home XY")

  print("\n[Phase 12] Descend to home")
  move_z(home_pos[2], [home_pos[0], home_pos[1]], GRIPPER_OPEN, label="Home Z")

  final_obj = block.get_pos()[0]
  # Block drops from 100 mm above box; expected to land on box inner floor
  target = torch.tensor(
    [place_xy[0], place_xy[1], layout.box_inner_floor_z + layout.obj_size[2] / 2],
    device=gs.device,
    dtype=gs.tc_float,
  )
  err_mm = torch.norm(final_obj - target).item() * 1000
  print(f"\n  Place error: {err_mm:.1f} mm  {'OK' if err_mm < 60 else 'CHECK'}")
  return ctx


def _capture_keyframes_interactive(scene, robot, block, layout, speed: float, ctx: ShowcaseRobotCtx) -> None:
  """Capture startup keyframes using the interactive viewer camera."""
  out_dir = EXAMPLES_ROOT.parent / "debug" / "showcase_keyframes"
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"\nCapturing keyframes to {out_dir} (viewer open; inspect then Ctrl+C)")

  hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, speed))
  run_pick_place_cycle(scene, robot, block, layout, speed=speed, ctx=ctx, stop_after_phase0=True)
  print(f"\nKeyframe workflow complete. For headless PNGs run:\n"
        f"  python scripts/capture_showcase_keyframes.py")


def main() -> None:
  parser = argparse.ArgumentParser(description="xArm6 + Gripper G2 physics packaging showcase")
  parser.add_argument(
    "--table-height",
    type=float,
    default=None,
    help="Tabletop surface height in meters (default: 0.75)",
  )
  parser.add_argument("--speed", type=float, default=1.0, help="Motion speed multiplier")
  parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Loop pick-place cycle (default: on)",
  )
  parser.add_argument(
    "--capture-keyframes",
    action="store_true",
    help="Run startup pose debug sequence then exit (use headless script for PNGs)",
  )
  args = parser.parse_args()

  from ufactory.glb_visual import enable_glb_pbr_surfaces

  enable_glb_pbr_surfaces()
  gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)

  table_top_z = args.table_height if args.table_height is not None else make_layout().table_top_z
  scene, robot, block, layout = build_packaging_scene(
    table_top_z, sim_dt=SIM_DT, build_scene=False, show_viewer=False,
  )
  scene.build(n_envs=1)

  stiffen_gripper_mimic_constraints(robot)

  print("xArm6 + Gripper G2 packaging showcase — Ctrl+C to exit")
  print(f"  table_top_z={layout.table_top_z:.2f}m  speed={args.speed}  loop={args.loop}")

  ctx = init_showcase_robot(robot, layout, scene)
  hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, args.speed))

  start_deferred_viewer(scene)

  if args.capture_keyframes:
    _reset_block(block, layout)
    _capture_keyframes_interactive(scene, robot, block, layout, args.speed, ctx)
    return

  try:
    while True:
      _reset_block(block, layout)
      hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, args.speed))
      run_pick_place_cycle(scene, robot, block, layout, speed=args.speed, ctx=ctx)
      if not args.loop:
        print("\nSingle cycle complete (--no-loop). Viewer stays open.")
        try:
          while True:
            scene.step()
            time.sleep(SIM_DT)
        except gs.GenesisException as exc:
          if "Viewer closed" not in str(exc):
            raise
        return
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
