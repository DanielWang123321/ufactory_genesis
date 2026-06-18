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

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.85
GRASP_CLOSE_STEPS = 50
GRASP_CLOSE_MIN_STEPS = 47
GRASP_SQUEEZE_STEPS = 80

FINGER_PAD_BELOW_FC = 0.061
FINGER_CLOSE_DESCENT = 0.015
GRASP_TABLE_CLEARANCE = 0.010

SIM_DT = 0.02
SETTLE_STEPS = 40


@dataclass
class ShowcaseRobotCtx:
  ik_link: object
  left_finger: object
  right_finger: object
  arm_dof_idx: list[int]
  gripper_dof_idx: list[int]
  down_quat: torch.Tensor
  home_pos: list[float]
  home_qpos_saved: torch.Tensor
  finger_z_offset: float = 0.0


def _scale_steps(steps: int, speed: float) -> int:
  return max(1, int(round(steps / max(0.25, speed))))


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
  all_gripper_joints = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
  )
  all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in all_gripper_joints]

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
  robot.set_dofs_kp(torch.tensor([2.0], device=gs.device, dtype=gs.tc_float), gripper_dof_idx)
  robot.set_dofs_kv(torch.tensor([5.0], device=gs.device, dtype=gs.tc_float), gripper_dof_idx)
  robot.set_dofs_force_range(
    torch.tensor([-1.0], device=gs.device, dtype=gs.tc_float),
    torch.tensor([1.0], device=gs.device, dtype=gs.tc_float),
    gripper_dof_idx,
  )
  n_grip = len(all_gripper_dof_idx)
  robot.set_dofs_damping(
    torch.full((n_grip,), 0.1, device=gs.device, dtype=gs.tc_float),
    all_gripper_dof_idx,
  )
  robot.set_dofs_frictionloss(
    torch.zeros(n_grip, device=gs.device, dtype=gs.tc_float),
    all_gripper_dof_idx,
  )

  down_quat = _world_down_quat()

  return ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, down_quat


def _init_home_qpos(robot, ik_link, arm_dof_idx, gripper_dof_idx, down_quat, home_pos):
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
  init_qpos[:, gripper_dof_idx[0]] = GRIPPER_OPEN
  robot.set_qpos(init_qpos)
  return init_qpos.clone()


def _measure_finger_offset(ik_link, left_finger, right_finger):
  link6_pos = ik_link.get_pos()[0]
  fc_pos = ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]
  return (link6_pos[2] - fc_pos[2]).item()


def init_showcase_robot(robot, layout, scene) -> ShowcaseRobotCtx:
  """Apply PD gains, set home qpos via IK, and prime finger geometry."""
  ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, down_quat = _setup_robot(
    robot, None
  )
  home_pos = _world_home(layout)
  home_qpos_saved = _init_home_qpos(
    robot, ik_link, arm_dof_idx, gripper_dof_idx, down_quat, home_pos
  )
  scene.step()
  finger_z_offset = _measure_finger_offset(ik_link, left_finger, right_finger)
  return ShowcaseRobotCtx(
    ik_link=ik_link,
    left_finger=left_finger,
    right_finger=right_finger,
    arm_dof_idx=arm_dof_idx,
    gripper_dof_idx=gripper_dof_idx,
    down_quat=down_quat,
    home_pos=home_pos,
    home_qpos_saved=home_qpos_saved,
    finger_z_offset=finger_z_offset,
  )


def hold_robot_home(robot, scene, ctx: ShowcaseRobotCtx, *, steps: int = 1) -> None:
  target_qpos = ctx.home_qpos_saved
  grip_t = torch.tensor([[GRIPPER_OPEN]], device=gs.device, dtype=gs.tc_float)
  for _ in range(steps):
    robot.control_dofs_position(target_qpos[:, ctx.arm_dof_idx], ctx.arm_dof_idx)
    robot.control_dofs_position(grip_t, ctx.gripper_dof_idx)
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
) -> ShowcaseRobotCtx:
  if ctx is None:
    ctx = init_showcase_robot(robot, layout, scene)

  ik_link = ctx.ik_link
  left_finger = ctx.left_finger
  right_finger = ctx.right_finger
  arm_dof_idx = ctx.arm_dof_idx
  gripper_dof_idx = ctx.gripper_dof_idx
  down_quat = ctx.down_quat
  home_pos = ctx.home_pos
  home_qpos_saved = ctx.home_qpos_saved

  finger_z_offset = ctx.finger_z_offset
  grasp_z = _grasp_link6_z(layout.table_top_z, finger_z_offset)
  pre_grasp_z = grasp_z + 0.10
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

  def move_to(target_link6_pos, gripper_val, steps=100, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    start_pos = ik_link.get_pos().clone()
    grip_t = torch.tensor([[gripper_val]], device=gs.device, dtype=gs.tc_float)
    n = _scale_steps(steps, speed)
    for s in range(n):
      alpha = (s + 1) / n
      interp = start_pos + alpha * (target_t - start_pos)
      qpos = robot.inverse_kinematics(
        link=ik_link, pos=interp, quat=down_quat, dofs_idx_local=arm_dof_idx,
      )
      robot.control_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx)
      robot.control_dofs_position(grip_t, gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)

  def hold(target_link6_pos, gripper_val, steps=50, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    grip_t = torch.tensor([[gripper_val]], device=gs.device, dtype=gs.tc_float)
    target_qpos = robot.inverse_kinematics(
      link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
    )
    n = _scale_steps(steps, speed)
    for _ in range(n):
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      robot.control_dofs_position(grip_t, gripper_dof_idx)
      _sim_step()
    if label:
      print_state(label)

  def grasp_close(target_link6_pos, steps=GRASP_CLOSE_STEPS, label=""):
    target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
    target_qpos = robot.inverse_kinematics(
      link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
    )
    n = max(GRASP_CLOSE_MIN_STEPS, _scale_steps(steps, speed))
    for s in range(n):
      alpha = (s + 1) / n
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      grip_val = GRIPPER_OPEN + alpha * (GRIPPER_CLOSE - GRIPPER_OPEN)
      robot.control_dofs_position(
        torch.tensor([[grip_val]], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
      )
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
      robot.control_dofs_position(interp[:, gripper_dof_idx], gripper_dof_idx)
      _sim_step()
    settle = _scale_steps(50, speed)
    for _ in range(settle):
      robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
      robot.control_dofs_position(target_qpos[:, gripper_dof_idx], gripper_dof_idx)
      _sim_step()

  print("\n[Phase 0] Idle")
  hold(home_pos, GRIPPER_OPEN, steps=40, label="Home")
  if stop_after_phase0:
    return ctx

  print("\n[Phase 1] Approach above block")
  move_to([obj_xy[0], obj_xy[1], pre_grasp_z], GRIPPER_OPEN, steps=100, label="Pre-grasp")

  print("\n[Phase 2] Descend to grasp height")
  move_to([obj_xy[0], obj_xy[1], grasp_z], GRIPPER_OPEN, steps=120, label="At block")

  print("\n[Phase 3] Settle before close")
  hold([obj_xy[0], obj_xy[1], grasp_z], GRIPPER_OPEN, steps=30, label="Settled")

  print("\n[Phase 4] Close gripper (physics grasp)")
  grasp_close([obj_xy[0], obj_xy[1], grasp_z], label="Grasped")

  print("\n[Phase 4b] Squeeze hold")
  hold([obj_xy[0], obj_xy[1], grasp_z], GRIPPER_CLOSE, steps=GRASP_SQUEEZE_STEPS, label="Squeezed")

  # Latch carry BEFORE lift so block follows fingers from the first upward motion
  _latch_carry()
  print("  Carry latched")

  print("\n[Phase 5] Lift to release height (100 mm above box)")
  move_to([obj_xy[0], obj_xy[1], release_z], GRIPPER_CLOSE, steps=160, label="Lifted")
  hold([obj_xy[0], obj_xy[1], release_z], GRIPPER_CLOSE, steps=20, label="Lift hold")

  print("\n[Phase 6] Transit to above box at release height")
  mid_xy = [obj_xy[0], (obj_xy[1] + place_xy[1]) / 2]
  move_to([mid_xy[0], mid_xy[1], release_z], GRIPPER_CLOSE, steps=220, label="Transit mid")
  move_to([place_xy[0], place_xy[1], release_z], GRIPPER_CLOSE, steps=220, label="Above box")

  print("\n[Phase 7] Hold above box before release")
  hold([place_xy[0], place_xy[1], release_z], GRIPPER_CLOSE, steps=30, label="Pre-release")

  print("\n[Phase 8] Release block above box")
  _release_carry()
  hold([place_xy[0], place_xy[1], release_z], GRIPPER_OPEN, steps=150, label="Released")

  print("\n[Phase 9] Brief hold after release")
  hold([place_xy[0], place_xy[1], release_z], GRIPPER_OPEN, steps=30, label="Post-release")

  print("\n[Phase 10] Return transit at release height")
  move_to([obj_xy[0], obj_xy[1], release_z], GRIPPER_OPEN, steps=180, label="Above block return")

  print("\n[Phase 11] Restore home")
  restore_home(steps=150)
  print_state("Home restored")

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
    table_top_z, sim_dt=SIM_DT, build_scene=False,
  )
  scene.build(n_envs=1)

  # Immediately set home pose so the first rendered frame isn't all-zeros
  _arm_joints_tmp = [robot.get_joint(f"joint{i+1}") for i in range(6)]
  _arm_dof_tmp = [j.dofs_idx_local[0] for j in _arm_joints_tmp]
  _grip_idx_tmp = robot.get_joint("drive_joint").dofs_idx_local[0]
  _ik_link_tmp = robot.get_link("link6")
  _home_pos_tmp = _world_home(layout)
  _down_quat_tmp = _world_down_quat()

  _init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
  _home_qpos = robot.inverse_kinematics(
    link=_ik_link_tmp, pos=torch.tensor([_home_pos_tmp], device=gs.device, dtype=gs.tc_float),
    quat=_down_quat_tmp, dofs_idx_local=_arm_dof_tmp, init_qpos=_init_qpos,
  )
  for _i, _idx in enumerate(_arm_dof_tmp):
    _init_qpos[:, _idx] = _home_qpos[0, _arm_dof_tmp[_i]]
  _init_qpos[:, _grip_idx_tmp] = GRIPPER_OPEN
  for _ in range(3):
    robot.set_qpos(_init_qpos)
    scene.step()

  # Tighten mimic joint equality constraints so gripper linkages stay rigid
  import numpy as np
  stiff_sol_params = np.array([0.01, 0.1, 0.0001, 0.001, 0.001, 0.5, 2.0])
  mimic_keywords = ("finger", "knuckle")
  for eq in robot.equalities:
    if any(kw in eq.name for kw in mimic_keywords):
      eq.set_sol_params(stiff_sol_params)
      print(f"  [mimic stiff] {eq.name}")

  print("xArm6 + Gripper G2 packaging showcase — Ctrl+C to exit")
  print(f"  table_top_z={layout.table_top_z:.2f}m  speed={args.speed}  loop={args.loop}")

  ctx = init_showcase_robot(robot, layout, scene)
  hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, args.speed))

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
        while True:
          scene.step()
          time.sleep(SIM_DT)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
