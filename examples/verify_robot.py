"""Generic Genesis verification for supported UFACTORY arms."""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.kinematics import prepare_robot_model_for_verification
from ufactory.paths import robot_urdf
from ufactory.robot_registry import ROBOT_PROFILES, get_robot_profile, joint_names


def quat_to_rpy(quat):
  w, x, y, z = quat[0], quat[1], quat[2], quat[3]
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)
  sinp = 2.0 * (w * y - z * x)
  pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)
  return roll, pitch, yaw


def resolve_entity_name(entity, requested_name: str, kind: str) -> str:
  available = {item.name for item in entity.joints} if kind == "joint" else {item.name for item in entity.links}
  if requested_name in available:
    return requested_name
  fallback = requested_name.split("/")[-1]
  if fallback in available:
    return fallback
  raise KeyError(f"{kind} not found: {requested_name}")


def run_tests(profile_key: str, urdf_path: str, vis: bool) -> None:
  profile = get_robot_profile(profile_key)
  jnames = joint_names(profile)
  ee = profile.ee_link

  gs.init(backend=gs.gpu)
  scene = gs.Scene(show_viewer=vis, sim_options=gs.options.SimOptions(dt=0.01))
  robot = scene.add_entity(
    gs.morphs.URDF(file=urdf_path, fixed=True, requires_jac_and_IK=True),
  )
  scene.build()

  assert robot.n_dofs == profile.dof, f"Expected {profile.dof} DOF, got {robot.n_dofs}"
  print(f"PASS: loaded {profile.key} ({robot.n_dofs} DOF, {robot.n_links} links)")

  joint_map = {resolve_entity_name(robot, j.name, "joint"): j for j in robot.joints}
  arm_idx = [joint_map[n].dofs_idx_local[0] for n in jnames]
  q = np.linspace(-0.2, 0.2, profile.dof)
  robot.set_dofs_position(q, arm_idx)

  ee_link_name = resolve_entity_name(robot, ee, "link")
  ee_link = next(l for l in robot.links if resolve_entity_name(robot, l.name, "link") == ee_link_name)
  q_t = torch.tensor(q, dtype=torch.float32, device=gs.device)
  links_pos, _ = robot.forward_kinematics(qpos=q_t)
  idx = int(ee_link.idx_local)
  fk_pos = links_pos[idx].cpu().numpy() if links_pos.ndim == 2 else links_pos[0, idx].cpu().numpy()
  ee_pos = ee_link.get_pos()
  if hasattr(ee_pos, "cpu"):
    ee_pos = ee_pos.cpu().numpy()
  err_mm = float(np.linalg.norm(fk_pos - np.asarray(ee_pos).reshape(3)) * 1000)
  print(f"PASS: FK EE error {err_mm:.3f} mm")
  assert err_mm < 5.0, f"FK error too large: {err_mm} mm"

  target = q + 0.05
  robot.control_dofs_position(target, arm_idx)
  for _ in range(50):
    scene.step()
  print("PASS: PD stepping")


def main() -> None:
  parser = argparse.ArgumentParser(description="Verify robot URDF in Genesis")
  parser.add_argument("--robot", required=True, choices=sorted(ROBOT_PROFILES.keys()))
  parser.add_argument("--urdf", default=None)
  parser.add_argument("--kinematics-suffix", default=None)
  parser.add_argument("--kinematics-yaml", default=None)
  parser.add_argument("-v", "--vis", action="store_true")
  args = parser.parse_args()

  profile = get_robot_profile(args.robot)
  default_urdf = args.urdf or robot_urdf(args.robot)
  urdf_path, _ = prepare_robot_model_for_verification(
    default_urdf,
    args.kinematics_yaml,
    args.kinematics_suffix,
    robot_name=profile.robot_name,
    joint_count=profile.dof,
  )
  run_tests(args.robot, urdf_path, args.vis)
  print(f"\nAll checks passed for {args.robot}")


if __name__ == "__main__":
  main()
