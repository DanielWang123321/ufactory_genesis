"""FK verification: Genesis URDF vs xArm Python SDK (simulation mode)."""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.kinematics import (
  get_robot_sn,
  log_kinematics_sn_status,
  prepare_robot_model_for_verification,
  validate_kinematics_calibration_request,
)
from ufactory.paths import robot_urdf
from ufactory.robot_registry import get_robot_profile, joint_names

PASS_POS_MM = 1.0
PASS_RPY_DEG = 0.5


def quat_to_rpy(quat):
  w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)
  sinp = 2.0 * (w * y - z * x)
  pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)
  return roll, pitch, yaw


def angle_diff_deg(a: float, b: float) -> float:
  diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
  return abs(diff) * 180.0 / math.pi


def test_configs(dof: int):
  configs = [("home", np.zeros(dof))]
  if dof >= 5:
    configs.append(("A", np.array([0.5, -0.3, 0.0, 0.0, 0.3] + [0.0] * (dof - 5))[:dof]))
  if dof >= 6:
    configs.append(("B", np.array([0.0, -0.5, -0.1, 0.5, 0.5, 0.0] + [0.0] * (dof - 6))[:dof]))
  return configs


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--robot", required=True)
  parser.add_argument("--ip", required=True)
  parser.add_argument("--kinematics-suffix", default=None)
  parser.add_argument("--kinematics-yaml", default=None)
  args = parser.parse_args()

  profile = get_robot_profile(args.robot)

  from xarm.wrapper import XArmAPI

  arm = XArmAPI(args.ip, is_radian=True)
  arm.connect()
  sn = get_robot_sn(arm)
  validate_kinematics_calibration_request(
    sn,
    profile.robot_name,
    kinematics_yaml=args.kinematics_yaml,
    kinematics_suffix=args.kinematics_suffix,
  )
  log_kinematics_sn_status(
    sn,
    profile.robot_name,
    kinematics_yaml=args.kinematics_yaml,
    kinematics_suffix=args.kinematics_suffix,
  )
  arm.motion_enable(enable=True)
  arm.set_mode(0)
  arm.set_state(0)

  urdf_path, _ = prepare_robot_model_for_verification(
    robot_urdf(args.robot),
    args.kinematics_yaml,
    args.kinematics_suffix,
    robot_name=profile.robot_name,
    joint_count=profile.dof,
  )

  gs.init(backend=gs.cpu)
  scene = gs.Scene(show_viewer=False)
  robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True, requires_jac_and_IK=True))
  scene.build()

  jnames = joint_names(profile)
  joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
  ee_link = next(l for l in robot.links if l.name.split("/")[-1] == profile.ee_link)

  failed = 0
  for name, q in test_configs(profile.dof):
    code, pose = arm.get_forward_kinematics(q.tolist(), input_is_radian=True, return_is_radian=True)
    if code != 0:
      raise RuntimeError(f"SDK FK failed for {name}: code={code}")
    sdk_pos = np.array(pose[:3])
    sdk_rpy = np.array(pose[3:6])

    q_t = torch.tensor(q, dtype=torch.float32, device=gs.device)
    links_pos, links_quat = robot.forward_kinematics(qpos=q_t)
    idx = int(ee_link.idx_local)
    g_pos = links_pos[idx].cpu().numpy() if links_pos.ndim == 2 else links_pos[0, idx].cpu().numpy()
    g_quat = links_quat[idx].cpu().numpy() if links_quat.ndim == 2 else links_quat[0, idx].cpu().numpy()
    g_rpy = np.array(quat_to_rpy(g_quat))

    pos_mm = float(np.linalg.norm((g_pos - sdk_pos) * 1000))
    rpy_deg = max(angle_diff_deg(a, b) for a, b in zip(g_rpy, sdk_rpy))
    ok = pos_mm < PASS_POS_MM and rpy_deg < PASS_RPY_DEG
    status = "PASS" if ok else "FAIL"
    print(f"{status} {name}: pos={pos_mm:.2f}mm rpy={rpy_deg:.2f}deg")
    if not ok:
      failed += 1

  arm.disconnect()
  if failed:
    raise SystemExit(1)
  print("All FK checks passed")


if __name__ == "__main__":
  main()
