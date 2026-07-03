#!/usr/bin/env python3
"""Extract per-robot kinematic calibration from xArm control box.

Vendored from xarm_ros2 (rolling branch). Output YAML matches xarm_ros2 format:
  kinematics.joint1..N: {x, y, z, roll, pitch, yaw}  (meters, radians)

SN eligibility (no compensation file expected):
  - xArm 5/6/7: SN positions 3-6 < 1304
  - Lite6: SN positions 3-6 < 1006
  - UF850: all units have compensation

Usage:
    python scripts/gen_kinematics_params.py <robot-ip> [suffix]
    python scripts/gen_kinematics_params.py <robot-ip> [suffix] --force

When suffix is omitted, defaults to the last 6 characters of the robot SN.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ufactory.kinematics.calibration import (  # noqa: E402
  get_robot_sn,
  has_per_unit_kinematics_calibration,
  kinematics_suffix_from_sn,
  log_kinematics_sn_status,
  parse_sn_model_code,
  robot_name_from_firmware,
)

try:
  from yaml import dump
except ImportError:

  def dump(data, f, indent=0, **kwargs):
    buf = []
    for key, val in data.items():
      if isinstance(val, dict):
        buf.append("{}{}:".format(" " * indent, key))
        buf += dump(val, None, indent=indent + 2, **kwargs)
      else:
        buf.append("{}{}: {}".format(" " * indent, key, val))
    if f is not None:
      f.write("\n".join(buf))
    return buf


IS_PY3 = sys.version_info.major >= 3


def _output_dir_for_robot(robot_name: str) -> Path:
  return _REPO_ROOT / "assets" / "urdf" / robot_name / "kinematics" / "user"


def _fetch_kinematics_bytes(robot_ip: str) -> bytes:
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.settimeout(10.0)
  sock.connect((robot_ip, 502))
  send_data = [0x00, 0x01, 0x00, 0x02, 0x00, 0x01, 0x08]
  if IS_PY3:
    sock.send(bytes(send_data))
  else:
    sock.send("".join(map(chr, send_data)))
  recv_data = sock.recv(179)
  sock.close()
  return recv_data


def main() -> int:
  parser = argparse.ArgumentParser(description="Extract per-unit kinematics YAML from control box")
  parser.add_argument("robot_ip", help="Robot IP address")
  parser.add_argument(
    "kinematics_suffix",
    nargs="?",
    default=None,
    help="Suffix for output YAML filename (default: last 6 characters of SN)",
  )
  parser.add_argument("output_dir", nargs="?", default=None, help="Optional output directory")
  parser.add_argument(
    "--force",
    action="store_true",
    help="Export even when SN indicates no factory compensation (e.g. POE calibration in firmware)",
  )
  args = parser.parse_args()

  from xarm.wrapper import XArmAPI

  arm = XArmAPI(args.robot_ip, is_radian=True)
  if not arm.connected:
    print(f"[Failed] cannot connect to {args.robot_ip}")
    return 1
  sn = get_robot_sn(arm)
  arm.disconnect()

  if not args.kinematics_suffix:
    args.kinematics_suffix = kinematics_suffix_from_sn(sn)
    print(f"kinematics_suffix: {args.kinematics_suffix} (auto from SN {sn})")

  recv_data = _fetch_kinematics_bytes(args.robot_ip)
  if not (len(recv_data) == 179 and recv_data[8]):
    valid = 0 if len(recv_data) < 9 else recv_data[8]
    print("[Failed] recv_len={}, valid={}".format(len(recv_data), valid))
    return 1

  robot_dof = recv_data[9] if IS_PY3 else ord(recv_data[9])
  robot_type = recv_data[10] if IS_PY3 else ord(recv_data[10])
  robot_name = robot_name_from_firmware(robot_dof, robot_type)

  print(f"robot_name     : {robot_name}")
  log_kinematics_sn_status(
    sn,
    robot_name,
    kinematics_suffix=args.kinematics_suffix,
    allow_sn_override=args.force,
  )

  if not args.force and not has_per_unit_kinematics_calibration(sn, robot_name):
    model_code = parse_sn_model_code(sn)
    print(
      "[Skipped] SN model code {} indicates no per-unit kinematics compensation. "
      "Use nominal URDF without --kinematics-suffix. Pass --force to export anyway.".format(
        model_code
      )
    )
    return 2

  output_dir = (
    Path(args.output_dir).resolve()
    if args.output_dir
    else _output_dir_for_robot(robot_name)
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file = output_dir / "{}_kinematics_{}.yaml".format(robot_name, args.kinematics_suffix)

  params = struct.unpack("<42f", recv_data[11:])
  kinematics = {}
  data = {"kinematics": kinematics}
  for i in range(robot_dof):
    joint_param = {}
    kinematics["joint{}".format(i + 1)] = joint_param
    for axis_idx, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
      value = float(params[i * 6 + axis_idx])
      # xArm5 firmware may return mm-scale outliers on prismatic-like offsets.
      if axis in ("x", "y", "z") and abs(value) > 1.0:
        value /= 1000.0
      joint_param[axis] = value

  with open(output_file, "w", encoding="utf-8") as f:
    try:
      dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except TypeError:
      dump(data, f, default_flow_style=False, allow_unicode=True)

  print("[Success] save to {}".format(output_file))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
