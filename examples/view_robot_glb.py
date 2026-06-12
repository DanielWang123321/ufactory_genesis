"""View any supported robot GLB visual model in Genesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_ROOT))

from _robot_viewer import run_glb_viewer
from ufactory.paths import robot_visual_glb_urdf
from ufactory.robot_registry import ROBOT_PROFILES, get_robot_profile


def main() -> None:
  parser = argparse.ArgumentParser(description="View robot GLB visual model")
  parser.add_argument("--robot", required=True, choices=sorted(ROBOT_PROFILES.keys()))
  parser.add_argument(
    "--bio-gripper-g2",
    action="store_true",
    help="Load Bio Gripper G2 static visual combo URDF",
  )
  parser.add_argument(
    "--gripper-g2",
    action="store_true",
    help="Load Gripper G2 visual combo URDF",
  )
  parser.add_argument(
    "--movable",
    action="store_true",
    help="Gripper G2 per-link GLBs (required for open/close animation)",
  )
  parser.add_argument(
    "--gripper-demo",
    action="store_true",
    help="Cycle drive_joint open/close (requires --gripper-g2 --movable)",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--pd", action="store_true", help="Joint PD motion demo")
  args = parser.parse_args()

  profile = get_robot_profile(args.robot)
  if args.bio_gripper_g2 and args.gripper_g2:
    parser.error("--bio-gripper-g2 and --gripper-g2 are mutually exclusive")
  if args.movable and not args.gripper_g2:
    parser.error("--movable requires --gripper-g2")
  if args.gripper_demo and not args.movable:
    parser.error("--gripper-demo requires --movable")
  if args.bio_gripper_g2 and not profile.supports_bio_gripper_g2:
    parser.error(f"{args.robot} does not support Bio Gripper G2")
  if args.gripper_g2 and not profile.supports_gripper_g2:
    parser.error(f"{args.robot} does not support Gripper G2")

  urdf_path = robot_visual_glb_urdf(
    args.robot,
    with_bio_gripper_g2=args.bio_gripper_g2,
    with_gripper_g2=args.gripper_g2,
    movable=args.movable,
  )
  print(f"Loading: {urdf_path}")
  run_glb_viewer(
    profile,
    urdf_path,
    headless=args.headless,
    pd_demo=args.pd,
    gripper_demo=args.gripper_demo,
  )


if __name__ == "__main__":
  main()
