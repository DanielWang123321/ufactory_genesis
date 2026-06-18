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
    "--lite6-gripper",
    action="store_true",
    help="Load Lite6 parallel gripper visual combo URDF (lite6 only)",
  )
  parser.add_argument(
    "--lite6-vacuum-gripper",
    action="store_true",
    help="Load Lite6 vacuum gripper static visual combo URDF (lite6 only)",
  )
  parser.add_argument(
    "--gripper-g2",
    action="store_true",
    help="Load Gripper G2 visual combo URDF",
  )
  parser.add_argument(
    "--movable",
    action="store_true",
    help="Per-link GLBs for gripper open/close (requires --gripper-g2, --lite6-gripper, or --bio-gripper-g2)",
  )
  parser.add_argument(
    "--gripper-demo",
    action="store_true",
    help="Cycle gripper open/close (requires --movable with a gripper flag)",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--pd",
    action="store_true",
    help="Joint motion demo (smooth 50 deg/s waypoints; visual only, not stiff PD)",
  )
  parser.add_argument(
    "--show-tcp",
    action="store_true",
    help="Show red DH TCP debug marker on EE flange (default: hidden)",
  )
  args = parser.parse_args()

  profile = get_robot_profile(args.robot)
  accessory_flags = (
    args.bio_gripper_g2,
    args.gripper_g2,
    args.lite6_gripper,
    args.lite6_vacuum_gripper,
  )
  if sum(accessory_flags) > 1:
    parser.error("Only one end-effector flag may be set at a time")
  if args.movable and not (args.gripper_g2 or args.lite6_gripper or args.bio_gripper_g2):
    parser.error("--movable requires --gripper-g2, --lite6-gripper, or --bio-gripper-g2")
  if args.gripper_demo and not args.movable:
    parser.error("--gripper-demo requires --movable")
  if args.bio_gripper_g2 and not profile.supports_bio_gripper_g2:
    parser.error(f"{args.robot} does not support Bio Gripper G2")
  if args.gripper_g2 and not profile.supports_gripper_g2:
    parser.error(f"{args.robot} does not support Gripper G2")
  if args.lite6_gripper and not profile.supports_lite6_gripper:
    parser.error(f"{args.robot} does not support Lite6 Gripper")
  if args.lite6_vacuum_gripper and not profile.supports_lite6_vacuum_gripper:
    parser.error(f"{args.robot} does not support Lite6 Vacuum Gripper")

  urdf_path = robot_visual_glb_urdf(
    args.robot,
    with_bio_gripper_g2=args.bio_gripper_g2,
    with_gripper_g2=args.gripper_g2,
    with_lite6_gripper=args.lite6_gripper,
    with_lite6_vacuum_gripper=args.lite6_vacuum_gripper,
    movable=args.movable,
  )
  print(f"Loading: {urdf_path}")
  run_glb_viewer(
    profile,
    urdf_path,
    headless=args.headless,
    pd_demo=args.pd,
    gripper_demo=args.gripper_demo,
    show_tcp=args.show_tcp,
  )


if __name__ == "__main__":
  main()
