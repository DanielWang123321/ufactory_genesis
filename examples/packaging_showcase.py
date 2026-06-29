"""Generic packaging showcase entry point.

Currently implemented for the xArm6 + Gripper G2 task profile.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from ufactory.robot_params import get_robot_runtime_profile, robot_runtime_cli_choices


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--gripper-g2", action="store_true", default=True)
    args, remaining = parser.parse_known_args()

    runtime = get_robot_runtime_profile(args.robot)
    if not runtime.task.showcase_supported:
        raise SystemExit(f"{runtime.model.key} has no packaging showcase profile")
    if runtime.model.key != "xarm6_1305":
        raise SystemExit("The current packaging showcase implementation is xArm6 + Gripper G2 only")

    from xarm6.xarm6_g2_showcase import main as xarm6_showcase_main

    sys.argv = [sys.argv[0], *remaining]
    xarm6_showcase_main()


if __name__ == "__main__":
    main()
