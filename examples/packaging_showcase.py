"""Generic packaging showcase entry point.

Supports xArm5/6/7 and UF850 with Gripper G2 models, plus the Lite6 gripper.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from ufactory.robots.runtime import get_robot_runtime_profile, robot_runtime_cli_choices


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--gripper-g2", action="store_true", default=True)
    args, remaining = parser.parse_known_args()

    runtime = get_robot_runtime_profile(args.robot)
    if not runtime.task.showcase_supported:
        raise SystemExit(f"{runtime.model.key} has no packaging showcase profile")

    from ufactory.cli.packaging import main as packaging_main

    raise SystemExit(packaging_main(["--robot", args.robot, *remaining]))


if __name__ == "__main__":
    main()
