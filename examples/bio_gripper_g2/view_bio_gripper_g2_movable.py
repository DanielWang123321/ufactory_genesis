"""Standalone Bio Gripper G2 movable visual demo."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from _standalone_gripper_viewer import run_standalone_gripper_viewer
from ufactory.bio_gripper_g2 import BioGripperG2
from ufactory.paths import bio_gripper_g2_movable_visual_urdf


def _controller(robot):
    gripper = BioGripperG2(robot)
    gripper.setup_pd()
    return lambda step: gripper.control_pose(gripper.demo_target(step))


def main() -> None:
    run_standalone_gripper_viewer("Bio Gripper G2 open/close demo (standalone)", bio_gripper_g2_movable_visual_urdf(), _controller)


if __name__ == "__main__":
    main()
