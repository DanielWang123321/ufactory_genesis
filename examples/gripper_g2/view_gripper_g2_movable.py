"""Standalone Gripper G2 movable visual demo."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from _gripper_demo import (
    control_gripper_pose,
    gripper_demo_target,
    gripper_dof_indices,
    setup_gripper_pd,
)
from _standalone_gripper_viewer import run_standalone_gripper_viewer
from ufactory.paths import gripper_g2_movable_visual_urdf


def _controller(robot):
    drive_idx, all_idx = gripper_dof_indices(robot)
    setup_gripper_pd(robot, drive_idx, all_idx)
    return lambda step: control_gripper_pose(robot, drive_idx, all_idx, gripper_demo_target(step))


def main() -> None:
    run_standalone_gripper_viewer("Gripper G2 open/close demo (standalone)", gripper_g2_movable_visual_urdf(), _controller)


if __name__ == "__main__":
    main()
