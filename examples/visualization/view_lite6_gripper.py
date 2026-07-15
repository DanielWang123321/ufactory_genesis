"""Standalone Lite6 gripper movable visual demo."""

from __future__ import annotations

from ufactory.visualization.lite6_gripper_viewer import (
    control_lite6_gripper_pose,
    lite6_gripper_demo_target,
    lite6_gripper_dof_indices,
    setup_lite6_gripper_pd,
)
from ufactory.visualization.standalone_gripper import run_standalone_gripper_viewer
from ufactory.robots.paths import lite6_gripper_movable_visual_urdf


def _controller(robot):
    drive_idx, all_idx = lite6_gripper_dof_indices(robot)
    setup_lite6_gripper_pd(robot, drive_idx, all_idx)
    return lambda step: control_lite6_gripper_pose(robot, drive_idx, all_idx, lite6_gripper_demo_target(step))


def main() -> None:
    run_standalone_gripper_viewer(
        "Lite6 gripper open/close demo (standalone)", lite6_gripper_movable_visual_urdf(), _controller
    )


if __name__ == "__main__":
    main()
