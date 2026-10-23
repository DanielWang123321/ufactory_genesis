"""Standalone Lite6 gripper movable visual demo."""

from __future__ import annotations

from ufactory.visualization.lite6_gripper_viewer import (
    LITE6_GRIPPER_OPEN,
    control_lite6_gripper_pose,
    lite6_gripper_dof_indices,
    set_lite6_gripper_pose,
    setup_lite6_gripper_pd,
    step_lite6_gripper_demo_target,
)
from ufactory.visualization.standalone_gripper import run_standalone_gripper_viewer
from ufactory.robots.paths import lite6_gripper_movable_visual_urdf


def _controller(robot):
    drive_idx, all_idx = lite6_gripper_dof_indices(robot)
    setup_lite6_gripper_pd(robot, drive_idx, all_idx)
    set_lite6_gripper_pose(robot, drive_idx, all_idx, LITE6_GRIPPER_OPEN)
    return lambda _step: control_lite6_gripper_pose(
        robot,
        drive_idx,
        all_idx,
        step_lite6_gripper_demo_target(robot, drive_idx)[0],
    )


def main() -> None:
    run_standalone_gripper_viewer(
        "Lite6 gripper open/close demo (standalone)", lite6_gripper_movable_visual_urdf(), _controller
    )


if __name__ == "__main__":
    main()
