"""Standalone Gripper G2 movable visual demo."""

from __future__ import annotations

from ufactory.visualization.gripper_g2_viewer import (
    control_gripper_pose,
    gripper_dof_indices,
    setup_gripper_pd,
    step_gripper_demo_target,
)
from ufactory.visualization.standalone_gripper import run_standalone_gripper_viewer
from ufactory.robots.paths import gripper_g2_movable_visual_urdf


def _controller(robot):
    drive_idx, all_idx = gripper_dof_indices(robot)
    setup_gripper_pd(robot, drive_idx, all_idx)
    return lambda _step: control_gripper_pose(robot, drive_idx, all_idx, step_gripper_demo_target(robot)[0])


def main() -> None:
    run_standalone_gripper_viewer(
        "Gripper G2 open/close demo (standalone)", gripper_g2_movable_visual_urdf(), _controller, calibrated_g2=True
    )


if __name__ == "__main__":
    main()
