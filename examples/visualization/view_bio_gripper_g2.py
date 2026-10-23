"""Standalone Bio Gripper G2 movable visual demo."""

from __future__ import annotations

from ufactory.grippers.bio_g2 import BioGripperG2
from ufactory.robots.paths import bio_gripper_g2_movable_visual_urdf
from ufactory.visualization.standalone_gripper import run_standalone_gripper_viewer


def _controller(robot):
    gripper = BioGripperG2(robot)
    gripper.setup_pd()
    return lambda _step: gripper.control_pose(gripper.step_demo_target()[0])


def main() -> dict | None:
    return run_standalone_gripper_viewer(
        "Bio Gripper G2 open/close demo (standalone)",
        bio_gripper_g2_movable_visual_urdf(),
        _controller,
        calibrated_bio_g2=True,
    )


if __name__ == "__main__":
    main()
