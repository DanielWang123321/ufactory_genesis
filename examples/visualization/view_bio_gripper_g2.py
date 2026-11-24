"""Standalone Bio Gripper G2 movable visual demo."""

from __future__ import annotations

from ufactory.grippers.bio_g2 import BioGripperG2
from ufactory.robots.paths import bio_gripper_g2_movable_visual_urdf
from ufactory.visualization.standalone_gripper import run_standalone_gripper_viewer

# Shown as mounted on an arm at home: base link turned 180 deg about X, adapter flange up, TCP down.
MOUNTED_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
# The URDF base link is 0.12 m above its root, so it sits at 0.29 - 0.12 = 0.17 m after the flip. The gripper then
# spans the same 0.11-0.18 m heights as unflipped, and the shared camera still frames it.
ROOT_POS = (0.0, 0.0, 0.29)


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
        base_pos=ROOT_POS,
        base_quat=MOUNTED_QUAT_WXYZ,
    )


if __name__ == "__main__":
    main()
