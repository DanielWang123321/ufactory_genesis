"""Bio Gripper G2 parallel open/close demo helpers for Genesis viewers.

Thin backward-compatible wrapper around :class:`ufactory.bio_gripper_g2.BioGripperG2`.
Prefer the class for new code; this module is kept for existing callers.
"""

from __future__ import annotations

from ufactory.bio_gripper_g2 import (
    BioGripperG2,
    CLOSED_GAP,
    CLOSE_POS,
    OPEN_GAP,
    OPEN_POS,
    STROKE,
)

# Re-export constants for backward compatibility.
# Joint zero is CLOSED (71 mm gap); the dof opens to STROKE (150 mm gap).
BIO_GRIPPER_G2_OPEN = OPEN_POS
BIO_GRIPPER_G2_CLOSE = CLOSE_POS
BIO_GRIPPER_G2_CLOSED_GAP = CLOSED_GAP
BIO_GRIPPER_G2_OPEN_GAP = OPEN_GAP
BIO_GRIPPER_G2_HOLD_STEPS = 200

BIO_GRIPPER_G2_JOINTS = (
    "bio_gripper_g2_right_finger_joint",
    "bio_gripper_g2_left_finger_joint",
)


def bio_gripper_g2_dof_indices(robot) -> tuple[list[int], list[int]]:
    """Discover Bio Gripper G2 DOF indices on *robot*.

    Returns:
        ``(drive_idx, all_idx)`` — drive_idx has the right-finger DOF;
        all_idx has both finger DOFs.  Returns ``([], [])`` if the gripper
        is not found.
    """
    g = BioGripperG2(robot)
    if not g.found:
        return [], []
    return g.drive_dof_idx, g.all_dof_idx


def setup_bio_gripper_g2_pd(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
) -> None:
    """Configure PD gains for Bio Gripper G2 on *robot*."""
    BioGripperG2(robot).setup_pd()


def set_bio_gripper_g2_pose(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    right_value: float,
) -> None:
    """Teleport Bio Gripper G2 to *right_value* (kinematic)."""
    BioGripperG2(robot).set_pose(right_value)


def control_bio_gripper_g2_pose(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    right_value: float,
) -> None:
    """PD-control Bio Gripper G2 toward *right_value*."""
    BioGripperG2(robot).control_pose(right_value)


def bio_gripper_g2_demo_target(step: int) -> float:
    """Alternating open/close target for demo loops."""
    return BioGripperG2.demo_target(step)
