"""Lite6 gripper collision-view demo must match GLB viewer keyframes."""

from __future__ import annotations

import pytest

from ufactory.grippers.lite6 import (
    LITE6_GRIPPER_DEMO_HOLD_STEPS,
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    lite6_gripper_demo_drive,
    lite6_gripper_demo_label,
)


def test_lite6_collision_demo_drive_matches_glb_keyframes():
    cases = (
        (0, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (LITE6_GRIPPER_DEMO_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (LITE6_GRIPPER_DEMO_HOLD_STEPS, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
        (2 * LITE6_GRIPPER_DEMO_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
    )
    for step, expected in cases:
        assert lite6_gripper_demo_drive(step) == pytest.approx(expected)


def test_lite6_collision_demo_labels_snap_at_hold_boundary():
    assert lite6_gripper_demo_label(0) == "open"
    assert lite6_gripper_demo_label(LITE6_GRIPPER_DEMO_HOLD_STEPS - 1) == "open"
    assert lite6_gripper_demo_label(LITE6_GRIPPER_DEMO_HOLD_STEPS) == "closed"
    assert lite6_gripper_demo_label(2 * LITE6_GRIPPER_DEMO_HOLD_STEPS - 1) == "closed"
