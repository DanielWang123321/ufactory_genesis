"""Lite6 gripper collision-view demo must match GLB viewer keyframes."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from ufactory.grippers.lite6 import LITE6_GRIPPER_SIM_CLOSED_DRIVE, LITE6_GRIPPER_SIM_OPEN_DRIVE

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_ROOT.parent
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))


def _modules():
    vpc = importlib.import_module("dev.ref_scripts.view_pose_collision")
    lite6 = importlib.import_module("_lite6_gripper_demo")
    return vpc, lite6


def test_lite6_collision_demo_drive_matches_glb_keyframes():
    vpc, lite6 = _modules()
    cases = (
        (0, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (lite6.LITE6_GRIPPER_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (lite6.LITE6_GRIPPER_HOLD_STEPS, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
        (2 * lite6.LITE6_GRIPPER_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
    )
    for step, expected in cases:
        glb_drive = lite6.lite6_gripper_demo_target(step)
        collision_drive = vpc._lite6_gripper_demo_drive(step)
        assert glb_drive == pytest.approx(expected)
        assert collision_drive == pytest.approx(glb_drive)


def test_lite6_collision_demo_labels_snap_at_hold_boundary():
    vpc, lite6 = _modules()
    assert vpc._lite6_gripper_demo_label(0) == "open"
    assert vpc._lite6_gripper_demo_label(lite6.LITE6_GRIPPER_HOLD_STEPS - 1) == "open"
    assert vpc._lite6_gripper_demo_label(lite6.LITE6_GRIPPER_HOLD_STEPS) == "closed"
    assert vpc._lite6_gripper_demo_label(2 * lite6.LITE6_GRIPPER_HOLD_STEPS - 1) == "closed"
