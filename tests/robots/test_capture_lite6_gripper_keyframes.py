"""Standalone Lite6 gripper keyframe capture must match GLB kinematics."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.mark.slow
def test_standalone_lite6_gripper_collision_matches_glb_kinematics():
    capture = importlib.import_module("dev.ref_scripts.capture_lite6_gripper_keyframes")
    assert capture.main([]) == 0
