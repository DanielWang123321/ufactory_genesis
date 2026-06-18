"""Unit tests for xArm6 LeRobot feature encoding (no Genesis)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

LEROBOT_DIR = Path(__file__).resolve().parents[1] / "examples" / "xarm6" / "lerobot"
sys.path.insert(0, str(LEROBOT_DIR))

from constants import ALLOWED_FEATURE_KEYS, DRIVE_CLOSED  # noqa: E402
from xarm6_lerobot_features import (  # noqa: E402
    assert_allowed_features,
    drive_from_gripper_openness,
    gripper_openness_from_drive,
    gripper_openness_from_width_mm,
    pack_ee_state,
    quat_wxyz_to_rotvec,
    rotvec_to_quat_wxyz,
    unpack_ee_state,
)


def test_gripper_openness_semantics():
    assert gripper_openness_from_drive(0.0) == pytest.approx(1.0)
    assert gripper_openness_from_drive(DRIVE_CLOSED) == pytest.approx(0.0)
    mid = gripper_openness_from_drive(DRIVE_CLOSED / 2)
    assert 0.0 < mid < 1.0
    assert drive_from_gripper_openness(1.0) == pytest.approx(0.0)
    assert drive_from_gripper_openness(0.0) == pytest.approx(DRIVE_CLOSED)


def test_width_mm_mapping():
    assert gripper_openness_from_width_mm(0.0) == 0.0
    assert gripper_openness_from_width_mm(84.0) == pytest.approx(1.0)


def test_pack_unpack_roundtrip():
    state = pack_ee_state(np.array([0.3, 0.0, 0.3]), np.array([0.1, 0.0, 0.0]), 0.5)
    pos, rot, g = unpack_ee_state(state)
    assert pos.shape == (3,)
    assert rot.shape == (3,)
    assert g == pytest.approx(0.5)


def test_quat_rotvec_roundtrip():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rv = quat_wxyz_to_rotvec(q)
    q2 = rotvec_to_quat_wxyz(rv)
    assert q2 == pytest.approx(q, abs=1e-6)


def test_feature_whitelist_excludes_tof():
    good = {
        "observation.state": {},
        "action": {},
        "observation.images.wrist": {},
    }
    assert_allowed_features(good)
    assert "observation.images.wrist" in ALLOWED_FEATURE_KEYS
    with pytest.raises(ValueError, match="ToF"):
        assert_allowed_features({**good, "observation.depth": {}})
