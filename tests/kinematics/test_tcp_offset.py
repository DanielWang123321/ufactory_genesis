"""Unit tests for flange ↔ TCP pose conversion via firmware tcp_offset."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.kinematics.tcp_offset import (
    pose_flange_to_tcp,
    pose_tcp_to_flange,
    read_tcp_offset,
)
from ufactory.kinematics.validation import angle_diff_deg


def test_tcp_z_200mm_shifts_down_when_flange_points_down():
    # roll=pi → tool +Z aligns with world -Z (Lite6 home flange orientation).
    flange = np.array([87.0, 0.0, 153.6, math.pi, 0.0, 0.0], dtype=np.float64)
    offset = np.array([0.0, 0.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tcp = pose_flange_to_tcp(flange, offset)
    assert tcp[:3] == pytest.approx([87.0, 0.0, -46.4], abs=1e-6)
    assert max(angle_diff_deg(a, b) for a, b in zip(tcp[3:], flange[3:])) < 1e-6


def test_pose_tcp_round_trip_restores_flange():
    flange = np.array([100.0, -20.0, 150.0, 0.3, -0.2, 0.4], dtype=np.float64)
    offset = np.array([10.0, -5.0, 200.0, 0.1, -0.05, 0.02], dtype=np.float64)
    tcp = pose_flange_to_tcp(flange, offset)
    restored = pose_tcp_to_flange(tcp, offset)
    assert restored[:3] == pytest.approx(flange[:3], abs=1e-9)
    assert max(angle_diff_deg(a, b) for a, b in zip(restored[3:], flange[3:])) < 1e-9


def test_zero_tcp_offset_is_identity():
    flange = np.array([50.0, 25.0, 80.0, -0.5, 0.25, 1.0], dtype=np.float64)
    offset = np.zeros(6, dtype=np.float64)
    assert pose_flange_to_tcp(flange, offset) == pytest.approx(flange)
    assert pose_tcp_to_flange(flange, offset) == pytest.approx(flange)


def test_read_tcp_offset_returns_nonzero_once_report_arrives(monkeypatch):
    values = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]]
    calls = {"i": 0}

    class _Arm:
        @property
        def tcp_offset(self):
            idx = min(calls["i"], len(values) - 1)
            calls["i"] += 1
            return list(values[idx])

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    offset = read_tcp_offset(_Arm(), timeout_s=1.0, poll_s=0.01)
    assert offset == pytest.approx([0.0, 0.0, 200.0, 0.0, 0.0, 0.0])


def test_read_tcp_offset_accepts_persistent_zero(monkeypatch):
    arm = SimpleNamespace(tcp_offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    offset = read_tcp_offset(arm, timeout_s=0.05, poll_s=0.01)
    assert offset == pytest.approx(np.zeros(6))
