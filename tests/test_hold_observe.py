"""Unit tests for hold time-series observation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ufactory.dynamics.observe import analyze_hold_timeseries
from ufactory.real_robot_session import HoldTimeSeriesSample, RealRobotSession
from ufactory.robot_params import get_robot_runtime_profile


def _make_mock_arm(initial_q):
    current_q = np.asarray(initial_q, dtype=np.float64)
    tick = {"n": 0}
    arm = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    arm.currents = [0.0] * 7

    def get_joint_states(*, is_radian=True, num=3):
        del is_radian, num
        n = tick["n"]
        tau = np.full(current_q.size, -10.0 - 3.0 * math.exp(-0.5 * n), dtype=np.float64)
        current = np.full(current_q.size, 1.0 + 0.5 * math.exp(-0.5 * n), dtype=np.float64)
        arm.currents = current.tolist() + [0.0]
        tick["n"] += 1
        return 0, [current_q.tolist(), np.zeros(current_q.size).tolist(), tau.tolist()]

    arm.get_joint_states.side_effect = get_joint_states
    arm.get_joints_torque.return_value = (0, np.zeros(current_q.size).tolist())
    return arm


def _make_session(arm):
    session = object.__new__(RealRobotSession)
    session.ip = "mock"
    session.dof = 6
    session.home_qpos = np.zeros(6)
    session._motion_mode = 0
    session.arm = arm
    return session


def test_collect_hold_timeseries_sample_count(monkeypatch):
    clock = {"t": 0.0}

    def fake_monotonic():
        return clock["t"]

    def fake_sleep(dt):
        clock["t"] += dt

    monkeypatch.setattr("ufactory.real_robot_session.time.monotonic", fake_monotonic)
    monkeypatch.setattr("ufactory.real_robot_session.time.sleep", fake_sleep)
    arm = _make_mock_arm(np.zeros(6))
    session = _make_session(arm)

    samples = session.collect_hold_timeseries(duration_s=0.5, poll_s=0.1)

    assert len(samples) == 6
    assert samples[0].t_s == pytest.approx(0.0, abs=1e-3)
    assert samples[-1].t_s == pytest.approx(0.5, abs=0.15)
    assert samples[0].current.shape == (6,)
    assert samples[0].tau.shape == (6,)


def test_analyze_hold_timeseries_detects_decay():
    samples = []
    for i in range(601):
        t_s = i * 0.1
        decay = math.exp(-0.4 * t_s)
        tau_j2 = -20.0 + 6.0 * (1.0 - decay)
        current_j2 = 2.0 * decay + 0.2
        tau = np.full(6, tau_j2, dtype=np.float64)
        current = np.full(6, current_j2, dtype=np.float64)
        samples.append(
            HoldTimeSeriesSample(
                t_s=t_s,
                q=np.zeros(6),
                qvel=np.zeros(6),
                tau=tau,
                current=current,
            )
        )

    j2_limit = float(get_robot_runtime_profile("lite6").dynamics.abs_err_limits[1])
    summary = analyze_hold_timeseries(
        samples,
        joint=2,
        context="direct",
        pose="0",
        duration_s=60.0,
        poll_s=0.1,
        pin_g_j=-11.586,
        abs_err_limit_j=j2_limit,
    )

    assert summary.segment_a.n > 0
    assert summary.segment_c.n > 0
    assert summary.segment_a.tau_mean < summary.segment_c.tau_mean
    assert summary.l1_fail_segment_a is True
    assert summary.tau_current_corr_all is not None
    assert summary.interpretation
