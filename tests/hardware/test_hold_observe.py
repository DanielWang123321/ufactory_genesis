"""Unit tests for hold time-series analysis (synthetic samples, robot-agnostic)."""

from __future__ import annotations

import math

import numpy as np

from ufactory.hardware.observe import analyze_hold_timeseries
from ufactory.hardware.session import HoldTimeSeriesSample

# Synthetic J2 reference values, deliberately not taken from any robot profile.
PIN_G_J2_NM = -10.0
ABS_ERR_LIMIT_J2_NM = 8.0


def test_analyze_hold_timeseries_detects_decay():
    # Segment A mean is about 2.5 Nm above the t=0 torque. Start far enough
    # below pin_G that the mean still exceeds the absolute error limit.
    tau_start = PIN_G_J2_NM - (ABS_ERR_LIMIT_J2_NM + 4.0) - 2.5
    samples = []
    for i in range(601):
        t_s = i * 0.1
        decay = math.exp(-0.4 * t_s)
        tau_j2 = tau_start + 6.0 * (1.0 - decay)
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

    summary = analyze_hold_timeseries(
        samples,
        joint=2,
        context="direct",
        pose="0",
        duration_s=60.0,
        poll_s=0.1,
        pin_g_j=PIN_G_J2_NM,
        abs_err_limit_j=ABS_ERR_LIMIT_J2_NM,
    )

    assert summary.segment_a.n > 0
    assert summary.segment_c.n > 0
    assert summary.segment_a.tau_mean < summary.segment_c.tau_mean
    assert summary.l1_fail_segment_a is True
    assert summary.tau_current_corr_all is not None
    assert summary.interpretation
