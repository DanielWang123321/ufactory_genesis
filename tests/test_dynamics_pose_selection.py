"""Unit tests for dynamics pose selection helpers."""

from __future__ import annotations

import numpy as np

from ufactory.dynamics_pose_selection import (
    EE_Y_SIDE_NEG,
    EE_Y_SIDE_POS,
    PoseCandidate,
    classify_ee_y_side,
    select_stratified_by_ee_y,
)


def _candidate(name: str, y_mm: float, q: np.ndarray | None = None) -> PoseCandidate:
    return PoseCandidate(
        name=name,
        q=q if q is not None else np.zeros(6),
        ee_x_mm=0.0,
        ee_y_mm=y_mm,
        ee_z_mm=100.0,
        tau_norm=float(abs(y_mm)),
        ee_y_side=classify_ee_y_side(y_mm),
    )


def test_classify_ee_y_side():
    assert classify_ee_y_side(20.0) == EE_Y_SIDE_POS
    assert classify_ee_y_side(-20.0) == EE_Y_SIDE_NEG
    assert classify_ee_y_side(5.0) == "neutral"


def test_select_stratified_by_ee_y_balances_hemispheres():
    candidates = [_candidate(f"p{i}", 100.0 + i, np.full(6, i * 0.5)) for i in range(12)]
    candidates += [_candidate(f"n{i}", -100.0 - i, np.full(6, -i * 0.5)) for i in range(12)]
    selected = select_stratified_by_ee_y(candidates, n_y_pos=5, n_y_neg=5, min_separation_rad=0.1)
    y_pos = sum(1 for c in selected if c.ee_y_mm > 10.0)
    y_neg = sum(1 for c in selected if c.ee_y_mm < -10.0)
    assert len(selected) == 10
    assert y_pos == 5
    assert y_neg == 5
