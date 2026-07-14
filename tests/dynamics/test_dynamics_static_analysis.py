"""Unit tests for layered static dynamics analysis (L2/L3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufactory.dynamics.analysis import (
    LAYER_L2A,
    LAYER_L2B,
    LAYER_L3A,
    LAYER_L3B,
    StaticPoseAnalysis,
    analyze_genesis_internal,
    analyze_mass_matrices,
    analyze_pin_vs_real,
    build_static_pose_analysis,
    estimate_pd_residual,
    parse_strict_static_layers,
    summarize_static_layers,
)
from ufactory.dynamics.report import (
    GenesisDynamicsSample,
    SafePose,
    ValidationStatus,
)
from ufactory.dynamics.analysis import build_dynamics_sample
from ufactory.robots.runtime import get_robot_runtime_profile


def _runtime():
    return get_robot_runtime_profile("xarm6")


def _genesis_sample(
    *,
    pd_hold: np.ndarray | None = None,
    dof_force: np.ndarray | None = None,
    settled: bool = True,
) -> GenesisDynamicsSample:
    n = 6
    pd = np.asarray(pd_hold if pd_hold is not None else np.ones(n), dtype=np.float64)
    dof = np.asarray(dof_force if dof_force is not None else pd, dtype=np.float64)
    return GenesisDynamicsSample(
        q_actual=np.zeros(n),
        qvel=np.zeros(n),
        pd_hold_tau=pd,
        actual_dof_force=dof,
        mass_matrix=np.eye(n),
        settled=settled,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )


def test_estimate_pd_residual_matches_formula():
    kp = np.array([100.0, 200.0])
    kv = np.array([10.0, 20.0])
    residual = estimate_pd_residual([0.1, 0.2], [0.0, 0.0], [0.05, 0.1], kp=kp, kv=kv)
    expected = np.array([100.0 * 0.1 - 10.0 * 0.05, 200.0 * 0.2 - 20.0 * 0.1])
    np.testing.assert_allclose(residual, expected)


def test_l2a_passes_when_pd_matches_dof_force():
    sample = _genesis_sample()
    layer = analyze_genesis_internal(sample, runtime_profile=_runtime())
    assert layer.layer == LAYER_L2A
    assert layer.severity == "pass"


def test_l2a_warns_on_internal_mismatch():
    sample = _genesis_sample(pd_hold=np.ones(6), dof_force=np.zeros(6))
    layer = analyze_genesis_internal(sample, runtime_profile=_runtime())
    assert layer.severity in {"warn", "fail"}


def test_mass_matrix_relative_frobenius():
    m_gen = np.eye(6)
    m_pin = np.eye(6) * 1.02
    layer, rel = analyze_mass_matrices(m_gen, m_pin, runtime_profile=_runtime())
    assert layer.layer == LAYER_L3A
    assert rel > 0.0
    assert layer.severity in {"pass", "warn", "fail"}


def test_build_static_pose_analysis_without_reference_only_l2a():
    analysis = build_static_pose_analysis(
        "0",
        np.zeros(6),
        _genesis_sample(),
        runtime_profile=_runtime(),
        reference=None,
    )
    assert len(analysis.layers) == 1
    assert analysis.layers[0].layer == LAYER_L2A
    assert analysis.clamp_slack_est is not None


def test_build_static_pose_analysis_with_mock_reference():
    reference = MagicMock()
    reference.gravity.return_value = np.ones(6)
    reference.rnea.return_value = np.ones(6)
    reference.mass_matrix.return_value = np.eye(6)
    analysis = build_static_pose_analysis(
        "0",
        np.zeros(6),
        _genesis_sample(),
        runtime_profile=_runtime(),
        reference=reference,
        tau_real=np.ones(6),
    )
    layers = {layer.layer for layer in analysis.layers}
    assert LAYER_L2A in layers
    assert LAYER_L2B in layers
    assert LAYER_L3B in layers
    assert analysis.mass_rel_fro is not None
    # New L2b oracle: pd_hold_tau vs pin_G(q_actual). Here both are ones(6),
    # so the L2b residual must be ~0 (pass), confirming the gravity oracle is
    # the controller output compared to G(q), not the old ~0 ``gravity_est``.
    l2b = next(layer for layer in analysis.layers if layer.layer == LAYER_L2B)
    assert l2b.l2_err < 1e-6
    assert l2b.notes == "pd_hold_tau vs pin_G"


def test_l1_pass_unchanged_when_l3_warns_by_default():
    pose = SafePose("0", np.zeros(6), 108.0)
    reference = MagicMock()
    reference.gravity.return_value = np.array([10.0, 0, 0, 0, 0, 0])
    reference.rnea.return_value = np.array([10.0, 0, 0, 0, 0, 0])
    reference.mass_matrix.return_value = np.eye(6) * 2.0
    sample = build_dynamics_sample(
        pose,
        _genesis_sample(),
        runtime_profile=_runtime(),
        tau_real=np.ones(6),
        reference=reference,
    )
    assert sample.status == ValidationStatus.PASS
    assert any(layer.severity != "pass" for layer in sample.static_layers) or sample.static_warnings
    # New L2b oracle (pd_hold_tau vs pin_G(q_actual)): pd_hold=ones(6) while
    # pin_G=[10,0,0,0,0,0] => residual ~9 Nm on J1, well over the abs limit,
    # so the L2b static layer must flag as warn/fail (the old ``gravity_est``
    # oracle would have flagged J1 error as the full G(q) magnitude instead).
    l2b = next(layer for layer in sample.static_layers if layer.layer == LAYER_L2B)
    assert l2b.severity in {"warn", "fail"}
    assert l2b.l2_err > 1.0


def test_strict_static_summary_flags_warn_layers():
    warn_layer = SimpleNamespace(layer=LAYER_L3B, severity="warn")
    analysis = StaticPoseAnalysis(pose="0", layers=[warn_layer])
    _, strict_fail = summarize_static_layers([analysis], strict_layers=frozenset({LAYER_L3B}))
    assert strict_fail is True


def test_parse_strict_static_layers_aliases():
    assert "L2a" in parse_strict_static_layers("l2a")
    assert "L2a" in parse_strict_static_layers("l2")
    assert "L3b" in parse_strict_static_layers("l3")
    assert len(parse_strict_static_layers("all")) == 4


@pytest.mark.parametrize(
    ("value", "expected_count"),
    [
        (None, 4),
        ("l2a,l3a", 2),
    ],
)
def test_parse_strict_static_layers_counts(value, expected_count):
    assert len(parse_strict_static_layers(value)) == expected_count


def test_analyze_pin_vs_real_uses_scaled_limits():
    runtime = _runtime()
    pin_g = np.zeros(6)
    tau_real = np.zeros(6)
    tau_real[0] = runtime.dynamics.abs_err_limits[0] * runtime.dynamics.pin_vs_real_abs_scale + 0.01
    layer = analyze_pin_vs_real(pin_g, tau_real, runtime_profile=runtime)
    assert layer.layer == LAYER_L3B
    assert layer.severity in {"warn", "fail"}


def test_pd_hold_check_xarm5_pose4_tracking_saturation_bypass():
    """Regression: pose 4 saturated pd_hold_tau on J4 while pin_G stays ~1 Nm."""
    from ufactory.dynamics.analysis import (
        evaluate_pd_hold_gate,
        genesis_sample_for_torque_compare,
    )

    runtime = get_robot_runtime_profile("xarm5")
    target_q = np.deg2rad([90.0, -90.0, -60.0, 160.0, -90.0])
    q_actual = np.deg2rad([90.0, -90.216, -59.898, 155.574, -90.005])
    sample = GenesisDynamicsSample(
        q_actual=q_actual,
        qvel=np.zeros(5),
        pd_hold_tau=np.array([0.0006, 11.318, -3.549, 20.0, 0.0824]),
        actual_dof_force=np.zeros(5),
        mass_matrix=np.eye(5),
        settled=False,
        saturated=True,
        pos_err=float(np.abs(q_actual - target_q).max()),
        vel_mag=0.0,
    )
    reference = MagicMock()
    reference.gravity.side_effect = lambda q: np.array(
        [-0.0, 11.317, -3.548, -0.897, 0.0],
        dtype=np.float64,
    )

    gate = evaluate_pd_hold_gate(sample, target_q, runtime_profile=runtime, reference=reference)
    assert gate.block_hardware is False
    assert gate.reason == "pd_tracking_saturation"
    assert gate.tracking_limited_joints == (3,)

    compare_sample = genesis_sample_for_torque_compare(sample, gate)
    assert compare_sample.settled is True
    assert compare_sample.saturated is False
    assert compare_sample.pd_hold_tau[3] == pytest.approx(-0.897, abs=0.05)
