"""Real-URDF dry-run layered analysis regression (L2/L3 finite-ness).

This is a *non-simulation* regression: it builds the Pinocchio reference oracle
straight from the real ``xarm6_1305`` URDF and drives ``build_dynamics_sample``
with a self-consistent synthetic Genesis sample (whose ``pd_hold_tau`` equals the
gravity oracle and whose mass matrix equals the reference CRBA). It pins down
that, under the refactored semantics:

* L2a/L2b/L3a/L3b layer metrics are all finite
* the gravity oracle ``pin_G(q_actual)`` is a non-trivial quantity (NOT the old
  ~0 ``gravity_est``), so its L2 norm exceeds a threshold
* the L2b oracle ``pd_hold_tau vs pin_G(q_actual)`` returns ~0 when the two
  agree (the saturation-floor behaviour we want), not the prior full-|G| failure
* armature-aligned mass comparison is a no-op (rel-Frobenius ~0) when Genesis
  and the reference share the same mass matrix

The genuine Genesis-vs-Pinocchio CRBA discrepancy (a large mass_rel_fro on the
real simulation) is exercised by ``test_dynamics_sim_regression``; this test
isolates the analysis logic deterministically and without GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from ufactory.dynamics.analysis import build_dynamics_sample
from ufactory.dynamics.reference import load_reference_backend
from ufactory.dynamics.report import GenesisDynamicsSample, SafePose
from ufactory.robots.paths import xarm6_1305_urdf
from ufactory.dynamics.poses_config import dynamics_pose_tuples
from ufactory.robots.runtime import get_robot_runtime_profile

_RUNTIME = get_robot_runtime_profile("xarm6")
_URDF = xarm6_1305_urdf()


def _synthetic_genesis_sample(q: np.ndarray, reference) -> GenesisDynamicsSample:
    q = np.asarray(q, dtype=np.float64)
    pin_g = np.asarray(reference.gravity(q), dtype=np.float64)
    mass = np.asarray(reference.mass_matrix(q), dtype=np.float64)
    # A self-consistent hold: controller output equals gravity, zero velocity,
    # and the "measured" mass matrix equals the reference CRBA so L3a is a no-op.
    return GenesisDynamicsSample(
        q_actual=q,
        qvel=np.zeros_like(q),
        pd_hold_tau=pin_g,
        actual_dof_force=np.zeros_like(q),
        mass_matrix=mass,
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
        armature=np.zeros_like(q),
    )


@pytest.fixture(scope="module")
def reference():
    return load_reference_backend(_URDF, required=True)


@pytest.mark.parametrize(
    "pose_name, q",
    [(name, np.asarray(q, dtype=np.float64)) for name, q in dynamics_pose_tuples("xarm6")[:3]],
)
def test_dryrun_layer_metrics_finite_real_urdf(pose_name, q, reference):
    gsample = _synthetic_genesis_sample(q, reference)
    pose = SafePose(pose_name, q, ee_z_mm=200.0)
    sample = build_dynamics_sample(
        pose,
        gsample,
        runtime_profile=_RUNTIME,
        reference=reference,
        tau_real=np.asarray(reference.gravity(q), dtype=np.float64),
    )

    assert np.isfinite(sample.l2a_l2_err)
    assert np.isfinite(sample.l2b_l2_err)
    assert np.isfinite(sample.l3a_mass_rel_fro) and sample.l3a_mass_rel_fro >= 0.0
    assert sample.l3b_l2_err is not None and np.isfinite(sample.l3b_l2_err)
    assert np.isfinite(sample.clamp_slack_l2)

    # Oracle must be non-trivial: pin_G(q_actual) L2 norm well above zero.
    assert sample.pin_G_l2 is not None and sample.pin_G_l2 > 1.0

    # New L2b semantics: pd_hold_tau == pin_G here => residual ~0 (NOT the old
    # behaviour where gravity_est(~0) vs pin_G produced the full |G| failure).
    assert sample.l2b_l2_err < 1e-6

    # Armature-aligned mass comparison: shared mass matrix => rel-Frobenius ~0,
    # comfortably within the warn threshold.
    assert sample.l3a_mass_rel_fro <= _RUNTIME.dynamics.mass_rel_fro_limit_warn
    # L3b (pin_G vs tau_real=pin_G) also ~0 here.
    assert sample.l3b_l2_err < 1e-6


def test_dryrun_real_urdf_mass_matrix_is_physically_plausible(reference):
    q = np.asarray(dynamics_pose_tuples("xarm6")[1][1], dtype=np.float64)
    mass = np.asarray(reference.mass_matrix(q), dtype=np.float64)
    assert mass.shape == (6, 6)
    assert np.allclose(mass, mass.T, atol=1e-6)
    eig = np.linalg.eigvalsh(mass)
    assert (eig > 0).all()
