"""Simulation regression for the dynamics pipeline (non-real-robot, GPU).

Drives ``dynamics-sim-check`` on the real ``xarm6_1305`` URDF with the calibrated
pose set (deterministic, no random poses) plus the Pinocchio reference oracle, then
asserts the invariants the CI gate cares about:

* every pose settles (no NOT_SETTLED / SATURATED)
* per-sample ``pd_hold_tau`` / ``pin_G`` / mass matrix are finite
* the mass matrix is symmetric and positive-definite
* L1/L2b/L3a layer metrics are finite
* the gravity oracle is non-trivial (``pin_G_l2`` > threshold)
* the report is schema version 2 and records a joint_frictionloss run value

This is a GPU/Genesis-dependent test by design (it actually runs the solver); the
``dynamics-hardware-check``/real-robot variants stay opt-in behind the
``hardware`` marker.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("genesis")
pytest.importorskip("pinocchio")

pytestmark = pytest.mark.gpu

from ufactory.dynamics_validation import cli_sim_check  # noqa: E402


def _load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") == "sample":
                out.append(item["data"])
            elif item.get("type") == "run_config":
                out.append(("run_config", item["data"]))
    return out


def test_dynamics_sim_regression_layers_finite(tmp_path):
    csv_path = tmp_path / "sim.csv"
    jsonl_path = tmp_path / "sim.jsonl"
    rc = cli_sim_check(
        [
            "--robot",
            "xarm6",
            "--random-count",
            "0",
            "--require-reference",
            "--z-min-mm",
            "0",
            "--report",
            str(csv_path),
            "--jsonl-report",
            str(jsonl_path),
        ]
    )
    assert rc == 0, f"cli_sim_check exited {rc}; see {csv_path}/{jsonl_path}"

    records = _load_jsonl(jsonl_path)
    run_config = next(r[1] for r in records if isinstance(r, tuple) and r[0] == "run_config")
    samples = [r for r in records if not isinstance(r, tuple)]
    assert samples, "no sample records written"
    assert run_config["version"] == "2"
    assert run_config.get("joint_frictionloss") is not None

    for s in samples:
        assert s["status"] not in {"NOT_SETTLED", "SATURATED"}, s["pose"]
        tau = np.asarray(s["pd_hold_tau"], dtype=np.float64)
        pin = np.asarray(s["reference_gravity_tau"], dtype=np.float64)
        mass = np.asarray(s["mass_matrix"], dtype=np.float64)
        assert np.isfinite(tau).all()
        assert np.isfinite(pin).all() and np.isfinite(mass).all()
        assert mass.shape == (6, 6)
        assert np.allclose(mass, mass.T, atol=1e-4)
        assert (np.linalg.eigvalsh(mass) > 0.0).all()
        assert np.isfinite(s["l2a_l2_err"])
        assert np.isfinite(s["l2b_l2_err"])
        assert np.isfinite(s["l3a_mass_rel_fro"])
        assert s["pin_G_l2"] > 1.0, s["pose"]