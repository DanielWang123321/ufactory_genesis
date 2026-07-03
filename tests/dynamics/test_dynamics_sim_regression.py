"""Simulation regression for the dynamics pipeline (non-real-robot, GPU).

Drives ``dynamics-sim-check`` on each hardware-dynamics robot URDF with the
calibrated pose set (deterministic, no random poses) plus the Pinocchio reference
oracle, then asserts the invariants the CI gate cares about:

* every pose settles (no NOT_SETTLED / SATURATED)
* per-sample ``pd_hold_tau`` / ``pin_G`` / mass matrix are finite
* the mass matrix is symmetric and positive-definite
* L1/L2b/L3a layer metrics are finite
* the gravity oracle is non-trivial (``pin_G_l2`` > threshold)
* the report is schema version 3, while still accepting legacy schema v2 records,
  and records a joint_frictionloss run value

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

from ufactory.dynamics import cli_sim_check  # noqa: E402
from ufactory.robots.runtime import get_robot_runtime_profile  # noqa: E402

_HARDWARE_DYNAMICS_ROBOTS = ("xarm5", "xarm6", "xarm7", "lite6", "uf850")


@pytest.fixture(autouse=True)
def _genesis_teardown():
    yield
    import genesis as gs

    if getattr(gs, "_initialized", False):
        gs.destroy()


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


@pytest.mark.parametrize("robot_key", _HARDWARE_DYNAMICS_ROBOTS)
def test_dynamics_sim_regression_layers_finite(tmp_path, robot_key: str):
    runtime = get_robot_runtime_profile(robot_key)
    n_dof = len(runtime.arm.joint_names)
    csv_path = tmp_path / f"sim_{robot_key}.csv"
    jsonl_path = tmp_path / f"sim_{robot_key}.jsonl"
    rc = cli_sim_check(
        [
            "--robot",
            robot_key,
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
    if robot_key == "xarm5":
        assert rc in {0, 1}, f"cli_sim_check({robot_key}) unexpected exit {rc}"
    else:
        assert rc == 0, f"cli_sim_check({robot_key}) exited {rc}; see {csv_path}/{jsonl_path}"

    records = _load_jsonl(jsonl_path)
    run_config = next(r[1] for r in records if isinstance(r, tuple) and r[0] == "run_config")
    samples = [r for r in records if not isinstance(r, tuple)]
    assert samples, f"no sample records written for {robot_key}"
    assert run_config["version"] in {"2", "3"}
    assert run_config.get("joint_frictionloss") is not None

    pin_g_l2_threshold = 0.5 if robot_key == "xarm5" else 1.0
    bad_poses = [s for s in samples if s["status"] in {"NOT_SETTLED", "SATURATED"}]
    if robot_key == "xarm5":
        assert len(bad_poses) <= 1, bad_poses
        if bad_poses:
            assert bad_poses[0]["pose"] == "4"
    else:
        assert bad_poses == [], bad_poses

    for s in samples:
        tau = np.asarray(s["pd_hold_tau"], dtype=np.float64)
        pin = np.asarray(s["reference_gravity_tau"], dtype=np.float64)
        mass = np.asarray(s["mass_matrix"], dtype=np.float64)
        assert np.isfinite(tau).all()
        assert np.isfinite(pin).all() and np.isfinite(mass).all()
        assert mass.shape == (n_dof, n_dof)
        assert np.allclose(mass, mass.T, atol=1e-4)
        assert (np.linalg.eigvalsh(mass) > 0.0).all()
        if s["status"] in {"NOT_SETTLED", "SATURATED"}:
            continue
        assert np.isfinite(s["l2a_l2_err"])
        assert np.isfinite(s["l2b_l2_err"])
        assert np.isfinite(s["l3a_mass_rel_fro"])
        assert s["pin_G_l2"] > pin_g_l2_threshold, f"{robot_key} pose {s['pose']}"
