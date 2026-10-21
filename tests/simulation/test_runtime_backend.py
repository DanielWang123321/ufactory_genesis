"""Tests for Genesis simulation backend overrides."""

from __future__ import annotations

from pathlib import Path

from ufactory.config import load_runtime_config
from ufactory.simulation import (
    BACKEND_INIT_HINT,
    genesis_backend_constant,
    override_simulation_backend,
)


def test_override_simulation_backend_cpu() -> None:
    config = load_runtime_config("xarm6")
    assert config.simulation.backend == "gpu"
    overridden = override_simulation_backend(config, "cpu")
    assert overridden.simulation.backend == "cpu"
    assert config.simulation.backend == "gpu"


def test_override_simulation_backend_noop() -> None:
    config = load_runtime_config("xarm6")
    same = override_simulation_backend(config, "gpu")
    assert same is config


def test_genesis_backend_constant_maps_cpu_gpu() -> None:
    class _Gs:
        cpu = object()
        gpu = object()

    gs = _Gs()
    assert genesis_backend_constant(gs, "cpu") is gs.cpu
    assert genesis_backend_constant(gs, "gpu") is gs.gpu


def test_packaging_simulation_does_not_hardcode_gs_gpu() -> None:
    source = Path("ufactory/manipulation/packaging/simulation.py").read_text(encoding="utf-8")
    assert "gs.init(backend=gs.gpu" not in source
    assert "GenesisRuntimeManager(runtime_config.simulation)" in source


def test_backend_init_hint_mentions_cpu_path() -> None:
    assert "--backend cpu" in BACKEND_INIT_HINT
    assert "pytorch.org" in BACKEND_INIT_HINT


def test_default_pick_place_holds_the_cube_without_creep(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys
    import pytest

    pytest.importorskip("genesis")
    pytest.importorskip("pinocchio")
    root = Path(__file__).resolve().parents[2]
    report = tmp_path / "pick.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/pick_place/run.py",
            "--robot",
            "xarm6",
            "--mode",
            "sim",
            "--executor",
            "servo_cartesian",
            "--backend",
            "cpu",
            "--sim-report",
            str(report),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    data = json.loads(report.read_text())
    acceptance = data["acceptance"]
    assert data["physics_parameters"] and data["completed"] and data["timing_passed"]
    assert acceptance["carry_max_slip_mm"] <= 1.0, acceptance["carry_max_slip_mm"]
    assert data["transport_passed"], acceptance["failed_checks"]


test_default_pick_place_holds_the_cube_without_creep = __import__("pytest").mark.integration(
    test_default_pick_place_holds_the_cube_without_creep
)
