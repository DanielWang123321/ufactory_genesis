"""Package imports must not require the optional ``sim`` extra (Torch / Genesis)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Entry points imported by CPU-only tests; blocking the optional stacks catches
# eager imports even on machines where Torch and Genesis are installed.
_CHECK = """
import sys
sys.modules["torch"] = None
sys.modules["genesis"] = None
sys.modules["trimesh"] = None
import ufactory.simulation
import ufactory.visualization
import ufactory.dynamics.cli
import ufactory.training.tasks
from ufactory.simulation import G2_PHYSICS_PROFILE, make_rigid_options, GenesisRuntimeManager
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive, gripper_g2_sim_drive_to_gap_m
for gap in (0, 0.010, 0.02415587200800565, 0.030, 0.060, 0.084):
    assert abs(gripper_g2_sim_drive_to_gap_m(gripper_g2_gap_m_to_sim_drive(gap)) - gap) < 1e-12
"""


def test_package_imports_do_not_require_torch_or_genesis() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_g2_tests_collect_without_optional_simulation_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHECK
            + "\nimport pytest\nraise SystemExit(pytest.main(['--collect-only', '-q', 'tests/simulation', 'tests/robots']))",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
