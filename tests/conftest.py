"""Shared pytest helpers and fixture paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Public CI installs ``.[dev]`` only (no Torch / Genesis). Modules that import
# those stacks at collection time must be ignored there; maintainer machines
# with ``.[sim]`` still collect and run them under the usual markers.
_TORCH_OR_GENESIS_COLLECTION_MODULES = frozenset(
    {
        "test_pinocchio_pose_ik.py",
        "test_tcp_offset.py",
        "test_packaging_drop.py",
        "test_artifacts.py",
        "test_logic_pick_place.py",
        "test_packaging_multi_robot.py",
        "test_packaging_showcase.py",
        "test_pick_place_preposition.py",
        "test_pick_place_program_defaults.py",
        "test_pick_place_visual_cli.py",
        "test_sim_executor_gripper.py",
        "test_trajectory_ik.py",
        "test_trajectory_mirror.py",
    }
)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:  # noqa: ARG001
    name = collection_path.name
    if name == "test_pick_place_next_params.py":
        decision = PROJECT_ROOT / "dev/rl/pick_place_next_params.py"
        recipe = PROJECT_ROOT / "examples/rl/pick_place/recipe.yaml"
        return not (decision.is_file() and recipe.is_file())
    if name in _TORCH_OR_GENESIS_COLLECTION_MODULES:
        try:
            import torch  # noqa: F401
        except ImportError:
            return True
    return False
