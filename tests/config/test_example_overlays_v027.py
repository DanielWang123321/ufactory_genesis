"""Strict partial runtime overlays shipped beside pick-place/packaging examples."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from ufactory.config import ConfigError, load_runtime_config


def test_pick_place_example_overlay_changes_only_motion_values() -> None:
    overlay = PROJECT_ROOT / "examples" / "pick_place" / "runtime.example.yaml"
    baseline = load_runtime_config("xarm6", task="pick_place")
    resolved = load_runtime_config("xarm6", task="pick_place", config_path=overlay)

    assert resolved.motion.cartesian_speed_m_s == pytest.approx(0.150)
    assert resolved.motion.cartesian_acceleration_m_s2 == pytest.approx(0.800)
    assert resolved.task.parameters == baseline.task.parameters
    assert str(overlay.resolve()) in resolved.sources


def test_packaging_example_overlay_inherits_core_physical_defaults() -> None:
    overlay = PROJECT_ROOT / "examples" / "packaging" / "runtime.example.yaml"
    baseline = load_runtime_config("lite6", task="packaging_showcase")
    resolved = load_runtime_config("lite6", task="packaging_showcase", config_path=overlay)

    assert resolved.task.parameters["post_release_settle_s"] == pytest.approx(0.500)
    assert resolved.task.parameters["object_size_m"] == baseline.task.parameters["object_size_m"]
    assert resolved.robot == baseline.robot
    assert resolved.gripper == baseline.gripper


def test_example_style_overlay_rejects_unknown_fields(tmp_path: Path) -> None:
    overlay = tmp_path / "invalid.yaml"
    overlay.write_text("schema_version: 1\nmotion:\n  unknown_rate: 50\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown"):
        load_runtime_config("xarm6", task="pick_place", config_path=overlay)
