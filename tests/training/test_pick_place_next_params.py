"""Unit tests for pick_place_next_params decision axes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/rl/pick_place/recipe.yaml"
_SPEC = importlib.util.spec_from_file_location(
    "pick_place_next_params",
    ROOT / "dev/rl/pick_place_next_params.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
decide = _MOD.decide


def _base() -> dict:
    return yaml.safe_load(RECIPE.read_text(encoding="utf-8"))


def test_frac_down_on_grasp_forgetting():
    tag, recipe = decide(
        {"grasp_pct": 40.0, "place_pct": 25.0, "success_pct": 5.0},
        _base(),
    )
    assert tag == "frac_down"
    assert recipe["environment"]["place_phase_reset_frac"] == 0.2


def test_place_lower_when_grasp_high_place_low():
    tag, recipe = decide(
        {"grasp_pct": 80.0, "place_pct": 10.0, "success_pct": 0.0},
        _base(),
    )
    assert tag == "place_lower"
    assert recipe["reward"]["lower"] > _base()["reward"]["lower"]
    assert recipe["reward"]["place_z"] > _base()["reward"]["place_z"]


def test_stabilize_release():
    tag, recipe = decide(
        {"grasp_pct": 75.0, "place_pct": 45.0, "success_pct": 10.0},
        _base(),
    )
    assert tag == "stabilize_release"
    assert recipe["reward"]["success"] > _base()["reward"]["success"]


def test_ppo_mode_cycles_lr():
    base = _base()
    tag, recipe = decide({"ppo_axis": 0}, base, mode="ppo")
    assert tag.startswith("ppo_lr")
    assert recipe["train"]["algorithm"]["learning_rate"] in (2e-4, 3e-4, 4.5e-4)


def test_hold_streak_triggers_ppo_nudge():
    tag, _recipe = decide(
        {"grasp_pct": 75.0, "place_pct": 45.0, "success_pct": 25.0},
        _base(),
        hold_streak=2,
    )
    assert tag.startswith("ppo_")
