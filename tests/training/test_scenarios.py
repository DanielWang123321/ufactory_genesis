from __future__ import annotations

import json

import pytest

from ufactory.training import (
    ArtifactError,
    generate_pick_place_scenario_bank,
    load_scenario_bank,
    scenario_bank_sha256,
    write_scenario_bank,
)


def _generate(*, count: int = 32, seed: int = 17, mode: str = "fixed") -> dict:
    return generate_pick_place_scenario_bank(
        count=count,
        seed=seed,
        mode=mode,
        runtime_config_sha256="a" * 64,
        fixed_obj=[0.30, 0.00, 0.015],
        fixed_target=[0.30, 0.30, 0.015],
        obj_spawn_lower=[0.28, -0.05, 0.015],
        obj_spawn_upper=[0.34, 0.05, 0.015],
        target_spawn_lower=[0.28, 0.25, 0.015],
        target_spawn_upper=[0.34, 0.35, 0.015],
    )


def test_scenario_bank_is_deterministic_and_roundtrips(tmp_path) -> None:
    first = _generate()
    second = _generate()
    assert first == second
    path = write_scenario_bank(tmp_path / "bank.json", first)
    loaded = load_scenario_bank(path, expected_runtime_config_sha256="a" * 64)
    assert loaded == first
    assert len(scenario_bank_sha256(path)) == 64


def test_fixed_scenario_bank_repeats_demo_layout() -> None:
    bank = _generate(count=64, mode="fixed")
    assert bank["mode"] == "fixed"
    for index, scenario in enumerate(bank["scenarios"]):
        assert scenario["id"] == index
        assert scenario["object_position_m"] == [0.30, 0.00, 0.015]
        assert scenario["target_position_m"] == [0.30, 0.30, 0.015]


def test_scenario_bank_rejects_runtime_mismatch(tmp_path) -> None:
    path = write_scenario_bank(tmp_path / "bank.json", _generate())
    with pytest.raises(ArtifactError, match="runtime config hash"):
        load_scenario_bank(path, expected_runtime_config_sha256="b" * 64)


def test_scenario_bank_rejects_noncontiguous_ids(tmp_path) -> None:
    bank = _generate()
    bank["scenarios"][0]["id"] = 99
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ArtifactError, match="contiguous"):
        load_scenario_bank(path)
