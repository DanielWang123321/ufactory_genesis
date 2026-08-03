from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.config import (
    ConfigError,
    GraspObjectSpec,
    RepositoryAssetStore,
    dump_runtime_config,
    load_runtime_config,
    resolve_pick_place_object_spec,
)


ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")


def test_all_default_robot_configs_are_stable_and_assets_exist():
    store = RepositoryAssetStore.discover()
    first = {name: load_runtime_config(name).sha256 for name in ROBOTS}
    second = {name: load_runtime_config(name).sha256 for name in ROBOTS}
    assert first == second
    assert len(set(first.values())) == len(ROBOTS)
    store.validate_manifest()


def test_unknown_config_field_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nmotion:\n  mystery: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_runtime_config("xarm6", config_path=path)


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_non_finite_config_is_rejected(tmp_path: Path, value: str):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"schema_version: 1\nmotion:\n  joint_speed_rad_s: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="finite"):
        load_runtime_config("xarm6", config_path=path)


def test_cli_scalar_override_changes_hash_and_xarm_ip_does_not(monkeypatch: pytest.MonkeyPatch):
    base = load_runtime_config("xarm6")
    changed = load_runtime_config("xarm6", overrides={"motion.rate_hz": 100.0})
    monkeypatch.setenv("XARM_IP", "192.0.2.10")
    with_ip = load_runtime_config("xarm6")
    assert changed.sha256 != base.sha256
    assert with_ip.sha256 == base.sha256


def test_dump_config_contains_sources_and_hash_without_importing_genesis():
    text = dump_runtime_config(load_runtime_config("xarm6"))
    assert "schema_version: 1" in text
    assert "sources:" in text
    assert "sha256:" in text


def test_default_rigid_contact_profile_is_explicit_genesis_131() -> None:
    simulation = load_runtime_config("xarm6").simulation
    assert simulation.constraint_solver == "newton"
    assert simulation.friction_cone == "elliptic"
    assert simulation.contact_resolution == "signorini"
    assert simulation.solver_iterations == 100
    assert simulation.noslip_iterations == 0
    assert simulation.constraint_time_constant_s == pytest.approx(0.005)


@pytest.mark.parametrize(
    "body,match",
    [
        ("friction_cone: future", "friction cone"),
        ("contact_resolution: future", "contact resolution"),
        ("constraint_solver: cg", "constraint solver"),
        ("noslip_iterations: 5", "elliptic friction"),
        ("friction_cone: pyramidal", "signorini contact"),
    ],
)
def test_invalid_rigid_contact_configuration_is_rejected(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / "bad_physics.yaml"
    path.write_text(f"schema_version: 1\nsimulation:\n  {body}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_runtime_config("xarm6", config_path=path)


@pytest.mark.parametrize("robot", ROBOTS)
def test_all_robots_share_the_30mm_17g_table_object(robot: str):
    config = load_runtime_config(robot)
    spec = resolve_pick_place_object_spec(config)
    assert spec.size_m == pytest.approx((0.030, 0.030, 0.030))
    assert spec.mass_kg == pytest.approx(0.017)
    assert spec.rest_center_z_m == pytest.approx(0.015)
    params = config.task.parameters
    for name in (
        "fixed_object_position_m",
        "fixed_target_position_m",
        "object_spawn_lower_m",
        "object_spawn_upper_m",
        "target_spawn_lower_m",
        "target_spawn_upper_m",
    ):
        assert float(params[name][2]) == pytest.approx(0.015)


def test_grasp_object_spec_defensively_freezes_values():
    mutable_size = [0.030, 0.030, 0.030]
    spec = GraspObjectSpec(mutable_size, 0.017)  # type: ignore[arg-type]
    mutable_size[2] = 1.0

    assert spec.size_m == pytest.approx((0.030, 0.030, 0.030))
    assert spec.rest_center_z_m == pytest.approx(0.015)


@pytest.mark.parametrize(
    "body,match",
    [
        ("object_size_m: [0.03, 0.03]", "expected 3 values"),
        ("object_size_m: [0.03, -0.03, 0.03]", "must be positive"),
        ("object_mass_kg: 0", "must be finite and positive"),
        ("fixed_object_position_m: [0.3, 0.0]", "expected 3 values"),
        ("default_ee_position_m: [.nan, 0.0, 0.3]", "must be finite"),
    ],
)
def test_invalid_grasp_object_configuration_is_rejected(tmp_path: Path, body: str, match: str):
    path = tmp_path / "bad_object.yaml"
    path.write_text(f"schema_version: 1\ntask:\n  parameters:\n    {body}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_runtime_config("xarm6", config_path=path)


def test_object_mass_override_changes_config_hash():
    base = load_runtime_config("xarm6")
    changed = load_runtime_config("xarm6", overrides={"task.parameters.object_mass_kg": 0.018})
    assert resolve_pick_place_object_spec(changed).mass_kg == pytest.approx(0.018)
    assert changed.sha256 != base.sha256
