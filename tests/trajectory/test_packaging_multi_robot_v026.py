from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.cli.packaging import _backends, _model_and_hashes, _sdk_evidence_path
from ufactory.config import ConfigError, load_runtime_config
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.trajectory.packaging import (
    build_packaging_program,
    packaging_layout,
    packaging_obstacles,
    packaging_scene_sha256,
)
from ufactory.trajectory.preflight import create_safety_gate


ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")
G2_ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850")


@pytest.mark.parametrize("robot", ROBOTS)
def test_packaging_profiles_are_available_for_all_supported_robots(robot: str) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")
    layout = packaging_layout(config)

    assert get_robot_runtime_profile(robot).task.showcase_supported is True
    assert config.gripper is not None
    assert layout.object_size_m == pytest.approx((0.030, 0.030, 0.030))
    assert layout.object_mass_kg == pytest.approx(0.017)
    assert layout.simulation_substeps == 32


@pytest.mark.parametrize("robot", G2_ROBOTS)
def test_g2_packaging_profiles_preserve_reference_geometry(robot: str) -> None:
    layout = packaging_layout(load_runtime_config(robot, task="packaging_showcase"))

    assert layout.object_position_m == pytest.approx((0.300, 0.000, 0.015))
    assert layout.target_position_m == pytest.approx((0.300, 0.300, 0.018))
    assert layout.grasp_link6_z_m == pytest.approx(0.1871)
    assert layout.release_link6_z_m == pytest.approx(0.3901)


def test_lite6_packaging_profile_uses_lite6_gripper_geometry_and_safe_box_center() -> None:
    config = load_runtime_config("lite6", task="packaging_showcase")
    layout = packaging_layout(config)

    assert "assets/configs/runtime/tasks/robots/lite6_packaging_showcase.yaml" in config.sources
    assert layout.object_position_m == pytest.approx((0.200, 0.000, 0.015))
    assert layout.target_position_m == pytest.approx((0.200, 0.220, 0.018))
    assert layout.box_center_xy_m == pytest.approx((0.200, 0.220))
    assert layout.home_position_m == pytest.approx((0.200, 0.000, 0.200))
    assert layout.grasp_gap_m == pytest.approx(0.020)
    assert layout.grasp_link6_z_m == pytest.approx(0.0963)
    assert layout.release_link6_z_m == pytest.approx(0.2993)
    assert layout.simulation_pre_release_relax_duration_s == pytest.approx(0.0)
    assert layout.simulation_release_duration_s == pytest.approx(0.500)


@pytest.mark.parametrize(
    ("robot", "expected"),
    [("xarm6", True), ("lite6", True), ("xarm5", False), ("xarm7", False), ("uf850", False)],
)
def test_real_packaging_capability_matches_installed_end_effector(robot: str, expected: bool) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")

    assert config.gripper is not None
    assert config.gripper.real_command is expected


@pytest.mark.parametrize("robot", ROBOTS)
def test_packaging_program_transits_to_configured_target(robot: str) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")
    layout = packaging_layout(config)
    program = build_packaging_program(config)
    transit = next(segment for segment in program.segments if segment.label == "transit")

    assert transit.pose_end is not None
    assert transit.pose_end == pytest.approx(
        (layout.target_position_m[0], layout.target_position_m[1], layout.release_link6_z_m)
    )


def test_box_floor_top_is_consumed_as_physical_floor_geometry() -> None:
    config = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={
            "task.parameters.box_floor_top_z_m": 0.006,
            "task.parameters.fixed_target_position_m": [0.300, 0.300, 0.021],
        },
    )
    floor = next(obstacle for obstacle in packaging_obstacles(packaging_layout(config)) if obstacle.name == "box_floor")

    assert floor.size_m[2] == pytest.approx(0.006)
    assert floor.center_m[2] == pytest.approx(0.003)


def test_box_floor_thickness_is_relative_to_nonzero_table_height() -> None:
    config = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={
            "task.parameters.table_center_m": [0.337, 0.000, 0.180],
            "task.parameters.fixed_object_position_m": [0.300, 0.000, 0.215],
            "task.parameters.box_floor_top_z_m": 0.203,
            "task.parameters.fixed_target_position_m": [0.300, 0.300, 0.218],
        },
    )
    layout = packaging_layout(config)
    floor = next(obstacle for obstacle in packaging_obstacles(layout) if obstacle.name == "box_floor")

    assert layout.table_top_z_m == pytest.approx(0.200)
    assert floor.size_m[2] == pytest.approx(0.003)
    assert floor.center_m[2] == pytest.approx(0.2015)


def test_packaging_scene_hash_covers_effective_gripper_configuration() -> None:
    baseline = load_runtime_config("xarm6", task="packaging_showcase")
    assert baseline.gripper is not None
    tuned = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={"gripper.kp": baseline.gripper.kp + 1.0},
    )

    assert packaging_scene_sha256(baseline) != packaging_scene_sha256(tuned)


@pytest.mark.parametrize("robot", ROBOTS)
def test_servo_cartesian_packaging_preflight_passes_for_all_robots(robot: str) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")
    urdf, _urdf_hash, calibration_hash = _model_and_hashes(config, None, None)
    kinematics, collision = _backends(config, urdf)
    program = build_packaging_program(config, kinematics=kinematics)
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=collision,
        calibration_sha256=calibration_hash,
        scene_sha256=packaging_scene_sha256(config),
        urdf_path=urdf,
    )

    report = gate.preflight(program, executor="servo_cartesian")

    assert report.passed, [violation.message for violation in report.violations[:10]]


@pytest.mark.parametrize(
    ("robot", "body", "message"),
    [
        ("xarm6", "grasp_gap_m: 0.100", "gripper gap range"),
        ("xarm6", "simulation_pre_release_relax_gap_m: 0.100", "gripper gap range"),
        ("lite6", "grasp_gap_m: 0.019", "gripper gap range"),
        ("xarm6", "box_outer_size_m: [0.040, 0.040, 0.150]", "safety margin"),
        ("xarm6", "fixed_target_position_m: [0.300, 0.300, 0.019]", "fixed target Z"),
        ("xarm6", "box_center_xy_m: [0.000, 0.000]", "outside the box opening"),
        ("xarm6", "table_size_m: [0.100, 0.100, 0.040]", "outside the table"),
    ],
)
def test_invalid_packaging_yaml_is_rejected(
    tmp_path: Path,
    robot: str,
    body: str,
    message: str,
) -> None:
    path = tmp_path / "invalid_packaging.yaml"
    path.write_text(f"schema_version: 1\ntask:\n  parameters:\n    {body}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_runtime_config(robot, task="packaging_showcase", config_path=path)


@pytest.mark.parametrize("robot", ("xarm5", "xarm7", "uf850"))
def test_real_packaging_rejects_profiles_without_enabled_real_gripper(
    monkeypatch: pytest.MonkeyPatch,
    robot: str,
) -> None:
    from ufactory.cli import packaging

    connected = False

    def fail_if_connected(_ip: str) -> None:
        nonlocal connected
        connected = True
        raise AssertionError("controller connection must not be attempted")

    monkeypatch.setattr(packaging, "_connect", fail_if_connected)

    with pytest.raises(SystemExit):
        packaging.main(["--robot", robot, "--mode", "real", "--confirm-real"])
    assert connected is False


def test_packaging_sdk_report_name_is_robot_specific() -> None:
    assert _sdk_evidence_path(None, "xarm6") == Path("reports/sdk_sim_xarm6_packaging.json")
    assert _sdk_evidence_path(None, "lite6") == Path("reports/sdk_sim_lite6_packaging.json")
    explicit = Path("reports/custom.json")
    assert _sdk_evidence_path(explicit, "lite6") is explicit


def test_packaging_sim_forwarding_preserves_selected_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import ModuleType
    import sys

    from ufactory.cli import packaging

    received: list[str] = []
    fake = ModuleType("_packaging_showcase")
    fake.main = lambda argv: received.extend(argv) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_packaging_showcase", fake)

    class Args:
        robot = "lite6"
        speed = 1.0
        executor = "servo_cartesian"
        cycles = 3
        loop = None
        table_height = None
        config = None
        capture_keyframes = False

    assert packaging._run_sim(Args()) == 0
    assert received[:2] == ["--robot", "lite6"]
    assert received[-2:] == ["--cycles", "3"]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("executor", ("servo_j", "servo_cartesian"))
def test_multi_robot_packaging_physics_single_cycle(robot: str, executor: str) -> None:
    from examples._packaging_scene import build_packaging_scene
    from examples.xarm6.xarm6_g2_showcase import (
        _cycle_failure_reason,
        init_showcase_robot,
        prepare_packaging_cycle,
        run_pick_place_cycle,
        stiffen_gripper_mimic_constraints,
    )
    from ufactory.simulation import GenesisRuntimeManager

    config = load_runtime_config(robot, task="packaging_showcase")
    task_layout = packaging_layout(config)
    with GenesisRuntimeManager(config.simulation):
        scene, robot_entity, block, display_layout = build_packaging_scene(
            show_viewer=False,
            runtime_config=config,
        )
        stiffen_gripper_mimic_constraints(robot_entity)
        context = init_showcase_robot(
            robot_entity,
            display_layout,
            scene,
            runtime_config=config,
        )
        prepare_packaging_cycle(scene, robot_entity, block, display_layout, context)
        report = run_pick_place_cycle(
            scene,
            robot_entity,
            block,
            display_layout,
            ctx=context,
            robot_key=config.robot.key,
            executor=executor,
        )

    assert report is not None
    assert _cycle_failure_reason(report, display_layout, task_layout) is None
