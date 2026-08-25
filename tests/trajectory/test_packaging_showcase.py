from __future__ import annotations

import math
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ufactory.config import load_runtime_config, resolve_manipulation_object_spec
from ufactory.manipulation.packaging import (
    build_packaging_program,
    packaging_layout,
    packaging_obstacles,
    validate_payload_box_clearance,
)
from ufactory.trajectory.scene import TrajSceneContext
from ufactory.trajectory.sim_executor import PhaseStatus, SimReport


def test_packaging_config_is_versioned_in_robot_base_frame() -> None:
    config = load_runtime_config("xarm6", task="packaging_showcase")
    grasp_config = load_runtime_config("xarm6")
    layout = packaging_layout(config)

    assert config.task.name == "packaging_showcase"
    assert resolve_manipulation_object_spec(config).size_m == pytest.approx((0.03, 0.03, 0.03))
    assert resolve_manipulation_object_spec(config) == resolve_manipulation_object_spec(grasp_config)
    assert layout.object_position_m == pytest.approx((0.300, 0.0, 0.015))
    assert layout.home_position_m == pytest.approx((0.300, 0.0, 0.300))
    assert layout.object_position_m == pytest.approx(grasp_config.task.parameters["fixed_object_position_m"])
    assert layout.home_position_m == pytest.approx(grasp_config.task.parameters["default_ee_position_m"])
    assert layout.home_position_m[:2] == pytest.approx(layout.object_position_m[:2])
    assert layout.simulation_grasp_center_compensation_xy_m == pytest.approx((0.0025, 0.0))
    assert layout.simulation_arm_kp_scale == pytest.approx(1.5)
    assert layout.target_position_m == pytest.approx((0.300, 0.300, 0.018))
    assert layout.target_position_m[:2] == pytest.approx(grasp_config.task.parameters["fixed_target_position_m"][:2])
    assert layout.box_center_xy_m == pytest.approx(layout.target_position_m[:2])
    assert layout.box_outer_size_m == pytest.approx((0.300, 0.200, 0.150))
    assert layout.box_outer_size_m[0] > layout.box_outer_size_m[1]
    assert layout.release_object_bottom_clearance_m == pytest.approx(0.05)
    assert layout.release_object_center_m[2] - layout.object_size_m[2] / 2 == pytest.approx(layout.box_rim_z_m + 0.05)


def test_packaging_program_uses_shared_lspb_phases_without_box_descent() -> None:
    config = load_runtime_config("xarm6", task="packaging_showcase")
    layout = packaging_layout(config)
    program = build_packaging_program(config)
    labels = [segment.label for segment in program.segments]

    assert labels == [
        "home->pregrasp",
        "descend",
        "grip",
        "grip-settle",
        "lift",
        "transit",
        "pre-release-settle",
        "release",
        "post-release-settle",
        "return-transit",
        "return-home",
    ]
    assert "place-descend" not in labels
    approach = program.segments[labels.index("home->pregrasp")]
    assert approach.pose_start is not None
    assert approach.pose_end is not None
    assert approach.pose_start[:2] == pytest.approx(approach.pose_end[:2])
    transit = program.segments[labels.index("transit")]
    assert transit.pose_end is not None
    assert transit.pose_end == pytest.approx(
        (layout.box_center_xy_m[0], layout.box_center_xy_m[1], layout.release_link6_z_m)
    )
    grip = program.segments[labels.index("grip")]
    assert grip.gap_start == pytest.approx(0.084)
    assert grip.gap_end == pytest.approx(0.022)

    compensated = build_packaging_program(config, cartesian_xy_offset_m=(0.0025, 0.0))
    compensated_approach = compensated.segments[labels.index("home->pregrasp")]
    compensated_transit = compensated.segments[labels.index("transit")]
    assert compensated_approach.pose_start is not None
    assert compensated_approach.pose_end is not None
    assert compensated_approach.pose_start[:2] == pytest.approx((0.3025, 0.0))
    assert compensated_approach.pose_end[:2] == pytest.approx((0.3025, 0.0))
    assert compensated_transit.pose_end is not None
    assert compensated_transit.pose_end[:2] == pytest.approx((0.3025, 0.300))

    centered_place = build_packaging_program(
        config,
        cartesian_xy_offset_m=(0.0025, 0.0),
        place_xy_offset_m=(0.0, 0.0),
    )
    centered_lift = centered_place.segments[labels.index("lift")]
    centered_transit = centered_place.segments[labels.index("transit")]
    centered_return = centered_place.segments[labels.index("return-transit")]
    assert centered_lift.pose_end is not None
    assert centered_lift.pose_end[:2] == pytest.approx((0.3025, 0.0))
    assert centered_transit.pose_end is not None
    assert centered_transit.pose_end[:2] == pytest.approx((0.300, 0.300))
    assert centered_return.pose_end is not None
    assert centered_return.pose_end[:2] == pytest.approx((0.3025, 0.0))


def test_packaging_display_rotates_base_x_long_box_to_world_y() -> None:
    from ufactory.manipulation.packaging.scene import ROBOT_XY, TABLE_ORIGIN_X, TABLE_TOP_SIZE, make_layout

    display_layout = make_layout()
    expected_center = (ROBOT_XY[0] - 0.300, ROBOT_XY[1] + 0.300)

    assert display_layout.box_center_xy == pytest.approx(expected_center)
    assert display_layout.place_xy == pytest.approx(expected_center)
    assert display_layout.box_outer == pytest.approx((0.200, 0.300, 0.150))
    assert display_layout.box_outer[1] > display_layout.box_outer[0]

    table_min_x = TABLE_ORIGIN_X
    table_max_x = TABLE_ORIGIN_X + TABLE_TOP_SIZE[0]
    table_half_y = TABLE_TOP_SIZE[1] / 2.0
    assert display_layout.box_center_xy[0] - display_layout.box_outer[0] / 2.0 >= table_min_x
    assert display_layout.box_center_xy[0] + display_layout.box_outer[0] / 2.0 <= table_max_x
    assert abs(display_layout.box_center_xy[1]) + display_layout.box_outer[1] / 2.0 <= table_half_y


def test_packaging_robot_import_preserves_glb_pbr_materials(monkeypatch: pytest.MonkeyPatch) -> None:
    from ufactory.manipulation.packaging import scene as packaging_scene

    events: list[str] = []
    fallback_surface = object()
    robot = object()

    class _PbrScope:
        def __enter__(self):
            events.append("pbr-enter")

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            events.append("pbr-exit")

    class _Scene:
        def add_entity(self, morph, *, surface):
            assert events == ["pbr-enter"]
            assert morph["file"] == "calibrated-xarm6.urdf"
            assert surface is fallback_surface
            events.append("add-robot")
            return robot

    monkeypatch.setattr(packaging_scene, "glb_pbr_surfaces", _PbrScope)
    monkeypatch.setattr(packaging_scene, "glb_view_surface", lambda: fallback_surface)
    monkeypatch.setattr(packaging_scene.gs.morphs, "URDF", lambda **kwargs: kwargs)
    layout = SimpleNamespace(robot_xy=(0.65, -0.337), table_top_z=0.75)

    result = packaging_scene._add_packaging_robot(_Scene(), layout, "calibrated-xarm6.urdf")

    assert result is robot
    assert events == ["pbr-enter", "add-robot", "pbr-exit"]


def test_display_home_is_directly_above_cube_center() -> None:
    from ufactory.manipulation.packaging.scene import make_layout
    from ufactory.manipulation.packaging.simulation import _world_home

    display_layout = make_layout()
    home_world = _world_home(display_layout)

    assert home_world[0] == pytest.approx(display_layout.obj_spawn_xy[0])
    assert home_world[1] == pytest.approx(display_layout.obj_spawn_xy[1] + 0.0025)
    assert home_world[2] == pytest.approx(display_layout.table_top_z + 0.300)


def test_packaging_obstacles_include_hollow_box_and_object() -> None:
    layout = packaging_layout(load_runtime_config("xarm6", task="packaging_showcase"))
    obstacles = packaging_obstacles(layout)
    names = {obstacle.name for obstacle in obstacles}

    assert names == {
        "table",
        "box_floor",
        "box_wall_x_min",
        "box_wall_x_max",
        "box_wall_y_min",
        "box_wall_y_max",
        "object",
    }
    validate_payload_box_clearance(layout, margin_m=0.005)


def test_payload_clearance_rejects_excessive_margin() -> None:
    layout = packaging_layout(load_runtime_config("xarm6", task="packaging_showcase"))
    with pytest.raises(ValueError, match="does not fit"):
        validate_payload_box_clearance(layout, margin_m=0.2)


def test_scene_context_base_yaw_round_trip() -> None:
    ctx = TrajSceneContext.__new__(TrajSceneContext)
    ctx.base_pos_world = (0.65, -0.337, 0.75)
    ctx.base_yaw_rad = math.pi / 2

    world = ctx.base_to_world((0.337, 0.0, 0.015))
    assert world == pytest.approx((0.65, 0.0, 0.765))
    assert ctx.world_to_base(world) == pytest.approx((0.337, 0.0, 0.015))


def test_packaging_cli_print_config_does_not_start_simulation(capsys: pytest.CaptureFixture[str]) -> None:
    from ufactory.cli.packaging import main

    assert main(["--print-config"]) == 0
    output = capsys.readouterr().out
    assert "name: packaging_showcase" in output
    assert "release_object_bottom_clearance_m: 0.05" in output


def test_packaging_real_requires_confirmation_before_connection() -> None:
    from ufactory.cli.packaging import main

    with pytest.raises(SystemExit):
        main(["--mode", "real", "--executor", "servo_j", "--ip", "192.0.2.1"])


@pytest.mark.parametrize(
    ("repeat_args", "expected_cycles", "expected_loop"),
    [
        ([], None, None),
        (["--cycles", "3"], 3, None),
        (["--loop"], None, True),
        (["--no-loop"], None, False),
    ],
)
def test_packaging_cli_parses_simulation_repetition(
    monkeypatch: pytest.MonkeyPatch,
    repeat_args: list[str],
    expected_cycles: int | None,
    expected_loop: bool | None,
) -> None:
    from ufactory.cli import packaging

    captured: dict[str, object] = {}

    def _fake_run_sim(args) -> int:
        captured["cycles"] = args.cycles
        captured["loop"] = args.loop
        return 0

    monkeypatch.setattr(packaging, "_run_sim", _fake_run_sim)

    assert packaging.main(["--mode", "sim", *repeat_args]) == 0
    assert captured == {"cycles": expected_cycles, "loop": expected_loop}


@pytest.mark.parametrize("value", ["0", "-1"])
def test_packaging_cli_rejects_non_positive_cycles(value: str) -> None:
    from ufactory.cli.packaging import main

    with pytest.raises(SystemExit):
        main(["--mode", "sim", "--cycles", value])


@pytest.mark.parametrize("flag", ["--loop", "--no-loop"])
def test_packaging_cli_rejects_conflicting_cycle_flags(flag: str) -> None:
    from ufactory.cli.packaging import main

    with pytest.raises(SystemExit):
        main(["--mode", "sim", "--cycles", "2", flag])


def test_packaging_cli_rejects_cycles_outside_simulation() -> None:
    from ufactory.cli.packaging import main

    with pytest.raises(SystemExit):
        main(["--mode", "dry-run", "--cycles", "2"])


def test_packaging_cli_rejects_infinite_loop_outside_simulation() -> None:
    from ufactory.cli.packaging import main

    with pytest.raises(SystemExit):
        main(["--mode", "dry-run", "--loop"])


def test_packaging_cli_accepts_compatibility_no_loop_outside_simulation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ufactory.cli.packaging import main

    assert main(["--mode", "dry-run", "--no-loop", "--print-config"]) == 0
    assert "name: packaging_showcase" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cycles", "loop", "expected_tail"),
    [
        (None, None, []),
        (3, None, ["--cycles", "3"]),
        (None, True, ["--loop"]),
        (None, False, ["--no-loop"]),
    ],
)
def test_packaging_cli_forwards_repetition_to_showcase(
    monkeypatch: pytest.MonkeyPatch,
    cycles: int | None,
    loop: bool | None,
    expected_tail: list[str],
) -> None:
    from ufactory.cli import packaging

    received: list[str] = []

    def _fake_main(argv: list[str]) -> int:
        received.extend(argv)
        return 0

    monkeypatch.setattr(packaging, "_packaging_simulation_main", _fake_main)
    args = SimpleNamespace(
        speed=1.0,
        robot="xarm6",
        executor="servo_j",
        cycles=cycles,
        loop=loop,
        table_height=None,
        config=None,
        backend=None,
        capture_keyframes=False,
    )

    assert packaging._run_sim(args) == 0
    assert received == ["--robot", "xarm6", "--speed", "1.0", "--executor", "servo_j", *expected_tail]


def test_packaging_showcase_resolves_default_and_explicit_repetition() -> None:
    from ufactory.manipulation.packaging.simulation import _cycle_indices, _resolve_repetition

    assert _resolve_repetition(None, None) == (False, 1)
    assert _resolve_repetition(3, None) == (False, 3)
    assert _resolve_repetition(None, False) == (False, 1)
    assert _resolve_repetition(None, True) == (True, 1)
    assert list(_cycle_indices(infinite=False, cycle_limit=1)) == [1]
    assert list(_cycle_indices(infinite=False, cycle_limit=3)) == [1, 2, 3]
    assert list(islice(_cycle_indices(infinite=True, cycle_limit=1), 4)) == [1, 2, 3, 4]


def test_packaging_cycle_reset_restores_position_orientation_and_velocity(monkeypatch: pytest.MonkeyPatch) -> None:
    from ufactory.manipulation.packaging import simulation as showcase

    monkeypatch.setattr(showcase.gs, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(showcase.gs, "tc_float", torch.float32, raising=False)

    class _Block:
        def __init__(self) -> None:
            self.pos_call = None
            self.quat_call = None

        def set_pos(self, value, *, zero_velocity: bool) -> None:
            self.pos_call = (value, zero_velocity)

        def set_quat(self, value, *, zero_velocity: bool) -> None:
            self.quat_call = (value, zero_velocity)

    block = _Block()
    layout = SimpleNamespace(obj_spawn_xy=(0.65, -0.037), table_top_z=0.75, obj_spawn_center_z=0.015)

    showcase._reset_block(block, layout)

    assert block.pos_call is not None
    assert block.quat_call is not None
    assert block.pos_call[0][0].tolist() == pytest.approx([0.65, -0.037, 0.765])
    assert block.quat_call[0][0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert block.pos_call[1] is True
    assert block.quat_call[1] is True


def test_packaging_cycle_preparation_clears_release_control_and_restores_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ufactory.manipulation.packaging import simulation as showcase

    monkeypatch.setattr(showcase.gs, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(showcase.gs, "tc_float", torch.float32, raising=False)
    held: dict[str, object] = {}

    class _Robot:
        def __init__(self) -> None:
            self.force_call = None

        def control_dofs_force(self, value, indices) -> None:
            self.force_call = (value, indices)

    class _Block:
        def __init__(self) -> None:
            self.pos_call = None
            self.quat_call = None

        def set_pos(self, value, *, zero_velocity: bool) -> None:
            self.pos_call = (value, zero_velocity)

        def set_quat(self, value, *, zero_velocity: bool) -> None:
            self.quat_call = (value, zero_velocity)

    def _fake_hold(robot, scene, ctx, *, steps: int) -> None:
        held.update(robot=robot, scene=scene, ctx=ctx, steps=steps)

    monkeypatch.setattr(showcase, "hold_robot_home", _fake_hold)
    robot = _Robot()
    block = _Block()
    scene = object()
    ctx = SimpleNamespace(gripper_dof_idx=[10], all_gripper_dof_idx=[10, 11, 12, 13, 14, 15])
    layout = SimpleNamespace(obj_spawn_xy=(0.65, -0.037), table_top_z=0.75, obj_spawn_center_z=0.015)

    showcase.prepare_packaging_cycle(scene, robot, block, layout, ctx, speed=1.0)

    assert robot.force_call is not None
    assert robot.force_call[0].tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0]]
    assert robot.force_call[1] == [11, 12, 13, 14, 15]
    assert block.pos_call is not None and block.pos_call[1] is True
    assert block.quat_call is not None and block.quat_call[1] is True
    assert held == {"robot": robot, "scene": scene, "ctx": ctx, "steps": showcase.SETTLE_STEPS}


def test_packaging_sim_skips_preload_relax_and_steps_release_target() -> None:
    from ufactory.manipulation.packaging.simulation import _with_sim_release_timing
    from ufactory.trajectory.segments import Program, Segment

    grip = Segment(
        kind="gripper",
        duration=2.0,
        v_max=0.0,
        a_max=0.5,
        label="grip",
        gap_start=0.084,
        gap_end=0.022,
        samples_count=100,
    )
    pre_release = Segment(
        kind="gripper",
        duration=0.5,
        v_max=0.0,
        a_max=0.5,
        label="pre-release-settle",
        gap_start=0.022,
        gap_end=0.022,
        samples_count=25,
    )
    release = Segment(
        kind="gripper",
        duration=2.0,
        v_max=0.0,
        a_max=0.5,
        label="release",
        gap_start=0.022,
        gap_end=0.084,
        samples_count=100,
    )
    program = Program(rate=50.0, segments=[grip, pre_release, release])

    tuned = _with_sim_release_timing(program, speed=1.0)

    assert program.segments[1].gap_end == pytest.approx(0.022)
    assert program.segments[2].duration == pytest.approx(2.0)
    assert tuned.segments[0] is grip
    assert tuned.segments[1] is pre_release
    assert len(tuned.segments) == 3
    assert tuned.segments[2].label == "release"
    assert tuned.segments[2].gap_start == pytest.approx(0.022)
    assert tuned.segments[2].gap_end == pytest.approx(0.084)
    assert tuned.segments[2].duration == pytest.approx(0.02)
    assert tuned.segments[2].samples_count == 1


@pytest.mark.parametrize(
    ("lift_z_mm", "place_error_mm", "home_drift_mm", "expected"),
    [
        (900.0, 5.0, 2.0, None),
        (800.0, 5.0, 2.0, "grasp/lift failed"),
        (900.0, 30.0, 2.0, "place error"),
        (900.0, 5.0, 12.0, "home drift"),
    ],
)
def test_packaging_cycle_failure_reason(
    lift_z_mm: float,
    place_error_mm: float,
    home_drift_mm: float,
    expected: str | None,
) -> None:
    from ufactory.manipulation.packaging.simulation import _cycle_failure_reason

    report = SimReport(
        phases=[PhaseStatus("lift", 0, 1, 0.02, 1.0, 1.0, obj_pos_mm=(0.0, 0.0, lift_z_mm))],
        place_error_mm=place_error_mm,
        home_drift_mm=home_drift_mm,
    )
    display_layout = SimpleNamespace(table_top_z=0.75)
    task_layout = SimpleNamespace(place_success_distance_m=0.025)

    reason = _cycle_failure_reason(report, display_layout, task_layout)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
def test_packaging_physics_grasp_and_drop_regression() -> None:
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.manipulation.packaging.simulation import (
        _trajectory_context,
        _with_sim_release_timing,
        init_showcase_robot,
        prepare_packaging_cycle,
    )
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
    from ufactory.trajectory.sim_executor import replay_sim
    from ufactory.visualization.glb import enable_glb_pbr_surfaces

    config = load_runtime_config("xarm6", task="packaging_showcase")
    enable_glb_pbr_surfaces()
    with GenesisRuntimeManager(config.simulation):
        scene, robot, block, display_layout = build_packaging_scene(show_viewer=False)
        showcase_ctx = init_showcase_robot(robot, display_layout, scene)
        prepare_packaging_cycle(scene, robot, block, display_layout, showcase_ctx)
        traj_ctx = _trajectory_context(scene, robot, block, display_layout, showcase_ctx, config)
        source = build_packaging_program(
            config,
            q_home=traj_ctx.home_qpos[traj_ctx.arm_dof_idx],
            cartesian_xy_offset_m=packaging_layout(config).simulation_grasp_center_compensation_xy_m,
            place_xy_offset_m=(0.0, 0.0),
        )
        program = compile_cartesian_program_to_joint_stream(source, traj_ctx)
        program = _with_sim_release_timing(program, speed=1.0)

        release_start: dict[str, float] = {}
        release_ticks: list[tuple[int, float, float]] = []
        horizontal_center_errors_m: dict[str, float] = {}
        bilateral_contact: dict[str, bool] = {}

        def _finger_span() -> float:
            left = traj_ctx.left_finger.get_pos()[0]
            right = traj_ctx.right_finger.get_pos()[0]
            return max(abs(float(left[0] - right[0])), abs(float(left[1] - right[1])))

        def _on_phase(_status, segment) -> None:
            if segment.label in {"home->pregrasp", "grip-settle"}:
                obj = block.get_pos()[0]
                finger_mid = (traj_ctx.left_finger.get_pos()[0] + traj_ctx.right_finger.get_pos()[0]) / 2.0
                horizontal_center_errors_m[segment.label] = math.hypot(
                    float(finger_mid[0] - obj[0]),
                    float(finger_mid[1] - obj[1]),
                )
            if segment.label == "grip-settle":
                contacts = block.get_contacts(with_entity=robot)
                valid = contacts["valid_mask"][0]
                link_a = contacts["link_a"][0]
                link_b = contacts["link_b"][0]
                for name, finger in (("left", traj_ctx.left_finger), ("right", traj_ctx.right_finger)):
                    finger_contact = valid & ((link_a == finger.idx) | (link_b == finger.idx))
                    bilateral_contact[name] = bool(torch.any(finger_contact).item())
            if segment.label == "transit":
                release_start["object_z"] = float(block.get_pos()[0][2])
                release_start["finger_span"] = _finger_span()
            if segment.label == "post-release-settle":
                release_start["settled_finger_span"] = _finger_span()

        def _on_tick(segment, tick: int) -> None:
            if segment.label in {
                "pre-release-settle",
                "pre-release-relax",
                "release",
                "post-release-settle",
            }:
                release_ticks.append((tick, float(block.get_pos()[0][2]), _finger_span()))

        report = replay_sim(
            program,
            traj_ctx,
            stabilize_grasp_weld=False,
            on_phase=_on_phase,
            on_tick=_on_tick,
        )

        phases = {phase.label: phase for phase in report.phases}
        table_top_mm = display_layout.table_top_z * 1000.0
        assert phases["lift"].obj_pos_mm[2] > table_top_mm + 100.0
        transit_obj_xy = np.asarray(phases["transit"].obj_pos_mm[:2]) / 1000.0
        assert np.linalg.norm(transit_obj_xy - np.asarray(display_layout.box_center_xy)) < 0.025
        assert report.home_drift_mm < 10.0
        assert max(phase.eside_arm_mm for phase in report.phases) < 20.0
        assert horizontal_center_errors_m["home->pregrasp"] < 0.001
        assert horizontal_center_errors_m["grip-settle"] < 0.001
        assert bilateral_contact == {"left": True, "right": True}
        assert release_ticks
        start_z = release_start["object_z"]
        start_span = release_start["finger_span"]
        first_drop = next(sample for sample in release_ticks if sample[1] < start_z - 0.002)
        assert first_drop[2] > start_span + 0.003
        assert release_start["settled_finger_span"] > start_span + 0.050
        final = np.asarray(phases["post-release-settle"].obj_pos_mm) / 1000.0
        half_x = display_layout.box_outer[0] / 2.0 - display_layout.box_wall
        half_y = display_layout.box_outer[1] / 2.0 - display_layout.box_wall
        assert abs(final[0] - display_layout.box_center_xy[0]) < half_x
        assert abs(final[1] - display_layout.box_center_xy[1]) < half_y
