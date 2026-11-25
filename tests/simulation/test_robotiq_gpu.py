"""Representative Robotiq 2F-85 and 2F-140 GPU acceptance: pick-place, faults, packaging and batch control."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PICK_CASES = [
    ("xarm6", "robotiq_2f_85", "servo_cartesian", 50, 50),
    ("xarm6", "robotiq_2f_85", "servo_j", 10, 100),
    ("xarm6", "robotiq_2f_140", "servo_cartesian", 50, 50),
    ("xarm6", "robotiq_2f_140", "servo_j", 90, 10),
    ("xarm5", "robotiq_2f_85", "servo_j", 50, 50),
    ("xarm7", "robotiq_2f_140", "servo_cartesian", 50, 50),
    ("uf850", "robotiq_2f_85", "servo_cartesian", 50, 50),
    # The 2F-140 home sits higher; Genesis's default-tolerance home IK stopped 0.445 / 0.26 mm off it here.
    ("xarm5", "robotiq_2f_140", "servo_cartesian", 50, 50),
    ("uf850", "robotiq_2f_140", "servo_cartesian", 50, 50),
]

PACKAGING_CASES = [
    ("xarm6", "robotiq_2f_85", "servo_cartesian", 2),
    ("xarm6", "robotiq_2f_140", "servo_j", 2),
]


def _paths(base_dir: Path, suite: str, name: str):
    directory = base_dir / suite
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.csv", directory / f"{name}.json", directory / f"{name}.preflight.json"


def _pick_case(robot, variant, speed, force, executor, *, tmp_path: Path, fault=None, monkeypatch=None):
    import genesis as gs
    from ufactory.cli.pick_place import _backends, _build_ik_scene, _build_program
    from ufactory.config import load_runtime_config
    from ufactory.grippers.robotiq import get_robotiq_spec
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.trajectory.g2_confirmation import prepare_robotiq_program
    from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
    from ufactory.trajectory.preflight import create_safety_gate
    from ufactory.trajectory.sim_executor import replay_sim

    config = load_runtime_config(
        robot,
        gripper=variant,
        overrides={"simulation.robotiq.speed_percent": speed, "simulation.robotiq.force_percent": force},
    )
    name = f"{robot}_{variant}_{executor}_s{speed}_f{force}" + (f"_{fault}" if fault else "")
    csv, report_path, preflight_path = _paths(tmp_path, "faults" if fault else "pick-matrix", name)
    with GenesisRuntimeManager(config.simulation):
        assert str(gs.device).startswith("cuda")
        ctx = _build_ik_scene(config, calibration=None, calibrated_g2=True)
        program = prepare_robotiq_program(
            _build_program(config, q_home=ctx.home_qpos[ctx.arm_dof_idx]), config.simulation.robotiq, variant
        )
        if executor == "servo_j":
            program = compile_cartesian_program_to_joint_stream(program, ctx)
        kinematics, collision = _backends(config, Path(ctx.robot_urdf_path), calibrated_g2=True)
        gate = create_safety_gate(
            config,
            kinematics=kinematics,
            collision=collision,
            urdf_path=ctx.robot_urdf_path,
            calibration_sha256="0" * 64,
            scene_sha256=hashlib.sha256(json.dumps(asdict(config.simulation), sort_keys=True).encode()).hexdigest(),
        )
        preflight = gate.preflight(program, executor=executor)
        preflight_path.write_text(json.dumps(preflight.to_dict(), indent=2) + "\n")
        assert preflight.passed, preflight_path
        gate.approve_simulation(program, preflight)
        open_mm = get_robotiq_spec(variant).open_gap_mm
        if fault == "no-contact":
            ctx.obj.set_pos(ctx.obj.get_pos().new_tensor([[2, 2, 0.5]]), zero_velocity=True)
        elif fault == "blocked-release":
            original = ctx.robotiq_controller.command

            def command(position, speed=50, force=50):
                if position == open_mm and ctx.robotiq_controller.controller.position_mm < open_mm:
                    return
                original(position, speed, force)

            monkeypatch.setattr(ctx.robotiq_controller, "command", command)
        report = replay_sim(program, ctx, csv_path=csv, sim_report_path=report_path)
        if fault:
            expected = "grip" if fault == "no-contact" else "release"
            assert not report.completed and report.timeout_phase == expected, report_path
            assert report.records[-1]["phase"] == expected
        else:
            assert report.completed and report.timing_passed, report_path
            assert report.transport_passed, (report_path, report.acceptance["failed_checks"])
            assert all(item["next_action_delay_s"] == 0.02 for item in report.confirmations.values())


@pytest.mark.integration
@pytest.mark.parametrize("robot,variant,executor,speed,force", PICK_CASES)
def test_robotiq_pick_place_timing_matrix(robot, variant, executor, speed, force, tmp_path):
    _pick_case(robot, variant, speed, force, executor, tmp_path=tmp_path)


@pytest.mark.integration
@pytest.mark.parametrize("variant", ("robotiq_2f_85", "robotiq_2f_140"))
@pytest.mark.parametrize("fault", ("no-contact", "blocked-release"))
def test_robotiq_timeout_stops_following_motion(variant, fault, monkeypatch, tmp_path):
    _pick_case("xarm6", variant, 50, 50, "servo_cartesian", tmp_path=tmp_path, fault=fault, monkeypatch=monkeypatch)


@pytest.mark.integration
@pytest.mark.parametrize("robot,variant,executor,cycles", PACKAGING_CASES)
def test_robotiq_packaging_timing_and_cycle_reset(robot, variant, executor, cycles, tmp_path):
    import genesis as gs
    from ufactory.config import load_runtime_config
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.manipulation.packaging.simulation import (
        init_showcase_robot,
        make_layout,
        prepare_packaging_cycle,
        run_pick_place_cycle,
    )
    from ufactory.simulation import GenesisRuntimeManager

    config = load_runtime_config(robot, task="packaging_showcase", gripper=variant)
    with GenesisRuntimeManager(config.simulation):
        assert str(gs.device).startswith("cuda")
        table_top = make_layout(runtime_config=config).table_top_z
        scene, robot_ent, block, layout = build_packaging_scene(
            table_top, sim_dt=0.02, show_viewer=False, runtime_config=config
        )
        ctx = init_showcase_robot(robot_ent, layout, scene, runtime_config=config)
        for c in range(1, cycles + 1):
            prepare_packaging_cycle(scene, robot_ent, block, layout, ctx)
            csv, report_path, preflight_path = _paths(tmp_path, "packaging", f"{robot}_{variant}_{executor}_c{c}")
            report = run_pick_place_cycle(
                scene,
                robot_ent,
                block,
                layout,
                ctx=ctx,
                executor=executor,
                runtime_config=config,
                csv_path=csv,
                sim_report_path=report_path,
                preflight_report_path=preflight_path,
            )
            assert report is not None and report.completed and report.timing_passed and report.transport_passed


@pytest.mark.integration
@pytest.mark.parametrize("variant", ("robotiq_2f_85", "robotiq_2f_140"))
def test_robotiq_batch_controller_four_envs(variant):
    import genesis as gs
    from ufactory.config import load_runtime_config
    from ufactory.grippers.robotiq import get_robotiq_spec
    from ufactory.robots.paths import robotiq_2f_85_movable_visual_urdf, robotiq_2f_140_movable_visual_urdf
    from ufactory.simulation import (
        GenesisRuntimeManager,
        RobotiqController,
        RobotiqModel,
        apply_robotiq_batch_control,
        configure_robotiq_drive,
        configure_robotiq_mimic_constraints,
        make_rigid_options,
    )
    from ufactory.simulation.robotiq_adapter import robotiq_measured_gap_mm

    spec = get_robotiq_spec(variant)
    urdf = robotiq_2f_85_movable_visual_urdf() if variant == "robotiq_2f_85" else robotiq_2f_140_movable_visual_urdf()
    sim_cfg = load_runtime_config("xarm6", gripper=variant).simulation
    with GenesisRuntimeManager(sim_cfg):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.02, substeps=32),
            rigid_options=make_rigid_options(gs, constraint_timeconst=0.00125, batch_dofs_info=True),
            show_viewer=False,
        )
        robot = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=True))
        scene.build(n_envs=4)
        configure_robotiq_mimic_constraints(robot, variant)
        drive_idx, all_idx = configure_robotiq_drive(robot, variant)
        left = robot.get_link(f"{spec.prefix}_left_finger")
        right = robot.get_link(f"{spec.prefix}_right_finger")
        model = RobotiqModel.load(variant)
        ctrl = RobotiqController(model, num_envs=4, device=gs.device, dtype=gs.tc_float)
        q_open = model.linkage.drive_for_gap_m(spec.open_gap_m)
        robot.set_dofs_position(torch.full((4, len(all_idx)), q_open, device=gs.device, dtype=gs.tc_float), all_idx)
        targets = torch.tensor([10.0, 25.0, 45.0, spec.open_gap_mm - 5.0], device=gs.device, dtype=gs.tc_float)
        ctrl.command(targets, speed_percent=[60.0, 60.0, 60.0, 60.0], force_percent=[50.0, 50.0, 50.0, 50.0])
        for _ in range(75):
            gap = robotiq_measured_gap_mm(model, left, right)
            apply_robotiq_batch_control(robot, ctrl, drive_idx, gap, 0.02)
            scene.step()
        final_gap = robotiq_measured_gap_mm(model, left, right)
        assert torch.allclose(final_gap, targets, atol=2.0), final_gap
