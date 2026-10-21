"""Representative G2 GPU acceptance: manipulation, faults, RL and dynamic demos."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

EXECUTORS = ("servo_cartesian", "servo_j")
PICK_CASES = [
    ("xarm6", "servo_cartesian", 10, 10),
    ("xarm6", "servo_cartesian", 10, 100),
    ("xarm6", "servo_cartesian", 50, 50),
    ("xarm6", "servo_cartesian", 90, 10),
    ("xarm6", "servo_cartesian", 90, 100),
    ("xarm6", "servo_j", 50, 50),
    ("xarm5", "servo_j", 50, 50),
    ("xarm7", "servo_cartesian", 50, 50),
    ("uf850", "servo_j", 50, 50),
]
PACKAGING_CASES = [
    ("xarm6", "servo_cartesian", 2),
    ("xarm6", "servo_j", 2),
    ("xarm5", "servo_cartesian", 1),
    ("xarm7", "servo_j", 1),
    ("uf850", "servo_cartesian", 1),
]


def _paths(base_dir: Path, suite: str, name: str):
    directory = base_dir / suite
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.csv", directory / f"{name}.json", directory / f"{name}.preflight.json"


@pytest.mark.integration
@pytest.mark.parametrize("robot,executor,speed,force", PICK_CASES)
def test_pick_place_timing_matrix(robot, executor, speed, force, tmp_path):
    _pick_case(robot, speed, force, executor, tmp_path=tmp_path)


def _pick_case(robot, speed, force, executor, *, tmp_path: Path, fault=None, monkeypatch=None):
    import genesis as gs
    from ufactory.cli.pick_place import _backends, _build_ik_scene, _build_program
    from ufactory.config import load_runtime_config
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.trajectory.g2_confirmation import prepare_g2_program
    from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
    from ufactory.trajectory.preflight import create_safety_gate
    from ufactory.trajectory.sim_executor import replay_sim

    config = load_runtime_config(
        robot, overrides={"simulation.g2.speed_percent": speed, "simulation.g2.force_percent": force}
    )
    name = f"{robot}_{executor}_s{speed}_f{force}" + (f"_{fault}" if fault else "")
    csv, report_path, preflight_path = _paths(tmp_path, "faults" if fault else "pick-matrix", name)
    with GenesisRuntimeManager(config.simulation):
        assert str(gs.device).startswith("cuda")
        ctx = _build_ik_scene(config, calibration=None, calibrated_g2=True)
        program = prepare_g2_program(
            _build_program(config, q_home=ctx.home_qpos[ctx.arm_dof_idx]), config.simulation.g2
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
        if fault == "no-contact":
            ctx.obj.set_pos(ctx.obj.get_pos().new_tensor([[2, 2, 0.5]]), zero_velocity=True)
        elif fault == "blocked-release":
            original = ctx.g2_controller.command

            def command(position, speed=50, force=50):
                if position == 84 and ctx.g2_controller.controller.position_mm < 84:
                    return
                original(position, speed, force)

            monkeypatch.setattr(ctx.g2_controller, "command", command)
        report = replay_sim(program, ctx, csv_path=csv, sim_report_path=report_path)
        if fault:
            expected = "grip" if fault == "no-contact" else "release"
            assert not report.completed and report.timeout_phase == expected, report_path
            assert report.records[-1]["phase"] == expected
            forbidden = (
                {"lift", "transit", "retreat", "return-home"} if expected == "grip" else {"retreat", "return-home"}
            )
            assert not forbidden.intersection(phase.label for phase in report.phases)
            assert report.metric_settle_ticks == 0
        else:
            assert report.completed and report.timing_passed, report_path
            assert report.transport_passed, (report_path, report.acceptance["failed_checks"])
            assert all(item["next_action_delay_s"] == 0.02 for item in report.confirmations.values())


@pytest.mark.integration
@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("fault", ("no-contact", "blocked-release"))
def test_pick_place_timeout_stops_following_motion(executor, fault, monkeypatch, tmp_path):
    _pick_case("xarm6", 50, 50, executor, tmp_path=tmp_path, fault=fault, monkeypatch=monkeypatch)


@pytest.mark.integration
@pytest.mark.parametrize("robot,executor,cycles", PACKAGING_CASES)
def test_packaging_timing_and_cycle_reset(robot, executor, cycles, tmp_path):
    import genesis as gs
    from ufactory.config import load_runtime_config
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.manipulation.packaging.simulation import (
        SETTLE_STEPS,
        hold_robot_home,
        init_showcase_robot,
        prepare_packaging_cycle,
        run_pick_place_cycle,
    )
    from ufactory.simulation import GenesisRuntimeManager

    config = load_runtime_config(robot, task="packaging_showcase")
    with GenesisRuntimeManager(config.simulation):
        assert str(gs.device).startswith("cuda")
        scene, arm, block, layout = build_packaging_scene(show_viewer=False, runtime_config=config)
        ctx = init_showcase_robot(arm, layout, scene, runtime_config=config)
        hold_robot_home(arm, scene, ctx, steps=SETTLE_STEPS)
        for cycle in range(cycles):
            csv, report_path, preflight_path = _paths(tmp_path, "pack-matrix", f"{robot}_{executor}_cycle{cycle + 1}")
            prepare_packaging_cycle(scene, arm, block, layout, ctx)
            report = run_pick_place_cycle(
                scene,
                arm,
                block,
                layout,
                ctx=ctx,
                executor=executor,
                runtime_config=config,
                csv_path=csv,
                sim_report_path=report_path,
                preflight_report_path=preflight_path,
            )
            assert report.completed and report.timing_passed, report_path
            assert report.transport_passed, (report_path, report.acceptance["failed_checks"])
            assert report.confirmations["release"]["next_phase"] == "post-release-settle"
            assert report.confirmations["release"]["next_action_delay_s"] == 0.02
            assert any(phase.label == "post-release-settle" and phase.ticks >= 2 for phase in report.phases)


@pytest.mark.parametrize("num_envs", [1, 4])
def test_rl_batch_control_and_independent_reset(num_envs):
    import genesis as gs
    from examples.rl.pick_place.env import XArm6PickPlaceEnv
    from ufactory.config import load_runtime_config
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.training.tasks import build_pick_place_task_configs

    env_cfg, reward, robot = build_pick_place_task_configs(
        "xarm6", recipe_path=Path("examples/rl/pick_place/recipe.yaml")
    )
    env_cfg.update(num_envs=num_envs)
    with GenesisRuntimeManager(load_runtime_config("xarm6").simulation):
        assert str(gs.device).startswith("cuda"), "GPU acceptance must not silently fall back to CPU"
        env = XArm6PickPlaceEnv(env_cfg=env_cfg, reward_cfg=reward, robot_cfg=robot, show_viewer=False)
        env.reset()
        command = env.ee_setpoint.clone()
        gap_command = env.gap_cmd.clone()
        increments = torch.full((num_envs, 4), 0.1, device=gs.device)
        obs, _, done, _ = env.step(increments)
        assert not done.any()
        assert obs["policy"].shape == (num_envs, 70)
        assert not hasattr(env, "phase")
        torch.testing.assert_close(env.ee_setpoint, command + increments[:, :3] * env.action_scale)
        torch.testing.assert_close(env.gap_cmd, gap_command + increments[:, 3] * env.gap_action_scale)
        command, gap_command = env.ee_setpoint.clone(), env.gap_cmd.clone()
        env.step(torch.zeros_like(increments))
        torch.testing.assert_close(env.ee_setpoint, command, rtol=0, atol=0)
        torch.testing.assert_close(env.gap_cmd, gap_command, rtol=0, atol=0)
        env.g2_force_percent = torch.linspace(10, 100, num_envs, device=gs.device)
        actions = torch.zeros((num_envs, 4), device=gs.device)
        actions[:, 3] = -1
        for _ in range(40):
            obs, _, _, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all()
        lower, upper = env.robot.get_dofs_force_range(env.gripper_dof_idx)
        torch.testing.assert_close(upper.reshape(-1), env.g2_controller.torque_limit_nm)
        torch.testing.assert_close(lower.reshape(-1), -env.g2_controller.torque_limit_nm)
        if num_envs > 1:
            assert upper.max() > upper.min()
            untouched = torch.arange(1, num_envs, device=gs.device)
            integral = env.g2_controller.integral_torque_nm[untouched].clone()
            reference = env.g2_controller.reference_gap_mm[untouched].clone()
            command = env.ee_setpoint[untouched].clone()
            history = env.last_actions[untouched].clone()
            metrics = env.metrics.max_table_force[untouched].clone()
            env.reset_idx(torch.tensor([0], device=gs.device))
            torch.testing.assert_close(integral, env.g2_controller.integral_torque_nm[untouched], rtol=0, atol=0)
            torch.testing.assert_close(reference, env.g2_controller.reference_gap_mm[untouched], rtol=0, atol=0)
            torch.testing.assert_close(command, env.ee_setpoint[untouched], rtol=0, atol=0)
            torch.testing.assert_close(history, env.last_actions[untouched], rtol=0, atol=0)
            torch.testing.assert_close(metrics, env.metrics.max_table_force[untouched], rtol=0, atol=0)
            assert env.g2_controller.reference_gap_mm[0].item() == pytest.approx(env.pre_grasp_gap_m * 1000.0)
            assert env.g2_controller.integral_torque_nm[0].item() == 0.0
        for _ in range(5):
            env.step(torch.zeros_like(actions))


@pytest.mark.integration
@pytest.mark.parametrize("robot", ["standalone", "xarm6"])
def test_dynamic_demo_opens_and_closes_without_teleport(robot, monkeypatch, tmp_path):
    import genesis as gs
    from ufactory.robots.paths import gripper_g2_movable_visual_urdf, robot_visual_glb_urdf
    from ufactory.robots.registry import get_robot_profile
    from ufactory.visualization.robot_viewer import run_glb_viewer
    from ufactory.visualization.standalone_gripper import run_standalone_gripper_viewer
    from examples.visualization.view_gripper_g2 import _controller

    try:
        if robot == "standalone":
            monkeypatch.setattr("sys.argv", ["viewer", "--headless"])
            result = run_standalone_gripper_viewer(
                "G2", gripper_g2_movable_visual_urdf(), _controller, calibrated_g2=True
            )
        else:
            result = run_glb_viewer(
                get_robot_profile(robot),
                robot_visual_glb_urdf(robot, with_gripper_g2=True, movable=True),
                headless=True,
                gripper_demo=True,
            )
        assert str(gs.device).startswith("cuda")
        directory = tmp_path / "viewers"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{robot}.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["min_gap_mm"] <= 1
        assert result["max_gap_mm"] >= 83
        assert abs(result["final_gap_mm"] - 84) <= 1
    finally:
        gs.destroy()
