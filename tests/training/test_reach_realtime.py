"""CPU regressions for batch independence, episode accounting and realtime pacing."""

from __future__ import annotations

import copy
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("genesis")
pytest.importorskip("rsl_rl")
pytest.importorskip("pyglet")

from examples.rl.reach import visualize as vis_realtime
from examples.rl.reach.realtime_env import RealtimeReachEnv
from examples.rl.reach.realtime_viewer import RealtimeClock, warmup
from examples.rl.reach.env import XArm6ReachEnv


@pytest.mark.parametrize("entry", ["evaluate", "visualize"])
def test_inference_requires_an_explicit_checkpoint(entry, monkeypatch, capsys):
    from importlib import import_module

    module = import_module("examples.rl.reach." + entry)
    monkeypatch.setattr(sys, "argv", [entry])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "--checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["evaluate", "visualize"])
def test_missing_checkpoint_fails_before_genesis_initialization(entry, tmp_path, monkeypatch):
    from importlib import import_module

    module = import_module("examples.rl.reach." + entry)
    monkeypatch.setattr(sys, "argv", [entry, "--checkpoint", str(tmp_path / "missing.pt")])
    monkeypatch.setattr(module.gs, "init", lambda **kwargs: pytest.fail("initialized before validating the checkpoint"))
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        module.main()


def test_old_evaluate_vis_prints_a_copyable_migration_command(monkeypatch, capsys):
    from examples.rl.reach import evaluate

    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "model with space.pt", "--vis"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2
    assert "python -m examples.rl.reach.visualize --checkpoint 'model with space.pt'" in capsys.readouterr().err


def test_resume_environment_count_is_checked_before_creating_output(tmp_path, monkeypatch, capsys):
    from examples.rl.reach import train

    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()
    (tmp_path / "config.yaml").touch()
    output = tmp_path / "resume"
    monkeypatch.setattr(
        train, "validate_and_load_rsl_checkpoint", lambda *args, **kwargs: ({}, {"env": {"num_envs": 64}}, None)
    )
    monkeypatch.setattr(sys, "argv", ["train", "--resume", str(checkpoint), "--log-dir", str(output), "-B", "2"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2
    assert "original --num-envs" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="X11 screen fallback is Linux-specific")
def test_default_screen_falls_back_without_randr_outputs_and_keeps_primary_selection(monkeypatch):
    from pyglet.display import xlib

    fallback = object.__new__(xlib.XlibScreen)
    primary = object.__new__(xlib.XlibScreenXrandr)
    primary._is_primary = True
    display = SimpleNamespace(_screens=[fallback])
    display.get_screens = lambda: display._screens
    monkeypatch.setattr(xlib, "_have_xrandr", True)
    assert xlib.XlibDisplay.get_default_screen(display) is fallback
    display._screens.append(primary)
    assert xlib.XlibDisplay.get_default_screen(display) is primary


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(vis_realtime.gs, "tc_float", torch.float32, raising=False)
    monkeypatch.setattr(vis_realtime.gs, "tc_int", torch.int64, raising=False)
    obj = object.__new__(RealtimeReachEnv)
    obj.num_envs, obj.num_obs, obj.num_actions = 4, 24, 6
    obj.device = torch.device("cpu")
    obj.ctrl_dt, obj.max_episode_length = 0.02, 250
    obj.action_scale, obj.action_clip, obj.max_joint_delta_rad = 0.05, 1.0, 0.01
    obj.success_distance_m, obj.success_hold_s, obj.hold_steps = 0.005, 1.0, 50
    obj.hold_steps_left = torch.zeros(4, dtype=torch.int64)
    obj.default_qpos, obj.dof_idx = torch.zeros(6), list(range(6))
    obj.joint_lower, obj.joint_upper = torch.full((6,), -1.0), torch.ones(6)
    obj.position_command = torch.zeros(4, 6)
    q, v = torch.zeros(4, 6), torch.zeros(4, 6)
    command = torch.zeros(4, 6)

    def set_position(value, _idx, *, envs_idx=None, zero_velocity=False):
        idx = slice(None) if envs_idx is None else envs_idx
        q[idx] = value
        if zero_velocity:
            v[idx] = 0

    def control(value, _idx, *, envs_idx=None):
        command[slice(None) if envs_idx is None else envs_idx] = value

    obj.robot = SimpleNamespace(
        get_dofs_position=lambda _: q,
        get_dofs_velocity=lambda _: v,
        set_dofs_position=set_position,
        set_dofs_velocity=lambda value, _: v.copy_(value),
        control_dofs_position=control,
        command=command,
    )
    obj.ee_link = SimpleNamespace(get_pos=lambda: q[:, :3])
    obj.target_marker = SimpleNamespace(set_pos=lambda *args, **kwargs: None)
    obj.scene = SimpleNamespace(ticks=0)

    def step(*, update_visualizer=True):
        obj.scene.ticks += 1

    obj.scene.step = step
    obj.scene.get_state = lambda: (q.clone(), v.clone())
    obj.scene.reset = lambda state: (q.copy_(state[0]), v.copy_(state[1]))
    obj.target_pos, obj.target_pos_lower, obj.target_pos_upper = torch.ones(4, 3), torch.ones(3), torch.full((3,), 2.0)
    obj.episode_length_buf, obj.metric_steps = torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
    obj.reward_scales = {"reach": 0.02, "action_penalty": 0.00002, "success": 2.0}
    obj.reward_functions = {name: getattr(obj, "_reward_" + name) for name in obj.reward_scales}
    obj.episode_sums = {name: torch.zeros(4) for name in obj.reward_scales}
    obj.metric_sums = {name: torch.zeros(4) for name in ("action_saturation_rate", "command_tracking_error_rad")}
    obj.extras = {}
    return obj


def test_one_arm_holds_for_50_ticks_while_other_commands_keep_updating(env):
    env.target_pos[0] = 0
    actions = torch.ones(4, 6)
    actions[0] = 0
    _, _, done, info = env.step(actions)
    assert done.tolist() == [True, False, False, False]
    assert info["reach_done_steps"].tolist() == [1]
    assert env.hold_steps_left.tolist() == [50, 0, 0, 0]
    arrival_command = env.position_command[0].clone()
    for tick in range(50):
        previous = env.position_command[1:].clone()
        _, reward, done, info = env.step(torch.ones(4, 6))
        assert env.scene.ticks == tick + 2
        assert not done.any()
        assert reward[0] == 0
        assert "reach_done_success" not in info
        assert torch.equal(env.position_command[0], arrival_command)
        assert torch.allclose(env.position_command[1:] - previous, torch.full((3, 6), 0.01))
        if tick < 49:
            assert torch.equal(env.target_pos[0], torch.zeros(3))
            assert env.episode_length_buf[0] == 1
            assert env.metric_steps[0] == 1
    assert env.hold_steps_left[0] == 0
    assert env.episode_length_buf[0] == 0
    assert env.metric_steps[0] == 0
    assert torch.all(env.target_pos[0] >= 1)
    env.step(torch.ones(4, 6))
    assert env.position_command[0, 0] == pytest.approx(0.01)


def test_staggered_successes_reset_independently_without_duplicate_reports(env):
    env.target_pos[0] = 0
    reported = []
    for tick in range(62):
        if tick == 10:
            env.target_pos[1] = 0
        _, _, _, info = env.step(torch.zeros(4, 6))
        reported.extend(info.get("reach_done_envs_idx", torch.tensor([])).tolist())
        if tick == 50:
            assert env.hold_steps_left.tolist() == [0, 10, 0, 0]
            assert torch.all(env.target_pos[0] >= 1)
            assert torch.equal(env.target_pos[1], torch.zeros(3))
        if tick == 60:
            assert env.hold_steps_left.tolist() == [0, 0, 0, 0]
    assert reported == [0, 1]
    assert env.scene.ticks == 62


def test_all_arms_success_together_counts_once_and_steps_once_per_call(env):
    env.target_pos[:] = 0
    _, reward, done, info = env.step(torch.zeros(4, 6))
    assert done.tolist() == [True] * 4
    assert info["reach_done_envs_idx"].tolist() == [0, 1, 2, 3]
    assert torch.allclose(reward, torch.full((4,), 2.02))
    for _ in range(50):
        _, reward, done, _ = env.step(torch.ones(4, 6))
        assert not done.any()
        assert not reward.any()
    assert env.scene.ticks == 51
    assert torch.equal(env.episode_length_buf, torch.zeros(4, dtype=torch.int64))


def test_timeout_resets_immediately_and_success_takes_precedence(env):
    env.target_pos[0] = 0
    env.episode_length_buf[:2] = 249
    _, _, done, info = env.step(torch.zeros(4, 6))
    assert done.tolist() == [True, True, False, False]
    assert info["reach_done_success"].tolist() == [True, False]
    assert info["time_outs"].tolist() == [0, 1, 0, 0]
    assert env.hold_steps_left.tolist() == [50, 0, 0, 0]
    assert env.episode_length_buf.tolist() == [250, 0, 1, 1]
    assert env.scene.ticks == 1


def test_success_threshold_remains_strictly_below_5mm(env):
    env.target_pos[:3] = torch.tensor([[0.0049, 0, 0], [0.005, 0, 0], [0.0051, 0, 0]])
    _, _, done, _ = env.step(torch.zeros(4, 6))
    assert done.tolist() == [True, False, False, False]


def test_zero_hold_resets_on_arrival(env):
    env.hold_steps = 0
    env.target_pos[0] = 0
    _, _, done, _ = env.step(torch.zeros(4, 6))
    assert done[0]
    assert env.hold_steps_left[0] == 0
    assert env.episode_length_buf[0] == 0
    assert torch.all(env.target_pos[0] >= 1)


@pytest.mark.parametrize("actions", [torch.zeros(2, 6), torch.full((4, 6), float("nan"))])
def test_invalid_policy_actions_do_not_advance_scene(env, actions):
    with pytest.raises((ValueError, RuntimeError)):
        env.step(actions)
    assert env.scene.ticks == 0


def test_active_transition_matches_training_environment(env):
    # Switch the same fixture between both implementations, with no completion.
    state = env.training_state_dict()
    actions = torch.tensor([[2.0, -2.0, 0.1, -0.1, 0.0, 0.5]]).expand(4, -1)
    expected_obs, expected_reward, expected_done, _ = XArm6ReachEnv.step(env, actions)
    expected_state = env.training_state_dict()
    env.load_training_state_dict(state)
    obs, reward, done, _ = env.step(actions)
    assert torch.equal(obs["policy"], expected_obs["policy"])
    assert torch.equal(reward, expected_reward)
    assert torch.equal(done, expected_done)
    for key in ("position_command", "episode_length_buf", "metric_steps"):
        assert torch.equal(env.training_state_dict()[key], expected_state[key])


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_restores_initial_pose_episode_and_all_rng_states(env, fail):
    env.target_pos[0] = 0
    state = copy.deepcopy(env.training_state_dict())
    torch_state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()

    def policy(_):
        random.random()
        np.random.rand()
        actions = torch.rand(4, 6)
        if fail:
            raise RuntimeError("warmup failure")
        return actions

    if fail:
        with pytest.raises(RuntimeError, match="warmup failure"):
            warmup(env, policy)
    else:
        warmup(env, policy)
    restored = env.training_state_dict()
    for key in ("joint_pos", "joint_vel", "position_command", "episode_length_buf", "target_pos", "metric_steps"):
        assert torch.equal(restored[key], state[key])
    for group in ("episode_sums", "metric_sums"):
        for key in state[group]:
            assert torch.equal(restored[group][key], state[group][key])
    assert not env.hold_steps_left.any()
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


def test_realtime_clock_delays_deadline_after_overrun(monkeypatch):
    now, sleeps = [0.0], []
    monkeypatch.setattr("examples.rl.reach.realtime_viewer.time.perf_counter", lambda: now[0])

    def sleep(duration):
        sleeps.append(duration)
        now[0] += duration

    monkeypatch.setattr("examples.rl.reach.realtime_viewer.time.sleep", sleep)
    clock = RealtimeClock(0.02)
    now[0] = 0.005
    clock.wait()
    assert sleeps == [pytest.approx(0.015)]
    now[0] = 0.1
    clock.wait()
    assert clock.deadline == pytest.approx(0.12)
    now[0] = 0.101
    clock.wait()
    assert sleeps[-1] == pytest.approx(0.019)
