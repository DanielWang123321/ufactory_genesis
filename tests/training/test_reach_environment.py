"""Exercise reach transitions on CPU using a robot with a fixed tracking lag."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

gs = pytest.importorskip("genesis")
from examples.rl.reach.env import XArm6ReachEnv
from ufactory.training.reach_evaluation import evaluate_fixed_targets, sample_reach_targets, summarize_reach_reports


@pytest.fixture(autouse=True)
def cpu_tensor_types(monkeypatch):
    monkeypatch.setattr(gs, "tc_float", torch.float32, raising=False)
    monkeypatch.setattr(gs, "tc_int", torch.int64, raising=False)


class Robot:
    def __init__(self, count: int) -> None:
        self.q = torch.zeros(count, 6)
        self.v = torch.zeros_like(self.q)
        self.command = self.q.clone()

    def get_dofs_position(self, _idx):
        return self.q

    def get_dofs_velocity(self, _idx):
        return self.v

    def set_dofs_position(self, value, _idx, *, envs_idx=None, zero_velocity=False):
        idx = slice(None) if envs_idx is None else envs_idx
        self.q[idx] = value
        if zero_velocity:
            self.v[idx] = 0

    def set_dofs_velocity(self, value, _idx):
        self.v.copy_(value)

    def control_dofs_position(self, value, _idx, *, envs_idx=None):
        self.command[slice(None) if envs_idx is None else envs_idx] = value


def make_env(count=2):
    env = object.__new__(XArm6ReachEnv)
    env.num_envs, env.num_actions, env.num_obs = count, 6, 24
    env.device = torch.device("cpu")
    env.ctrl_dt, env.max_episode_length = 0.02, 250
    env.action_scale, env.action_clip, env.max_joint_delta_rad = 0.05, 1.0, 0.01
    env.success_distance_m = 0.005
    env.robot, env.dof_idx = Robot(count), list(range(6))
    env.ee_link = SimpleNamespace(get_pos=lambda: torch.zeros(count, 3))
    env.target_marker = SimpleNamespace(set_pos=lambda *args, **kwargs: None)
    env.scene = SimpleNamespace(step=lambda: None)
    env.default_qpos = torch.zeros(6)
    env.joint_lower, env.joint_upper = torch.full((6,), -1.0), torch.ones(6)
    env.position_command = torch.zeros(count, 6)
    env.target_pos = torch.ones(count, 3)
    env.target_pos_lower, env.target_pos_upper = torch.ones(3), torch.full((3,), 2.0)
    env.episode_length_buf = torch.zeros(count, dtype=torch.int64)
    env.metric_steps = torch.zeros(count, dtype=torch.int64)
    env.reward_scales = {"reach": 0.02, "action_penalty": 0.00002, "success": 2.0}
    env.reward_functions = {name: getattr(env, "_reward_" + name) for name in env.reward_scales}
    env.episode_sums = {name: torch.zeros(count) for name in env.reward_scales}
    env.metric_sums = {name: torch.zeros(count) for name in ("action_saturation_rate", "command_tracking_error_rad")}
    env.extras = {}
    return env


def test_command_accumulates_despite_tracking_lag_and_clamps_then_resets():
    env = make_env()
    for _ in range(120):
        observation, _, done, _ = env.step(torch.ones(2, 6))
        assert not done.any()
    assert torch.equal(env.position_command, torch.ones(2, 6))
    assert torch.equal(env.robot.q, torch.zeros(2, 6))
    assert torch.equal(observation["policy"][:, 18:], torch.ones(2, 6))
    env.reset_idx(torch.tensor([0]))
    assert torch.equal(env.position_command[0], env.default_qpos)
    assert torch.equal(env.robot.command[0], env.default_qpos)
    assert env.position_command[1, 0] == 1.0


def test_success_bonus_and_timeout_precedence_and_one_bonus_per_episode():
    env = make_env()
    env.target_pos[0] = 0
    env.episode_length_buf[:] = 249
    _, reward, done, info = env.step(torch.zeros(2, 6))
    assert done.tolist() == [True, True]
    assert reward[0].item() == pytest.approx(2.02)
    assert info["time_outs"].tolist() == [0, 1]
    assert info["episode"]["success_rate"].tolist() == [1, 0]
    assert info["episode"]["reward/success"].tolist() == [2, 0]
    assert info["reach_done_steps"].tolist() == [250, 250]
    _, reward, _, _ = env.step(torch.zeros(2, 6))
    assert torch.all(reward < 0.02)


def test_velocity_penalty_is_scaled_by_control_period():
    env = make_env()
    env.robot.v[:] = 2
    _, reward, _, _ = env.step(torch.zeros(2, 6))
    expected = 0.02 * (torch.exp(-10 * torch.sqrt(torch.tensor(3.0))) - 0.001 * 24)
    assert torch.allclose(reward, expected.expand(2))


def test_state_restores_joint_state_commands_and_episode_accounting():
    env = make_env()
    env.robot.q[:] = 0.2
    env.robot.v[:] = -0.3
    env.position_command[:] = 0.4
    env.step(torch.zeros(2, 6))
    state = env.training_state_dict()
    env.reset()
    env.load_training_state_dict(state)
    assert torch.equal(env.robot.q, state["joint_pos"])
    assert torch.equal(env.robot.v, state["joint_vel"])
    assert torch.equal(env.robot.command, state["position_command"])
    assert torch.equal(env.episode_length_buf, state["episode_length_buf"])
    assert torch.equal(env.target_pos, state["target_pos"])
    assert torch.equal(env.episode_sums["reach"], state["episode_sums"]["reach"])
    assert torch.equal(env.metric_steps, state["metric_steps"])
    env.position_command[:] = 0
    assert state["position_command"][0, 0] == pytest.approx(0.4)
    state["env_version"] = 1
    with pytest.raises(ValueError, match="retrain"):
        env.load_training_state_dict(state)


def test_evaluation_waits_for_slow_episodes_and_handles_partial_batches():
    env = make_env()
    env.max_episode_length = 5
    targets = torch.tensor([[0.0, 0, 0], [1.0, 1, 1], [0.004, 0, 0]])
    reports = evaluate_fixed_targets(env, lambda _: torch.zeros(2, 6), targets)
    assert [report["index"] for report in reports] == [0, 1, 2]
    assert [report["steps"] for report in reports] == [1, 5, 1]
    assert [report["success"] for report in reports] == [True, False, True]
    assert summarize_reach_reports(reports)["success_rate"] == pytest.approx(2 / 3)
    bank = sample_reach_targets([0, 0, 0], [1, 1, 1], 128, 17002)
    assert torch.equal(bank[:32], sample_reach_targets([0, 0, 0], [1, 1, 1], 32, 17002))


def _record_scene_steps(env):
    calls = []

    def scene_step():
        calls.append(
            {
                "command": env.position_command.clone(),
                "target": env.target_pos.clone(),
                "q": env.robot.q.clone(),
                "length": env.episode_length_buf.clone(),
                "metric_steps": env.metric_steps.clone(),
            }
        )

    env.scene.step = scene_step
    return calls


def test_success_without_hold_resets_on_the_arrival_step():
    env = make_env(count=1)
    env.target_pos[0] = 0
    env.position_command[:] = 0.4
    env.robot.q[:] = 0.4
    calls = _record_scene_steps(env)
    _, reward, done, info = env.step(torch.zeros(1, 6))
    assert done.tolist() == [True]
    assert len(calls) == 1
    assert info["reach_done_steps"].tolist() == [1]
    assert reward.item() == pytest.approx(2.02)
    assert torch.equal(env.position_command, torch.zeros(1, 6))


def test_training_success_is_strictly_below_5mm():
    env = make_env(count=3)
    env.success_distance_m = 0.005
    env.target_pos[:] = torch.tensor([[0.0049, 0, 0], [0.005, 0, 0], [0.0051, 0, 0]])
    _, _, done, _ = env.step(torch.zeros(3, 6))
    assert done.tolist() == [True, False, False]
