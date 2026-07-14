"""Unit tests for reach deploy safety and observation helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ufactory.deploy.action_adapter import apply_online_joint_delta, apply_reach_joint_delta, apply_servo_joint_delta
from ufactory.deploy.action_postprocess import (
    effective_max_joint_delta_rad,
    process_reach_action_np,
    reach_action_delta_torch,
)
from ufactory.deploy.obs_adapter import build_reach_obs, parse_target_xyz, validate_obs_shape
from ufactory.deploy.reach_config import EXECUTOR_ONLINE_JOINT, EXECUTOR_SERVO_J, ReachDeployConfig
from ufactory.deploy.safety import SafetyGuard, SafetyViolation
from ufactory.deploy.sdk_units import sdk_position_m
from ufactory.robots.runtime import get_robot_runtime_profile


@pytest.fixture
def reach_config() -> ReachDeployConfig:
    return ReachDeployConfig.for_robot("xarm6", z_min_mm=0.0)


@pytest.fixture
def online_reach_config() -> ReachDeployConfig:
    return ReachDeployConfig.for_robot("xarm6", z_min_mm=0.0, executor=EXECUTOR_ONLINE_JOINT)


@pytest.fixture
def safety(reach_config: ReachDeployConfig) -> SafetyGuard:
    runtime = get_robot_runtime_profile("xarm6")
    return SafetyGuard.from_runtime(runtime, reach_config)


@pytest.fixture
def online_safety(online_reach_config: ReachDeployConfig) -> SafetyGuard:
    runtime = get_robot_runtime_profile("xarm6")
    return SafetyGuard.from_runtime(runtime, online_reach_config)


def test_build_reach_obs_shape_and_target_rel():
    q = np.arange(6, dtype=np.float64)
    qd = np.ones(6)
    ee = np.array([0.4, 0.0, 0.3])
    target = np.array([0.5, 0.1, 0.35])
    obs = build_reach_obs(q, qd, ee, target)
    assert obs.shape == (18,)
    np.testing.assert_allclose(obs[12:15], ee)
    np.testing.assert_allclose(obs[15:18], target - ee)


def test_parse_target_xyz():
    np.testing.assert_allclose(parse_target_xyz("0.4,0.0,0.3"), [0.4, 0.0, 0.3])


def test_effective_max_joint_delta_uses_servo_speed(reach_config: ReachDeployConfig):
    assert reach_config.executor == EXECUTOR_SERVO_J
    assert reach_config.max_joint_delta_rad == pytest.approx(reach_config.servo_speed_rad_s * reach_config.ctrl_dt)
    assert reach_config.max_joint_delta_rad == pytest.approx(0.01)


def test_online_joint_executor_uses_policy_delta_limit(online_reach_config: ReachDeployConfig):
    assert online_reach_config.executor == EXECUTOR_ONLINE_JOINT
    assert online_reach_config.max_joint_delta_rad == pytest.approx(
        online_reach_config.action_scale * online_reach_config.action_clip
    )
    assert online_reach_config.max_joint_delta_rad == pytest.approx(0.05)


def test_checkpoint_action_contract_preserves_runtime_executor(
    reach_config: ReachDeployConfig,
    online_reach_config: ReachDeployConfig,
):
    merged = online_reach_config.with_action_contract(reach_config)

    assert merged.executor == EXECUTOR_ONLINE_JOINT
    assert merged.max_joint_delta_rad == pytest.approx(reach_config.max_joint_delta_rad)
    assert merged.max_joint_delta_rad == pytest.approx(0.01)
    assert merged.online_joint_speed_rad_s == pytest.approx(online_reach_config.online_joint_speed_rad_s)


def test_process_reach_action_np_clips_and_records_command(reach_config: ReachDeployConfig):
    current_q = np.zeros(6)
    action = np.array([2.0, -0.5, 0.1, 0.0, 0.0, 0.0])
    command = process_reach_action_np(
        current_q,
        action,
        action_scale=reach_config.action_scale,
        action_clip=reach_config.action_clip,
        max_joint_delta_rad=reach_config.max_joint_delta_rad,
    )

    np.testing.assert_allclose(command.raw_action, action)
    np.testing.assert_allclose(command.clipped_action[:3], [1.0, -0.5, 0.1])
    np.testing.assert_allclose(command.joint_delta[:3], [0.01, -0.01, 0.005])
    np.testing.assert_allclose(command.target_q, command.joint_delta)


def test_process_reach_action_np_rejects_shape_mismatch(reach_config: ReachDeployConfig):
    with pytest.raises(ValueError):
        process_reach_action_np(
            np.zeros(6),
            np.zeros(5),
            action_scale=reach_config.action_scale,
            action_clip=reach_config.action_clip,
            max_joint_delta_rad=reach_config.max_joint_delta_rad,
        )


def test_reach_action_delta_torch_matches_numpy(reach_config: ReachDeployConfig):
    action = np.array([[2.0, -0.5, 0.1, 0.0, 0.0, 0.0]], dtype=np.float32)
    delta_t = reach_action_delta_torch(
        torch.tensor(action),
        action_scale=reach_config.action_scale,
        action_clip=reach_config.action_clip,
        max_joint_delta_rad=reach_config.max_joint_delta_rad,
    )
    command = process_reach_action_np(
        np.zeros(6),
        action.reshape(-1),
        action_scale=reach_config.action_scale,
        action_clip=reach_config.action_clip,
        max_joint_delta_rad=reach_config.max_joint_delta_rad,
    )
    np.testing.assert_allclose(delta_t.cpu().numpy().reshape(-1), command.joint_delta, atol=1e-7)


def test_effective_max_joint_delta_keeps_legacy_when_no_servo_limit():
    limit = effective_max_joint_delta_rad(action_scale=0.05, action_clip=1.0)
    assert limit == pytest.approx(0.05)


def test_clip_action_and_joint_delta(reach_config: ReachDeployConfig, safety: SafetyGuard):
    action = np.array([2.0, -3.0, 0.5, 0.0, 0.0, 0.0])
    clipped = safety.clip_action(action)
    assert np.all(clipped <= reach_config.action_clip)
    assert np.all(clipped >= -reach_config.action_clip)
    delta = safety.joint_delta(action, reach_config.action_scale)
    assert np.all(np.abs(delta) <= reach_config.max_joint_delta_rad + 1e-9)
    np.testing.assert_allclose(delta[:3], [0.01, -0.01, 0.01])


def test_joint_limit_violation_raises(safety: SafetyGuard):
    with pytest.raises(SafetyViolation):
        safety.check_joint_limits(np.full(6, 100.0))


def test_ee_z_violation_raises(safety: SafetyGuard):
    with pytest.raises(SafetyViolation):
        safety.check_ee_position([0.3, 0.0, -0.01])


def test_sdk_position_m():
    np.testing.assert_allclose(sdk_position_m([400.0, 0.0, 300.0]), [0.4, 0.0, 0.3])


def test_apply_servo_joint_delta_uses_angles_kwarg(reach_config: ReachDeployConfig, safety: SafetyGuard):
    arm = MagicMock()
    arm.set_servo_angle_j.return_value = 0
    arm.mode = 1
    arm.state = 0
    q = np.zeros(6)
    apply_servo_joint_delta(arm, q, np.zeros(6), config=reach_config, safety=safety)
    assert "angles" in arm.set_servo_angle_j.call_args.kwargs
    assert "angle" not in arm.set_servo_angle_j.call_args.kwargs


def test_apply_servo_joint_delta_uses_limited_target(reach_config: ReachDeployConfig, safety: SafetyGuard):
    arm = MagicMock()
    arm.set_servo_angle_j.return_value = 0
    arm.mode = 1
    arm.state = 0
    q = np.zeros(6)
    q_cmd = apply_servo_joint_delta(arm, q, np.ones(6), config=reach_config, safety=safety)
    np.testing.assert_allclose(q_cmd, np.full(6, reach_config.max_joint_delta_rad))
    np.testing.assert_allclose(
        arm.set_servo_angle_j.call_args.kwargs["angles"],
        np.full(6, reach_config.max_joint_delta_rad),
    )


def test_apply_servo_joint_delta_retries_not_ready(reach_config: ReachDeployConfig, safety: SafetyGuard):
    arm = MagicMock()
    arm.set_servo_angle_j.side_effect = [9, 9, 0]
    arm.mode = 1
    arm.state = 0
    q = np.zeros(6)
    apply_servo_joint_delta(arm, q, np.zeros(6), config=reach_config, safety=safety)
    assert arm.set_servo_angle_j.call_count == 3


def test_apply_online_joint_delta_uses_mode6_set_servo_angle(
    online_reach_config: ReachDeployConfig,
    online_safety: SafetyGuard,
):
    arm = MagicMock()
    arm.set_servo_angle.return_value = 0
    arm.mode = 6
    arm.state = 0
    q = np.zeros(6)
    q_cmd = apply_online_joint_delta(arm, q, np.ones(6), config=online_reach_config, safety=online_safety)
    np.testing.assert_allclose(q_cmd, np.full(6, online_reach_config.max_joint_delta_rad))
    assert arm.set_servo_angle_j.call_count == 0
    assert "angle" in arm.set_servo_angle.call_args.kwargs
    assert arm.set_servo_angle.call_args.kwargs["wait"] is False
    assert arm.set_servo_angle.call_args.kwargs["is_radian"] is True
    np.testing.assert_allclose(
        arm.set_servo_angle.call_args.kwargs["angle"],
        np.full(6, online_reach_config.max_joint_delta_rad),
    )


def test_apply_reach_joint_delta_dispatches_online_joint(
    online_reach_config: ReachDeployConfig,
    online_safety: SafetyGuard,
):
    arm = MagicMock()
    arm.set_servo_angle.return_value = 0
    arm.mode = 6
    arm.state = 0
    q = np.zeros(6)
    apply_reach_joint_delta(arm, q, np.zeros(6), config=online_reach_config, safety=online_safety)
    assert arm.set_servo_angle.call_count == 1
    assert arm.set_servo_angle_j.call_count == 0


def test_validate_obs_shape(reach_config: ReachDeployConfig):
    validate_obs_shape(np.zeros(reach_config.num_obs), reach_config)
    with pytest.raises(ValueError):
        validate_obs_shape(np.zeros(reach_config.num_obs - 1), reach_config)
