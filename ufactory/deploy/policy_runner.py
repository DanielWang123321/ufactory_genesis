"""Load rsl-rl reach policies for real-robot inference (no Genesis env)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from rsl_rl.modules import ActorCritic

from ufactory.deploy.action_postprocess import effective_max_joint_delta_rad
from ufactory.deploy.obs_adapter import validate_obs_shape
from ufactory.deploy.reach_config import (
    DEFAULT_SERVO_SPEED_RAD_S,
    EXECUTOR_ONLINE_JOINT,
    EXECUTOR_SERVO_J,
    ReachDeployConfig,
    normalize_reach_executor,
)


class ReachPolicyRunner:
    """Inference wrapper for a trained reach ActorCritic checkpoint."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: ReachDeployConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor_critic = actor_critic
        self.config = config
        self.device = torch.device(device)
        self.actor_critic.to(self.device)
        self.actor_critic.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cfgs_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> ReachPolicyRunner:
        ckpt_path = Path(checkpoint_path)
        cfg_path = Path(cfgs_path)
        with cfg_path.open("rb") as f:
            env_cfg, _reward_cfg, _robot_cfg, train_cfg = pickle.load(f)

        policy_cfg = train_cfg["policy"]
        actor_critic = ActorCritic(
            env_cfg["num_obs"],
            env_cfg["num_obs"],
            env_cfg["num_actions"],
            actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=policy_cfg["critic_hidden_dims"],
            activation=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
        )
        loaded = torch.load(ckpt_path, weights_only=False, map_location=device)
        actor_critic.load_state_dict(loaded["model_state_dict"])
        action_clip = float(env_cfg.get("action_clip", 1.0))
        executor = normalize_reach_executor(env_cfg.get("executor", EXECUTOR_SERVO_J))
        servo_speed_value = env_cfg.get("servo_speed_rad_s", DEFAULT_SERVO_SPEED_RAD_S)
        servo_speed_for_limit = None if servo_speed_value is None else float(servo_speed_value)
        if executor == EXECUTOR_ONLINE_JOINT:
            servo_speed_for_limit = None
        servo_speed_rad_s = DEFAULT_SERVO_SPEED_RAD_S if servo_speed_for_limit is None else servo_speed_for_limit
        action_scale = float(env_cfg["action_scale"])
        ctrl_dt = float(env_cfg["ctrl_dt"])
        max_joint_delta_rad = effective_max_joint_delta_rad(
            action_scale=action_scale,
            action_clip=action_clip,
            ctrl_dt=ctrl_dt,
            servo_speed_rad_s=servo_speed_for_limit,
            max_joint_delta_rad=env_cfg.get("max_joint_delta_rad"),
        )
        config = ReachDeployConfig(
            dof=env_cfg["num_actions"],
            num_obs=env_cfg["num_obs"],
            num_actions=env_cfg["num_actions"],
            action_scale=action_scale,
            ctrl_dt=ctrl_dt,
            z_min_m=0.0,
            action_clip=action_clip,
            max_joint_delta_rad=max_joint_delta_rad,
            servo_speed_rad_s=servo_speed_rad_s,
            servo_mvacc_rad_s2=5.0,
            ee_link="link6",
            executor=executor,
        )
        return cls(actor_critic, config, device=device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        validate_obs_shape(obs, self.config)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor_critic.act_inference(obs_t)
        return action.cpu().numpy().reshape(-1)
