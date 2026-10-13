"""Bare-arm xArm6 reach environment.

Observation (24):
    joint position (6) + joint velocity (6) + end-effector position (3)
    + target relative to the end effector (3), world frame, meters
    + position command minus actual joint position (6).

Action (6):
    normalized joint-position increments, clipped by the action scale and the
    servo speed limit, accumulated into a position command within URDF limits.

An episode ends when the end effector is within the configured distance of the
target, or when the time limit is reached. The position command is sent on
every step.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import genesis as gs
from ufactory.dynamics.poses import parse_joint_limits
from ufactory.simulation import make_rigid_options
from ufactory.training.reach_config import REACH_ENV_VERSION, REACH_SUCCESS_DISTANCE_M
from ufactory.training.logic import action_penalty, reach_action_delta_torch, reach_observation, reward_reach_position


class XArm6ReachEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = int(env_cfg["num_envs"])
        self.num_obs = int(env_cfg["num_obs"])
        self.num_actions = int(env_cfg["num_actions"])
        self.num_privileged_obs = None
        self.device = gs.device
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg
        if env_cfg.get("env_version") != REACH_ENV_VERSION or self.num_obs != 24:
            raise ValueError("reach environment version changed; retrain the policy with 24 observations")

        self.ctrl_dt = float(env_cfg["ctrl_dt"])
        self.max_episode_length = math.ceil(float(env_cfg["episode_length_s"]) / self.ctrl_dt)
        self.action_scale = float(env_cfg["action_scale"])
        self.action_clip = float(env_cfg["action_clip"])
        self.max_joint_delta_rad = float(env_cfg["max_joint_delta_rad"])
        self.success_distance_m = float(env_cfg["success_distance_m"])
        if self.success_distance_m != REACH_SUCCESS_DISTANCE_M:
            raise ValueError("reach success_distance_m must be 0.005 (5 mm)")
        self.success_hold_s = 0.0
        self.target_pos_lower = torch.tensor(env_cfg["target_pos_lower"], device=self.device, dtype=gs.tc_float)
        self.target_pos_upper = torch.tensor(env_cfg["target_pos_upper"], device=self.device, dtype=gs.tc_float)

        self.reward_scales = reward_cfg.copy()
        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales:
            if name != "success":
                self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=int(env_cfg["substeps"])),
            rigid_options=make_rigid_options(
                gs,
                constraint_solver=env_cfg["constraint_solver"],
                friction_cone=env_cfg["friction_cone"],
                contact_resolution=env_cfg["contact_resolution"],
                enable_collision=True,
                enable_joint_limit=True,
                iterations=int(env_cfg["solver_iterations"]),
                noslip_iterations=int(env_cfg["noslip_iterations"]),
                constraint_timeconst=float(env_cfg["constraint_time_constant_s"]),
                use_gjk_collision=bool(env_cfg["use_gjk_collision"]),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.ctrl_dt),
                camera_pos=(1.5, -1.5, 1.5),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        # Genesis 1.4.4 hold at the default pose stays within a few milliradians
        # with a colliding ground plane, so the plane keeps collision.
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_cfg["urdf_path"],
                pos=tuple(float(value) for value in robot_cfg["base_pos"]),
                fixed=True,
            )
        )
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02, fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
        )
        self.scene.build(n_envs=self.num_envs)

        joint_map = {joint.name.split("/")[-1]: joint for joint in self.robot.joints}
        self.joint_names = list(robot_cfg["joint_names"])
        self.dof_idx = [joint_map[name].dofs_idx_local[0] for name in self.joint_names]
        self.ee_link = self.robot.get_link(robot_cfg["ee_link_name"])
        self.robot.set_dofs_kp(torch.tensor(robot_cfg["kp"], device=self.device), self.dof_idx)
        self.robot.set_dofs_kv(torch.tensor(robot_cfg["kv"], device=self.device), self.dof_idx)
        self.robot.set_dofs_force_range(
            torch.tensor(robot_cfg["force_lower"], device=self.device),
            torch.tensor(robot_cfg["force_upper"], device=self.device),
            self.dof_idx,
        )
        self.default_qpos = torch.tensor(robot_cfg["default_qpos"], dtype=gs.tc_float, device=self.device)
        lower, upper = parse_joint_limits(robot_cfg["urdf_path"], self.joint_names)
        self.joint_lower = torch.tensor(lower, device=self.device, dtype=gs.tc_float)
        self.joint_upper = torch.tensor(upper, device=self.device, dtype=gs.tc_float)
        if not torch.isfinite(self.joint_lower).all() or not torch.isfinite(self.joint_upper).all():
            raise ValueError("reach requires finite URDF limits for every arm joint")
        if torch.any(self.default_qpos < self.joint_lower) or torch.any(self.default_qpos > self.joint_upper):
            raise ValueError("reach default pose exceeds URDF joint limits")
        self.position_command = self.default_qpos.expand(self.num_envs, -1).clone()

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=gs.tc_float)
        self.metric_sums = {
            name: torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            for name in ("action_saturation_rate", "command_tracking_error_rad")
        }
        self.metric_steps = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.extras: dict = {"observations": {}}
        self.reset()

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations(), self.extras

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        count = len(envs_idx)
        home = self.default_qpos.unsqueeze(0).expand(count, -1).contiguous()
        self.robot.set_dofs_position(home, self.dof_idx, envs_idx=envs_idx, zero_velocity=True)
        self.robot.control_dofs_position(home, self.dof_idx, envs_idx=envs_idx)
        self.position_command[envs_idx] = home
        self.episode_length_buf[envs_idx] = 0
        self.metric_steps[envs_idx] = 0
        sampled = torch.rand(count, 3, device=self.device, dtype=gs.tc_float)
        self.target_pos[envs_idx] = self.target_pos_lower + sampled * (self.target_pos_upper - self.target_pos_lower)
        self.target_marker.set_pos(self.target_pos[envs_idx], envs_idx=envs_idx)
        for key in self.episode_sums:
            self.episode_sums[key][envs_idx] = 0.0
        for values in self.metric_sums.values():
            values[envs_idx] = 0.0

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Action shape {tuple(actions.shape)} does not match ({self.num_envs}, {self.num_actions})"
            )
        if not torch.isfinite(actions).all():
            raise RuntimeError("policy emitted a non-finite action")

        self.episode_length_buf += 1
        joint_delta = reach_action_delta_torch(
            actions,
            action_scale=self.action_scale,
            action_clip=self.action_clip,
            max_joint_delta_rad=self.max_joint_delta_rad,
        )
        self.position_command.copy_(
            torch.clamp(self.position_command + joint_delta, self.joint_lower, self.joint_upper)
        )
        self.robot.control_dofs_position(self.position_command, self.dof_idx)
        self.scene.step()
        self.metric_steps += 1
        self.metric_sums["action_saturation_rate"] += (actions.abs() >= 0.99 * self.action_clip).float().mean(-1)
        self.metric_sums["command_tracking_error_rad"] += (
            (self.position_command - self.robot.get_dofs_position(self.dof_idx)).abs().mean(-1)
        )

        ee_pos = self.ee_link.get_pos()
        distance_m = torch.linalg.norm(self.target_pos - ee_pos, dim=-1)
        success = distance_m < self.success_distance_m
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = success | timeout
        self.extras["time_outs"] = (timeout & ~success).to(dtype=gs.tc_float)

        reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            raw = success.float() if name == "success" else reward_func()
            term = raw * self.reward_scales[name]
            reward += term
            self.episode_sums[name] += term

        self.extras["episode"] = {}
        done_idx = dones.nonzero(as_tuple=False).flatten()
        if len(done_idx) > 0:
            done_distance = distance_m[done_idx].detach().clone()
            done_success = success[done_idx].detach().clone()
            self.extras["episode"] = {
                "success_rate": done_success.float(),
                "final_dist_m": done_distance,
                "elapsed_s": self.episode_length_buf[done_idx].float() * self.ctrl_dt,
            }
            self.extras["episode"].update(
                {"reward/" + name: values[done_idx].detach().clone() for name, values in self.episode_sums.items()}
            )
            self.extras["episode"].update(
                {
                    name: (values[done_idx] / self.metric_steps[done_idx]).detach().clone()
                    for name, values in self.metric_sums.items()
                }
            )
            self.extras["reach_done_distance_m"] = done_distance
            self.extras["reach_done_success"] = done_success
            self.extras["reach_done_steps"] = self.episode_length_buf[done_idx].detach().clone()
            self._hold_successful_pose(success)
            self.reset_idx(done_idx)
        else:
            self.extras.pop("reach_done_distance_m", None)
            self.extras.pop("reach_done_success", None)
            self.extras.pop("reach_done_steps", None)

        observation = self.get_observations()
        return observation, reward, dones, self.extras

    def get_observations(self) -> TensorDict:
        joint_pos = self.robot.get_dofs_position(self.dof_idx)
        joint_vel = self.robot.get_dofs_velocity(self.dof_idx)
        ee_pos = self.ee_link.get_pos()
        observation = reach_observation(joint_pos, joint_vel, ee_pos, self.target_pos, self.position_command)
        if observation.shape[-1] != self.num_obs:
            raise RuntimeError(f"Observation shape {observation.shape[-1]} does not match num_obs={self.num_obs}")
        self.extras["observations"] = {"policy": observation}
        return TensorDict({"policy": observation}, batch_size=[self.num_envs])

    def get_privileged_observations(self) -> None:
        return None

    def training_state_dict(self) -> dict:
        return {
            "env_version": REACH_ENV_VERSION,
            "joint_pos": self.robot.get_dofs_position(self.dof_idx).detach().cpu().clone(),
            "joint_vel": self.robot.get_dofs_velocity(self.dof_idx).detach().cpu().clone(),
            "position_command": self.position_command.detach().cpu().clone(),
            "episode_length_buf": self.episode_length_buf.detach().cpu().clone(),
            "target_pos": self.target_pos.detach().cpu().clone(),
            "episode_sums": {key: value.detach().cpu().clone() for key, value in self.episode_sums.items()},
            "metric_sums": {key: value.detach().cpu().clone() for key, value in self.metric_sums.items()},
            "metric_steps": self.metric_steps.detach().cpu().clone(),
        }

    def load_training_state_dict(self, state: dict) -> None:
        if state.get("env_version") != REACH_ENV_VERSION:
            raise ValueError("reach resume state is incompatible; retrain the policy")
        tensors = {
            name: state[name].to(self.device)
            for name in (
                "joint_pos",
                "joint_vel",
                "position_command",
                "episode_length_buf",
                "target_pos",
                "metric_steps",
            )
        }
        for name, expected in (
            ("joint_pos", self.position_command),
            ("joint_vel", self.position_command),
            ("position_command", self.position_command),
            ("episode_length_buf", self.episode_length_buf),
            ("target_pos", self.target_pos),
            ("metric_steps", self.metric_steps),
        ):
            if tensors[name].shape != expected.shape or not torch.isfinite(tensors[name]).all():
                raise ValueError(f"invalid reach resume state: {name}")
        if torch.any(tensors["position_command"] < self.joint_lower) or torch.any(
            tensors["position_command"] > self.joint_upper
        ):
            raise ValueError("reach resume command exceeds URDF limits")
        sums = {}
        for group in ("episode_sums", "metric_sums"):
            expected = getattr(self, group)
            if state[group].keys() != expected.keys():
                raise ValueError(f"invalid reach resume state: {group}")
            sums[group] = {key: value.to(self.device) for key, value in state[group].items()}
            if any(
                value.shape != self.metric_steps.shape or not torch.isfinite(value).all()
                for value in sums[group].values()
            ):
                raise ValueError(f"invalid reach resume state: {group}")
        self.robot.set_dofs_position(tensors["joint_pos"], self.dof_idx, zero_velocity=True)
        self.robot.set_dofs_velocity(tensors["joint_vel"], self.dof_idx)
        for name in ("position_command", "episode_length_buf", "target_pos", "metric_steps"):
            getattr(self, name).copy_(tensors[name])
        for group, values in sums.items():
            for key, value in values.items():
                getattr(self, group)[key].copy_(value)
        self.robot.control_dofs_position(self.position_command, self.dof_idx)
        self.target_marker.set_pos(self.target_pos)

    def _hold_successful_pose(self, success: torch.Tensor) -> None:
        """Keep the reached pose on screen. Extra steps are not part of the episode."""
        hold_steps = int(round(float(self.success_hold_s) / self.ctrl_dt))
        if hold_steps <= 0 or not bool(success.any()):
            return
        for _ in range(hold_steps):
            self.robot.control_dofs_position(self.position_command, self.dof_idx)
            self.scene.step()

    def _reward_reach(self) -> torch.Tensor:
        return reward_reach_position(self.ee_link.get_pos(), self.target_pos)

    def _reward_action_penalty(self) -> torch.Tensor:
        return action_penalty(self.robot.get_dofs_velocity(self.dof_idx))

    def _reward_success(self) -> torch.Tensor:
        # Success is computed before auto-reset in step().
        return (torch.linalg.norm(self.target_pos - self.ee_link.get_pos(), dim=-1) < self.success_distance_m).float()
