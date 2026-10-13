"""Reach visualization with independent, non-blocking success holds.

The training environment's observations, reset and reward functions are reused.
Scene construction stays here because the training scene has no display layout
or deferred-viewer configuration hooks.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import genesis as gs
from .env import XArm6ReachEnv
from ufactory.dynamics.poses import parse_joint_limits
from ufactory.simulation import make_rigid_options
from ufactory.training.logic import reach_action_delta_torch
from ufactory.training.reach_config import REACH_ENV_VERSION, REACH_SUCCESS_DISTANCE_M


class RealtimeReachEnv(XArm6ReachEnv):
    def __init__(
        self, env_cfg: dict, reward_cfg: dict, robot_cfg: dict, *, success_hold_s: float = 1.0, robot_surface=None
    ) -> None:
        self.num_envs = int(env_cfg["num_envs"])
        self.num_obs = int(env_cfg["num_obs"])
        self.num_actions = int(env_cfg["num_actions"])
        self.num_privileged_obs = None
        self.device = gs.device
        self.cfg = self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if env_cfg.get("env_version") != REACH_ENV_VERSION or self.num_obs != 24 or self.num_actions != 6:
            raise ValueError("reach visualization requires the 24-observation, 6-action reach policy")
        if not math.isfinite(success_hold_s) or success_hold_s < 0:
            raise ValueError("success_hold_s must be finite and non-negative")

        self.ctrl_dt = float(env_cfg["ctrl_dt"])
        self.max_episode_length = math.ceil(float(env_cfg["episode_length_s"]) / self.ctrl_dt)
        self.action_scale = float(env_cfg["action_scale"])
        self.action_clip = float(env_cfg["action_clip"])
        self.max_joint_delta_rad = float(env_cfg["max_joint_delta_rad"])
        self.success_distance_m = float(env_cfg["success_distance_m"])
        if self.success_distance_m != REACH_SUCCESS_DISTANCE_M:
            raise ValueError("reach success_distance_m must be 0.005 (5 mm)")
        self.success_hold_s = success_hold_s
        self.hold_steps = math.ceil(success_hold_s / self.ctrl_dt)
        self.hold_steps_left = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
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
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.num_envs))),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                run_in_thread=False,
                refresh_rate=60,
                realtime_factor=None,
                camera_pos=(1.5, -1.5, 1.5),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=40,
            ),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_cfg["urdf_path"],
                pos=tuple(float(value) for value in robot_cfg["base_pos"]),
                fixed=True,
            ),
            surface=robot_surface,
        )
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02, fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
        )
        # Genesis offsets the rendered copies; physics and policy coordinates remain local.
        self.scene.build(n_envs=self.num_envs, env_spacing=(1.6, 1.6))

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
        self.extras = {"observations": {}}
        self.reset()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        super().reset_idx(envs_idx)
        self.hold_steps_left[envs_idx] = 0

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Action shape {tuple(actions.shape)} does not match ({self.num_envs}, {self.num_actions})"
            )
        if not torch.isfinite(actions).all():
            raise RuntimeError("policy emitted a non-finite action")

        holding = self.hold_steps_left > 0
        active = ~holding
        self.episode_length_buf += active.to(self.episode_length_buf.dtype)
        joint_delta = reach_action_delta_torch(
            actions,
            action_scale=self.action_scale,
            action_clip=self.action_clip,
            max_joint_delta_rad=self.max_joint_delta_rad,
        )
        self.position_command.copy_(
            torch.clamp(
                self.position_command + joint_delta * active[:, None],
                self.joint_lower,
                self.joint_upper,
            )
        )
        self.robot.control_dofs_position(self.position_command, self.dof_idx)
        # Exactly one physics tick, even when several arms finish together.
        self.scene.step(update_visualizer=False)
        self.metric_steps += active.to(self.metric_steps.dtype)
        self.metric_sums["action_saturation_rate"] += (actions.abs() >= 0.99 * self.action_clip).float().mean(
            -1
        ) * active
        self.metric_sums["command_tracking_error_rad"] += (
            self.position_command - self.robot.get_dofs_position(self.dof_idx)
        ).abs().mean(-1) * active

        distance_m = torch.linalg.norm(self.target_pos - self.ee_link.get_pos(), dim=-1)
        success = (distance_m < self.success_distance_m) & active
        timeout = (self.episode_length_buf >= self.max_episode_length) & active
        dones = success | timeout
        self.extras["time_outs"] = (timeout & ~success).to(dtype=gs.tc_float)
        reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            raw = success.float() if name == "success" else reward_func()
            term = raw * self.reward_scales[name] * active
            reward += term
            self.episode_sums[name] += term

        self.extras["episode"] = {}
        done_idx = dones.nonzero(as_tuple=False).flatten()
        if len(done_idx):
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
            self.extras["reach_done_envs_idx"] = done_idx.detach().clone()
            self.extras["reach_done_distance_m"] = done_distance
            self.extras["reach_done_success"] = done_success
            self.extras["reach_done_steps"] = self.episode_length_buf[done_idx].detach().clone()
        else:
            for name in ("envs_idx", "distance_m", "success", "steps"):
                self.extras.pop("reach_done_" + name, None)

        self.hold_steps_left[holding] -= 1
        expired = holding & (self.hold_steps_left == 0)
        if self.hold_steps:
            self.hold_steps_left[success] = self.hold_steps
            reset = expired | (timeout & ~success)
        else:
            reset = expired | dones
        self.reset_idx(reset.nonzero(as_tuple=False).flatten())
        return self.get_observations(), reward, dones, self.extras
