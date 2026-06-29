"""Generic arm reach environment for UFACTORY robots in Genesis."""

from __future__ import annotations

import math

import torch

import genesis as gs


class ArmReachEnv:
    """Move an arm end-effector to a sampled target position."""

    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg.copy()
        self.action_scale = env_cfg["action_scale"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
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
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_cfg["urdf_path"],
                pos=tuple(robot_cfg.get("base_pos", (0.0, 0.0, 0.0))),
                fixed=True,
                requires_jac_and_IK=True,
            ),
        )
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02, fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
        )
        self.scene.build(n_envs=self.num_envs)

        self.ee_link = self.robot.get_link(robot_cfg["ee_link_name"])
        self.joint_names = robot_cfg["joint_names"]
        self.dof_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names]
        self.robot.set_dofs_kp(torch.tensor(robot_cfg["kp"], device=self.device), self.dof_idx)
        self.robot.set_dofs_kv(torch.tensor(robot_cfg["kv"], device=self.device), self.dof_idx)
        self.robot.set_dofs_force_range(
            torch.tensor(robot_cfg["force_lower"], device=self.device),
            torch.tensor(robot_cfg["force_upper"], device=self.device),
            self.dof_idx,
        )

        self.default_qpos = torch.tensor(robot_cfg["default_qpos"], dtype=gs.tc_float, device=self.device)
        self.target_pos_lower = torch.tensor(env_cfg["target_pos_lower"], device=self.device, dtype=gs.tc_float)
        self.target_pos_upper = torch.tensor(env_cfg["target_pos_upper"], device=self.device, dtype=gs.tc_float)

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=gs.tc_float)
        self.extras = {"observations": {}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, self.extras = self.get_observations()
        return obs, self.extras

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0
        n = len(envs_idx)
        self.robot.set_qpos(self.default_qpos.unsqueeze(0).repeat(n, 1), envs_idx=envs_idx)

        rand = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
        new_targets = self.target_pos_lower + rand * (self.target_pos_upper - self.target_pos_lower)
        self.target_pos[envs_idx] = new_targets
        self.target_marker.set_pos(new_targets, envs_idx=envs_idx)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1
        current_qpos = self.robot.get_dofs_position(self.dof_idx)
        self.robot.control_dofs_position(current_qpos + actions * self.action_scale, self.dof_idx)
        self.scene.step()

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=gs.tc_float)
        time_out_idx = self.reset_buf.nonzero(as_tuple=True)[0]
        self.extras["time_outs"][time_out_idx] = 1.0

        reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        if len(time_out_idx) > 0:
            self.reset_idx(time_out_idx)

        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        joint_pos = self.robot.get_dofs_position(self.dof_idx)
        joint_vel = self.robot.get_dofs_velocity(self.dof_idx)
        ee_pos = self.ee_link.get_pos()
        target_rel = self.target_pos - ee_pos
        obs = torch.cat([joint_pos, joint_vel, ee_pos, target_rel], dim=-1)
        self.extras["observations"] = {"critic": obs}
        return obs, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def _reward_reach(self) -> torch.Tensor:
        dist = torch.norm(self.ee_link.get_pos() - self.target_pos, dim=-1)
        return torch.exp(-10.0 * dist)

    def _reward_action_penalty(self) -> torch.Tensor:
        joint_vel = self.robot.get_dofs_velocity(self.dof_idx)
        return -torch.sum(joint_vel**2, dim=-1)

