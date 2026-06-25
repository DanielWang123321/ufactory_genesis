"""
xArm 6 Reach Environment for RL training in Genesis.
Task: Move end-effector to a randomly sampled target position.

Observation (18-dim): joint_pos(6) + joint_vel(6) + ee_pos(3) + target_rel(3)
Action (6-dim): delta joint positions (scaled by action_scale)

Follows the GraspEnv pattern from examples/manipulation/grasp_env.py
"""

import math

import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.paths import xarm6_urdf

XARM6_URDF_PATH = xarm6_urdf()


class XArm6ReachEnv:
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

        # === Build scene ===
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs)))
            ),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.ctrl_dt),
                camera_pos=(1.5, -1.5, 1.5),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # Ground plane
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # xArm 6 robot (loaded from URDF)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=XARM6_URDF_PATH,
                pos=(0.0, 0.0, 0.0),
                fixed=True,
                requires_jac_and_IK=True,
            ),
        )

        # Visual target marker (green sphere, no collision)
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.02,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
            ),
        )

        # Build with batched environments
        self.scene.build(n_envs=self.num_envs)

        # === Robot setup (must be after scene.build) ===
        self.ee_link = self.robot.get_link(robot_cfg["ee_link_name"])

        joint_names = robot_cfg["joint_names"]
        self.dof_idx = [
            self.robot.get_joint(name).dofs_idx_local[0] for name in joint_names
        ]

        # Set PD gains
        self.robot.set_dofs_kp(
            torch.tensor(robot_cfg["kp"], device=self.device),
            self.dof_idx,
        )
        self.robot.set_dofs_kv(
            torch.tensor(robot_cfg["kv"], device=self.device),
            self.dof_idx,
        )
        self.robot.set_dofs_force_range(
            torch.tensor(robot_cfg["force_lower"], device=self.device),
            torch.tensor(robot_cfg["force_upper"], device=self.device),
            self.dof_idx,
        )

        self.default_qpos = torch.tensor(
            robot_cfg["default_qpos"], dtype=gs.tc_float, device=self.device
        )

        # Workspace bounds for target sampling
        self.target_pos_lower = torch.tensor(
            env_cfg["target_pos_lower"], device=self.device, dtype=gs.tc_float
        )
        self.target_pos_upper = torch.tensor(
            env_cfg["target_pos_upper"], device=self.device, dtype=gs.tc_float
        )

        # === Reward functions ===
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                self.num_envs, device=self.device, dtype=gs.tc_float
            )

        # === Buffers ===
        self._init_buffers()
        self.reset()

    def _init_buffers(self):
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.target_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float
        )
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

        # Reset robot to default pose
        default_qpos_batch = self.default_qpos.unsqueeze(0).repeat(n, 1)
        self.robot.set_qpos(default_qpos_batch, envs_idx=envs_idx)

        # Sample new random target position within workspace
        rand = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
        new_targets = (
            self.target_pos_lower
            + rand * (self.target_pos_upper - self.target_pos_lower)
        )
        self.target_pos[envs_idx] = new_targets
        self.target_marker.set_pos(new_targets, envs_idx=envs_idx)

        # Episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1

        # Actions: delta joint positions, scaled
        scaled_actions = actions * self.action_scale
        current_qpos = self.robot.get_dofs_position(self.dof_idx)
        target_qpos = current_qpos + scaled_actions
        self.robot.control_dofs_position(target_qpos, self.dof_idx)

        self.scene.step()

        # Check termination (time-out only)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, dtype=gs.tc_float
        )
        time_out_idx = self.reset_buf.nonzero(as_tuple=True)[0]
        self.extras["time_outs"][time_out_idx] = 1.0

        # Compute reward
        reward = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_float
        )
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # Reset timed-out environments
        if len(time_out_idx) > 0:
            self.reset_idx(time_out_idx)

        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        joint_pos = self.robot.get_dofs_position(self.dof_idx)  # (B, 6)
        joint_vel = self.robot.get_dofs_velocity(self.dof_idx)  # (B, 6)
        ee_pos = self.ee_link.get_pos()                         # (B, 3)
        target_rel = self.target_pos - ee_pos                   # (B, 3)

        obs = torch.cat([joint_pos, joint_vel, ee_pos, target_rel], dim=-1)
        self.extras["observations"] = {"critic": obs}
        return obs, self.extras

    def get_privileged_observations(self) -> None:
        return None

    # ------------ Reward functions ------------

    def _reward_reach(self) -> torch.Tensor:
        """Exponential reward based on EE-to-target distance."""
        ee_pos = self.ee_link.get_pos()
        dist = torch.norm(ee_pos - self.target_pos, dim=-1)
        return torch.exp(-10.0 * dist)

    def _reward_action_penalty(self) -> torch.Tensor:
        """Penalize large joint velocities for smooth motion."""
        joint_vel = self.robot.get_dofs_velocity(self.dof_idx)
        return -torch.sum(joint_vel**2, dim=-1)
