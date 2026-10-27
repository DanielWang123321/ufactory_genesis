"""Reach visualization with independent, non-blocking success holds.

Configuration, state, actions, statistics and rewards reuse the training environment.
Scene construction stays here because the training scene has no display layout
or deferred-viewer configuration hooks.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import genesis as gs
from .env import ReachEnv
from ufactory.simulation import make_rigid_options


class RealtimeReachEnv(ReachEnv):
    def __init__(
        self, env_cfg: dict, reward_cfg: dict, robot_cfg: dict, *, success_hold_s: float = 1.0, robot_surface=None
    ) -> None:
        if not math.isfinite(success_hold_s) or success_hold_s < 0:
            raise ValueError("success_hold_s must be finite and non-negative")
        self._init_config(env_cfg, reward_cfg, robot_cfg)
        self.success_hold_s = success_hold_s
        self.hold_steps = math.ceil(success_hold_s / self.ctrl_dt)
        self.hold_steps_left = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.robot_surface = robot_surface
        self._build_scene(show_viewer=False)
        self._init_state()
        self.reset()

    def _build_scene(self, *, show_viewer: bool) -> None:
        env_cfg, robot_cfg = self.env_cfg, self.robot_cfg
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
            surface=self.robot_surface,
        )
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02, fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
        )
        # Genesis offsets the rendered copies; physics and policy coordinates remain local.
        self.scene.build(n_envs=self.num_envs, env_spacing=(1.6, 1.6))

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        super().reset_idx(envs_idx)
        self.hold_steps_left[envs_idx] = 0

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        holding = self.hold_steps_left > 0
        active = ~holding
        self._advance(actions, active=active, update_visualizer=False)
        reward, dones, distance_m, success, timeout = self._outcome(active)
        self._record_completed(dones, distance_m, success)

        self.hold_steps_left[holding] -= 1
        expired = holding & (self.hold_steps_left == 0)
        if self.hold_steps:
            self.hold_steps_left[success] = self.hold_steps
            reset = expired | (timeout & ~success)
        else:
            reset = expired | dones
        self.reset_idx(reset.nonzero(as_tuple=False).flatten())
        return self.get_observations(), reward, dones, self.extras
