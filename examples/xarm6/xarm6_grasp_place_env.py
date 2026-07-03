"""xArm6 + Gripper G2 grasp-place RL environment.

Observation (30-dim):
    q(6) + qd(6) + finger_center_base(3) + gripper_gap_m(1)
    + obj_pos_base(3) + target_pos_base(3)
    + ee_to_obj(3) + obj_to_target(3) + grasped(1) + ever_grasped(1)

Action (7-dim):
    normalized joint delta for six arm joints + normalized Gripper G2 gap delta.

The environment keeps object and target observations in robot base frame even
though Genesis entities live in world frame. The robot base is mounted at the
table surface height, so the current conversion is translation-only.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from genesis.utils.geom import xyz_to_quat
from ufactory.manipulation.frames import base_to_world_pos, world_to_base_pos
from ufactory.grippers.g2 import (
    GRIPPER_G2_OPEN_GAP_M,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
)


class XArm6GraspPlaceEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = int(env_cfg["num_envs"])
        self.num_obs = int(env_cfg["num_obs"])
        self.num_privileged_obs = None
        self.num_actions = int(env_cfg["num_actions"])
        self.device = gs.device

        self.ctrl_dt = float(env_cfg["ctrl_dt"])
        self.max_episode_length = math.ceil(float(env_cfg["episode_length_s"]) / self.ctrl_dt)

        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg.copy()
        self.action_scale = float(env_cfg["action_scale"])
        self.action_clip = float(env_cfg.get("action_clip", 1.0))
        self.max_joint_delta_rad = float(env_cfg["max_joint_delta_rad"])
        self.gripper_delta_m = float(env_cfg["gripper_delta_mm"]) / 1000.0

        self.table_height = float(env_cfg["table_height"])
        self.obj_size = tuple(float(v) for v in env_cfg["obj_size"])
        self.obj_size_t = torch.tensor(self.obj_size, device=self.device, dtype=gs.tc_float)
        self.obj_rest_z_base = self.obj_size[2] / 2.0
        self.grasp_center_offset_z = float(env_cfg.get("grasp_center_offset_z", 0.065))
        self.lift_height_m = float(env_cfg.get("lift_height_m", 0.08))
        self.place_success_dist_m = float(env_cfg.get("place_success_dist_m", 0.04))
        self.success_hold_steps = int(env_cfg.get("success_hold_steps", 10))

        self.gripper_open_gap_m = float(env_cfg["gripper_open_mm"]) / 1000.0
        self.gripper_close_gap_m = float(env_cfg["gripper_close_mm"]) / 1000.0
        if not math.isclose(self.gripper_open_gap_m, GRIPPER_G2_OPEN_GAP_M, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Gripper G2 open gap must match the 84 mm SDK contract")

        self.base_pos_world = torch.tensor(
            robot_cfg.get("base_pos", [0.0, 0.0, self.table_height]),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=env_cfg.get("substeps", 4)),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.ctrl_dt),
                camera_pos=(1.5, -1.5, 1.2),
                camera_lookat=(0.3, 0.0, self.table_height + 0.2),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.8, self.table_height),
                pos=(0.45, 0.0, self.table_height / 2.0),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6))),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_cfg["urdf_path"],
                pos=tuple(float(v) for v in self.base_pos_world.cpu().tolist()),
                fixed=True,
                requires_jac_and_IK=True,
            ),
        )
        self.obj = self.scene.add_entity(
            gs.morphs.Box(
                size=self.obj_size,
                pos=tuple(float(v) for v in self.base_to_world(self._cfg_vec("fixed_obj_pos")).cpu().tolist()),
                fixed=False,
            ),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1))),
        )
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02, fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
        )

        self.scene.build(n_envs=self.num_envs)

        self.ik_link = self.robot.get_link(robot_cfg["ik_link_name"])
        self.left_finger_link = self.robot.get_link(robot_cfg["gripper_link_names"][0])
        self.right_finger_link = self.robot.get_link(robot_cfg["gripper_link_names"][1])
        self.collision_monitor_links = [
            self.robot.get_link(name)
            for name in robot_cfg.get("collision_monitor_links", [])
        ]

        self.arm_joint_names = robot_cfg["arm_joint_names"]
        self.arm_dof_idx = [
            self.robot.get_joint(name).dofs_idx_local[0]
            for name in self.arm_joint_names
        ]
        self.gripper_joint_name = robot_cfg["gripper_joint_name"]
        self.gripper_dof_idx = [self.robot.get_joint(self.gripper_joint_name).dofs_idx_local[0]]
        self.all_dof_idx = self.arm_dof_idx + self.gripper_dof_idx

        self._setup_robot_control(robot_cfg)
        if env_cfg.get("stiffen_gripper_mimic", True):
            self._stiffen_gripper_mimic_constraints()

        self.default_gripper_drive = self.gap_m_to_drive_t(
            torch.full((self.num_envs,), self.gripper_open_gap_m, device=self.device, dtype=gs.tc_float)
        )[0].detach()
        self.default_arm_qpos = self._default_arm_qpos_from_ik(env_cfg)

        self.fixed_obj_pos = self._cfg_vec("fixed_obj_pos")
        self.fixed_target_pos = self._cfg_vec("fixed_target_pos")
        self.obj_spawn_lower = self._cfg_vec("obj_spawn_lower")
        self.obj_spawn_upper = self._cfg_vec("obj_spawn_upper")
        self.target_spawn_lower = self._cfg_vec("target_spawn_lower")
        self.target_spawn_upper = self._cfg_vec("target_spawn_upper")
        self.workspace_lower = self._cfg_vec("workspace_lower")
        self.workspace_upper = self._cfg_vec("workspace_upper")

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        self.csv_log_path = None
        self._csv_file = None
        self._csv_writer = None

        self._init_buffers()
        self.curriculum_stage = 0
        self._initial_reset_done = False

        hist_len = int(env_cfg.get("success_history_len", 2000))
        self.grasp_success_history = torch.zeros(hist_len, device=self.device)
        self.lift_success_history = torch.zeros(hist_len, device=self.device)
        self.place_success_history = torch.zeros(hist_len, device=self.device)
        self.success_history = torch.zeros(hist_len, device=self.device)
        self.grasp_history_idx = 0
        self.grasp_history_count = 0
        self.lift_history_idx = 0
        self.lift_history_count = 0
        self.place_history_idx = 0
        self.place_history_count = 0
        self.success_history_idx = 0
        self.success_history_count = 0

        self.reset()

    def _cfg_vec(self, key: str) -> torch.Tensor:
        return torch.tensor(self.env_cfg[key], device=self.device, dtype=gs.tc_float)

    def base_to_world(self, pos: torch.Tensor) -> torch.Tensor:
        return base_to_world_pos(pos, self.base_pos_world)

    def world_to_base(self, pos: torch.Tensor) -> torch.Tensor:
        return world_to_base_pos(pos, self.base_pos_world)

    def _setup_robot_control(self, robot_cfg: dict) -> None:
        self.robot.set_dofs_kp(torch.tensor(robot_cfg["kp"], device=self.device, dtype=gs.tc_float), self.arm_dof_idx)
        self.robot.set_dofs_kv(torch.tensor(robot_cfg["kv"], device=self.device, dtype=gs.tc_float), self.arm_dof_idx)
        self.robot.set_dofs_force_range(
            torch.tensor(robot_cfg["force_lower"], device=self.device, dtype=gs.tc_float),
            torch.tensor(robot_cfg["force_upper"], device=self.device, dtype=gs.tc_float),
            self.arm_dof_idx,
        )

        self.robot.set_dofs_kp(torch.tensor([robot_cfg["gripper_kp"]], device=self.device, dtype=gs.tc_float), self.gripper_dof_idx)
        self.robot.set_dofs_kv(torch.tensor([robot_cfg["gripper_kv"]], device=self.device, dtype=gs.tc_float), self.gripper_dof_idx)
        self.robot.set_dofs_force_range(
            torch.tensor([robot_cfg["gripper_force_lower"]], device=self.device, dtype=gs.tc_float),
            torch.tensor([robot_cfg["gripper_force_upper"]], device=self.device, dtype=gs.tc_float),
            self.gripper_dof_idx,
        )

        self.all_gripper_dof_idx = [
            self.robot.get_joint(name).dofs_idx_local[0]
            for name in robot_cfg["all_gripper_joint_names"]
        ]
        n_grip = len(self.all_gripper_dof_idx)
        self.robot.set_dofs_damping(
            torch.full((n_grip,), robot_cfg["gripper_damping"], device=self.device, dtype=gs.tc_float),
            self.all_gripper_dof_idx,
        )
        self.robot.set_dofs_frictionloss(
            torch.full((n_grip,), robot_cfg["gripper_frictionloss"], device=self.device, dtype=gs.tc_float),
            self.all_gripper_dof_idx,
        )

    def _stiffen_gripper_mimic_constraints(self) -> None:
        stiff_sol_params = np.array([0.01, 0.1, 0.0001, 0.001, 0.001, 0.5, 2.0])
        mimic_keywords = ("finger", "knuckle")
        for eq in self.robot.equalities:
            if any(keyword in eq.name for keyword in mimic_keywords):
                eq.set_sol_params(stiff_sol_params)

    def _default_arm_qpos_from_ik(self, env_cfg: dict) -> torch.Tensor:
        default_ee_base = torch.tensor(
            [env_cfg.get("default_ee_pos", [0.3, 0.0, 0.3])],
            device=self.device,
            dtype=gs.tc_float,
        ).expand(self.num_envs, 3)
        default_ee_world = self.base_to_world(default_ee_base)
        down_quat = xyz_to_quat(
            torch.tensor([[math.pi, 0.0, 0.0]], device=self.device, dtype=gs.tc_float),
            rpy=True,
            degrees=False,
        ).expand(self.num_envs, 4)
        init_qpos = self.robot.inverse_kinematics(
            link=self.ik_link,
            pos=default_ee_world,
            quat=down_quat,
            dofs_idx_local=self.arm_dof_idx,
        )
        return init_qpos[0, self.arm_dof_idx].detach()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=gs.tc_float)
        self.grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_place_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.place_stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.episode_action_sat_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_delta_sat_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_gripper_bound_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.extras = {"observations": {}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, self.extras = self.get_observations()
        return obs, self.extras

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return

        has_episode_steps = bool(torch.any(self.episode_length_buf[envs_idx] > 0).item())
        if self._initial_reset_done and has_episode_steps:
            self._record_episode_outcomes(envs_idx)
            self._maybe_update_curriculum()

        self._write_episode_extras(envs_idx)

        self.episode_length_buf[envs_idx] = 0
        self.grasped[envs_idx] = False
        self.ever_grasped[envs_idx] = False
        self.ever_lifted[envs_idx] = False
        self.episode_place_success[envs_idx] = False
        self.episode_success[envs_idx] = False
        self.place_stable_steps[envs_idx] = 0
        self.episode_action_sat_sum[envs_idx] = 0.0
        self.episode_delta_sat_sum[envs_idx] = 0.0
        self.episode_gripper_bound_sum[envs_idx] = 0.0

        n = len(envs_idx)
        default_qpos = torch.zeros(n, self.robot.n_dofs, device=self.device, dtype=gs.tc_float)
        for i, idx in enumerate(self.arm_dof_idx):
            default_qpos[:, idx] = self.default_arm_qpos[i]
        default_qpos[:, self.gripper_dof_idx[0]] = self.default_gripper_drive
        self.robot.set_qpos(default_qpos, envs_idx=envs_idx)

        obj_base, target_base = self._sample_object_and_target_base(n)
        obj_world = self.base_to_world(obj_base)
        target_world = self.base_to_world(target_base)
        obj_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float).expand(n, 4)
        self.obj.set_pos(obj_world, envs_idx=envs_idx, zero_velocity=True)
        self.obj.set_quat(obj_quat, envs_idx=envs_idx, zero_velocity=True)
        self.target_pos[envs_idx] = target_base
        self.target_marker.set_pos(target_world, envs_idx=envs_idx)

        if self.csv_log_path is not None:
            self._write_csv_log()
        self._initial_reset_done = True

    def _sample_object_and_target_base(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.curriculum_stage == 0:
            obj = self.fixed_obj_pos.unsqueeze(0).expand(n, 3).clone()
            target = self.fixed_target_pos.unsqueeze(0).expand(n, 3).clone()
            return obj, target

        rand_obj = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
        obj = self.obj_spawn_lower + rand_obj * (self.obj_spawn_upper - self.obj_spawn_lower)
        obj[:, 2] = self.obj_rest_z_base

        if self.curriculum_stage == 1:
            target = obj.clone()
            target[:, 0] += 0.10
        elif self.curriculum_stage == 2:
            rand = torch.rand(n, 2, device=self.device, dtype=gs.tc_float)
            target = obj.clone()
            target[:, 0] += 0.05 + rand[:, 0] * 0.10
            target[:, 1] += (rand[:, 1] - 0.5) * 0.06
        elif self.curriculum_stage == 3:
            rand = torch.rand(n, 2, device=self.device, dtype=gs.tc_float)
            target = obj.clone()
            target[:, 0] += 0.10 + rand[:, 0] * 0.15
            target[:, 1] += (rand[:, 1] - 0.5) * 0.10
        else:
            rand_target = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
            target = self.target_spawn_lower + rand_target * (self.target_spawn_upper - self.target_spawn_lower)
        target[:, 2] = self.obj_rest_z_base
        return obj, target

    def _record_episode_outcomes(self, envs_idx: torch.Tensor) -> None:
        for env_idx_t in envs_idx:
            idx = int(env_idx_t.item())
            self.grasp_success_history[self.grasp_history_idx] = self.ever_grasped[idx].float()
            self.grasp_history_idx = (self.grasp_history_idx + 1) % len(self.grasp_success_history)
            self.grasp_history_count = min(self.grasp_history_count + 1, len(self.grasp_success_history))

            self.lift_success_history[self.lift_history_idx] = self.ever_lifted[idx].float()
            self.lift_history_idx = (self.lift_history_idx + 1) % len(self.lift_success_history)
            self.lift_history_count = min(self.lift_history_count + 1, len(self.lift_success_history))

            self.place_success_history[self.place_history_idx] = self.episode_place_success[idx].float()
            self.place_history_idx = (self.place_history_idx + 1) % len(self.place_success_history)
            self.place_history_count = min(self.place_history_count + 1, len(self.place_success_history))

            self.success_history[self.success_history_idx] = self.episode_success[idx].float()
            self.success_history_idx = (self.success_history_idx + 1) % len(self.success_history)
            self.success_history_count = min(self.success_history_count + 1, len(self.success_history))

    def _maybe_update_curriculum(self) -> None:
        if self.grasp_history_count >= 500:
            grasp_rate = self._history_rate(self.grasp_success_history, self.grasp_history_count)
            if self.curriculum_stage == 0 and grasp_rate > 0.50:
                self.curriculum_stage = 1
                self.grasp_history_count = 0
                print(f"[Curriculum] Stage 0 -> 1 (grasp_rate={grasp_rate:.2f}): narrow random spawn")
            elif self.curriculum_stage == 1 and grasp_rate > 0.70:
                self.curriculum_stage = 2
                self.grasp_history_count = 0
                self.place_history_count = 0
                print(f"[Curriculum] Stage 1 -> 2 (grasp_rate={grasp_rate:.2f}): close target placement")

        if self.place_history_count >= 500:
            place_rate = self._history_rate(self.place_success_history, self.place_history_count)
            if self.curriculum_stage == 2 and place_rate > 0.60:
                self.curriculum_stage = 3
                self.place_history_count = 0
                print(f"[Curriculum] Stage 2 -> 3 (place_rate={place_rate:.2f}): medium target distance")
            elif self.curriculum_stage == 3 and place_rate > 0.50:
                self.curriculum_stage = 4
                print(f"[Curriculum] Stage 3 -> 4 (place_rate={place_rate:.2f}): full range target")

    def _history_rate(self, history: torch.Tensor, count: int) -> float:
        if count <= 0:
            return 0.0
        return float(history[:count].mean().item())

    def _write_episode_extras(self, envs_idx: torch.Tensor) -> None:
        steps = self.episode_length_buf[envs_idx].float().clamp(min=1.0)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.extras["episode"]["curriculum_stage"] = self.curriculum_stage
        self.extras["episode"]["grasp_success_rate"] = self._history_rate(
            self.grasp_success_history,
            self.grasp_history_count,
        )
        self.extras["episode"]["lift_success_rate"] = self._history_rate(
            self.lift_success_history,
            self.lift_history_count,
        )
        self.extras["episode"]["place_success_rate"] = self._history_rate(
            self.place_success_history,
            self.place_history_count,
        )
        self.extras["episode"]["success_rate"] = self._history_rate(
            self.success_history,
            self.success_history_count,
        )
        self.extras["episode"]["action_saturation_fraction"] = torch.mean(
            self.episode_action_sat_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["delta_saturation_fraction"] = torch.mean(
            self.episode_delta_sat_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["gripper_bound_fraction"] = torch.mean(
            self.episode_gripper_bound_sum[envs_idx] / steps
        ).item()

    def _write_csv_log(self) -> None:
        ep = self.extras.get("episode", {})
        if not ep:
            return
        row = {
            "curriculum_stage": ep.get("curriculum_stage", self.curriculum_stage),
            "grasp_success_rate": ep.get("grasp_success_rate", 0.0),
            "lift_success_rate": ep.get("lift_success_rate", 0.0),
            "place_success_rate": ep.get("place_success_rate", 0.0),
            "success_rate": ep.get("success_rate", 0.0),
            "action_saturation_fraction": ep.get("action_saturation_fraction", 0.0),
            "delta_saturation_fraction": ep.get("delta_saturation_fraction", 0.0),
            "gripper_bound_fraction": ep.get("gripper_bound_fraction", 0.0),
        }
        for key in self.reward_scales.keys():
            row["rew_" + key] = ep.get("rew_" + key, 0.0)

        if self._csv_writer is None:
            write_header = not os.path.exists(self.csv_log_path)
            self._csv_file = open(self.csv_log_path, "a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            if write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def write_metrics_snapshot(self) -> None:
        self._write_csv_log()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Action shape {tuple(actions.shape)} does not match ({self.num_envs}, {self.num_actions})")

        self.episode_length_buf += 1
        raw_actions = actions
        clipped_actions = torch.clamp(raw_actions, -self.action_clip, self.action_clip)

        current_q = self.robot.get_dofs_position(self.arm_dof_idx)
        joint_delta_unclipped = clipped_actions[:, :6] * self.action_scale
        joint_delta = torch.clamp(joint_delta_unclipped, -self.max_joint_delta_rad, self.max_joint_delta_rad)
        target_q = current_q + joint_delta

        current_gap = self.gripper_gap_m()
        gripper_delta = clipped_actions[:, 6] * self.gripper_delta_m
        target_gap = torch.clamp(
            current_gap + gripper_delta,
            min=self.gripper_close_gap_m,
            max=self.gripper_open_gap_m,
        )
        target_drive = self.gap_m_to_drive_t(target_gap)

        self._accumulate_action_stats(raw_actions, joint_delta_unclipped, joint_delta, target_gap)
        self.robot.control_dofs_position(target_q, self.arm_dof_idx)
        self.robot.control_dofs_position(target_drive.unsqueeze(-1), self.gripper_dof_idx)
        self.scene.step()

        self._update_task_state()

        timeout_buf = self.episode_length_buf > self.max_episode_length
        done_buf = timeout_buf | self.episode_success
        self.reset_buf = done_buf
        self.extras["time_outs"] = timeout_buf.float()

        reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        self.extras["episode_grasp_success"] = self.ever_grasped.clone()
        self.extras["episode_lift_success"] = self.ever_lifted.clone()
        self.extras["episode_place_success"] = self.episode_place_success.clone()
        self.extras["episode_success"] = self.episode_success.clone()

        done_idx = done_buf.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.reset_idx(done_idx)

        obs, self.extras = self.get_observations()
        return obs, reward, done_buf, self.extras

    def _accumulate_action_stats(
        self,
        raw_actions: torch.Tensor,
        joint_delta_unclipped: torch.Tensor,
        joint_delta: torch.Tensor,
        target_gap: torch.Tensor,
    ) -> None:
        eps = 1e-6
        action_sat = (raw_actions.abs() >= self.action_clip - eps).float().mean(dim=-1)
        delta_sat = (joint_delta_unclipped.abs() >= self.max_joint_delta_rad - eps).float().mean(dim=-1)
        gripper_bound = (
            (target_gap <= self.gripper_close_gap_m + eps)
            | (target_gap >= self.gripper_open_gap_m - eps)
        ).float()
        self.episode_action_sat_sum += action_sat
        self.episode_delta_sat_sum += delta_sat
        self.episode_gripper_bound_sum += gripper_bound

    def _update_task_state(self) -> None:
        obj_base = self.obj_pos_base()
        ee_base = self.finger_center_base()
        gap = self.gripper_gap_m()
        obj_vel = self.obj.get_vel()

        lifted = obj_base[:, 2] > (self.obj_rest_z_base + 0.02)
        object_width = self.obj_size[1]
        gripper_holding_gap = gap < (object_width + 0.02)
        ee_near_obj = torch.norm(ee_base - obj_base, dim=-1) < 0.07

        self.grasped = lifted & gripper_holding_gap & ee_near_obj
        self.ever_lifted = self.ever_lifted | lifted
        self.ever_grasped = self.ever_grasped | self.grasped

        xy_dist = torch.norm(obj_base[:, :2] - self.target_pos[:, :2], dim=-1)
        at_table = torch.abs(obj_base[:, 2] - self.obj_rest_z_base) < 0.025
        released = gap > (object_width + 0.02)
        stable = torch.norm(obj_vel, dim=-1) < 0.15
        place_candidate = (
            (xy_dist < self.place_success_dist_m)
            & at_table
            & released
            & stable
            & self.ever_grasped
        )
        self.place_stable_steps = torch.where(
            place_candidate,
            self.place_stable_steps + 1,
            torch.zeros_like(self.place_stable_steps),
        )
        self.episode_place_success = self.episode_place_success | place_candidate
        self.episode_success = self.episode_success | (self.place_stable_steps >= self.success_hold_steps)

    def drive_to_gap_m_t(self, drive: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(drive, 0.0, GRIPPER_G2_SIM_CLOSE_DRIVE)
        return self.gripper_open_gap_m * (1.0 - clipped / GRIPPER_G2_SIM_CLOSE_DRIVE)

    def gap_m_to_drive_t(self, gap_m: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(gap_m, self.gripper_close_gap_m, self.gripper_open_gap_m)
        return GRIPPER_G2_SIM_CLOSE_DRIVE * (1.0 - clipped / self.gripper_open_gap_m)

    def gripper_gap_m(self) -> torch.Tensor:
        drive = self.robot.get_dofs_position(self.gripper_dof_idx).squeeze(-1)
        return self.drive_to_gap_m_t(drive)

    def finger_center_world(self) -> torch.Tensor:
        return (self.left_finger_link.get_pos() + self.right_finger_link.get_pos()) / 2.0

    def finger_center_base(self) -> torch.Tensor:
        return self.world_to_base(self.finger_center_world())

    def obj_pos_base(self) -> torch.Tensor:
        return self.world_to_base(self.obj.get_pos())

    def desired_grasp_pos_base(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        offset = torch.tensor([0.0, 0.0, self.grasp_center_offset_z], device=self.device, dtype=gs.tc_float)
        return obj + offset

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        joint_pos = self.robot.get_dofs_position(self.arm_dof_idx)
        joint_vel = self.robot.get_dofs_velocity(self.arm_dof_idx)
        ee_base = self.finger_center_base()
        gripper_gap = self.gripper_gap_m().unsqueeze(-1)
        obj_base = self.obj_pos_base()
        target_base = self.target_pos
        ee_to_obj = obj_base - ee_base
        obj_to_target = target_base - obj_base
        obs = torch.cat(
            [
                joint_pos,
                joint_vel,
                ee_base,
                gripper_gap,
                obj_base,
                target_base,
                ee_to_obj,
                obj_to_target,
                self.grasped.unsqueeze(-1).float(),
                self.ever_grasped.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        if obs.shape[-1] != self.num_obs:
            raise RuntimeError(f"Observation shape {obs.shape[-1]} does not match num_obs={self.num_obs}")
        self.extras["observations"] = {"critic": obs}
        return obs, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def _reward_reach(self) -> torch.Tensor:
        dist = torch.norm(self.finger_center_base() - self.desired_grasp_pos_base(), dim=-1)
        return (1.0 / (1.0 + 8.0 * dist)) * (~self.ever_grasped).float()

    def _reward_align(self) -> torch.Tensor:
        ee = self.finger_center_base()
        grasp = self.desired_grasp_pos_base()
        xy_dist = torch.norm(ee[:, :2] - grasp[:, :2], dim=-1)
        z_diff = torch.abs(ee[:, 2] - grasp[:, 2])
        return torch.exp(-25.0 * xy_dist) * torch.exp(-25.0 * z_diff) * (~self.ever_grasped).float()

    def _reward_close_gripper(self) -> torch.Tensor:
        dist = torch.norm(self.finger_center_base() - self.desired_grasp_pos_base(), dim=-1)
        proximity = torch.clamp(1.0 - dist / 0.10, min=0.0)
        closed_fraction = (1.0 - self.gripper_gap_m() / self.gripper_open_gap_m).clamp(0.0, 1.0)
        return proximity * closed_fraction * (~self.ever_grasped).float()

    def _reward_lift(self) -> torch.Tensor:
        obj_z = self.obj_pos_base()[:, 2]
        height_gain = (obj_z - self.obj_rest_z_base).clamp(0.0, self.lift_height_m) / self.lift_height_m
        holding_gap = (self.gripper_gap_m() < (self.obj_size[1] + 0.02)).float()
        return height_gain * holding_gap

    def _reward_grasp(self) -> torch.Tensor:
        xy_dist = torch.norm(self.obj_pos_base()[:, :2] - self.target_pos[:, :2], dim=-1)
        return self.grasped.float() * (xy_dist > self.place_success_dist_m).float()

    def _reward_place(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        xy_dist = torch.norm(obj[:, :2] - self.target_pos[:, :2], dim=-1)
        transport = (1.0 / (1.0 + 8.0 * xy_dist)) * self.grasped.float()
        at_target = (xy_dist < self.place_success_dist_m).float()
        at_table = torch.exp(-30.0 * torch.abs(obj[:, 2] - self.obj_rest_z_base))
        lower = at_target * at_table * self.grasped.float()
        return transport + lower

    def _reward_release(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        xy_dist = torch.norm(obj[:, :2] - self.target_pos[:, :2], dim=-1)
        near_target = (xy_dist < self.place_success_dist_m).float()
        at_table = torch.exp(-30.0 * torch.abs(obj[:, 2] - self.obj_rest_z_base))
        open_fraction = (self.gripper_gap_m() / self.gripper_open_gap_m).clamp(0.0, 1.0)
        return near_target * at_table * open_fraction * self.ever_grasped.float()

    def _reward_success(self) -> torch.Tensor:
        return self.episode_success.float() * 10.0

    def _reward_action_penalty(self) -> torch.Tensor:
        joint_vel = self.robot.get_dofs_velocity(self.arm_dof_idx)
        return -torch.sum(joint_vel**2, dim=-1)

    def _reward_table_collision(self) -> torch.Tensor:
        penalty = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        margin = 0.04
        table_top_z = self.base_pos_world[2]
        for link in self.collision_monitor_links:
            link_z = link.get_pos()[:, 2]
            violation = torch.clamp((table_top_z + margin - link_z) / margin, min=0.0)
            penalty += violation
        return -penalty

    def _reward_workspace_violation(self) -> torch.Tensor:
        ee = self.finger_center_base()
        obj = self.obj_pos_base()
        ee_low = (ee < self.workspace_lower).float().sum(dim=-1)
        ee_high = (ee > self.workspace_upper).float().sum(dim=-1)
        obj_low = (obj < self.workspace_lower).float().sum(dim=-1)
        obj_high = (obj > self.workspace_upper).float().sum(dim=-1)
        dropped = (obj[:, 2] < -0.02).float()
        return -(ee_low + ee_high + obj_low + obj_high + dropped)
