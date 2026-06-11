"""
xArm 6 Grasp-Place Environment for RL training in Genesis.
Task: Grasp a cube from the table and place it at a random target location.

Observation (22-dim):
    ee_pos(3) + gripper_pos(1) + obj_pos(3) + obj_quat(4)
    + target_pos(3) + obj_to_ee(3) + obj_to_target(3) + grasped(1) + ever_grasped(1)

Action (4-dim): delta EE position (3) + gripper command (1)

Uses Cartesian space control with Genesis IK, following grasp_env.py pattern.
"""

import csv
import math
import os

import torch

import _bootstrap  # noqa: F401
import genesis as gs
from genesis.utils.geom import xyz_to_quat
from ufactory.paths import xarm6_urdf

XARM6_GRIPPER_URDF = xarm6_urdf("xarm6_with_gripper.urdf")


class XArm6GraspPlaceEnv:
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
        self.action_scales = torch.tensor(
            env_cfg["action_scales"], device=gs.device, dtype=gs.tc_float,
        )
        self.table_height = env_cfg["table_height"]
        self.obj_size = env_cfg["obj_size"]

        # === Build scene ===
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=env_cfg.get("substeps", 4)),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs))),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(1.5, -1.5, 1.2),
                camera_lookat=(0.3, 0.0, self.table_height + 0.2),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # Ground plane
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

        # Table (in front of the robot, surface at z = table_height)
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.8, self.table_height),
                pos=(0.45, 0.0, self.table_height / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6)),
            ),
        )

        # Robot (xArm6 + gripper), base mounted at table height
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=XARM6_GRIPPER_URDF,
                pos=(0.0, 0.0, self.table_height),
                fixed=True,
                requires_jac_and_IK=True,
            ),
        )

        # Object to grasp (red cube)
        half = self.obj_size[2] / 2
        self.obj = self.scene.add_entity(
            gs.morphs.Box(
                size=tuple(self.obj_size),
                pos=(0.35, 0.0, self.table_height + half),
                fixed=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1)),
            ),
        )

        # Place target marker (green sphere, no collision)
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

        # === Robot setup (after scene.build) ===
        # Note: link_tcp / xarm_gripper_base_link are merged into link6 by
        # Genesis (fixed joints are collapsed). Use link6 for IK and compute
        # finger center pose for observations.
        self.ik_link = self.robot.get_link(robot_cfg["ik_link_name"])
        self.left_finger_link = self.robot.get_link(robot_cfg["gripper_link_names"][0])
        self.right_finger_link = self.robot.get_link(robot_cfg["gripper_link_names"][1])

        # Arm links to monitor for table collision (should NOT touch the table)
        self.collision_monitor_links = [
            self.robot.get_link(name)
            for name in robot_cfg.get("collision_monitor_links", [])
        ]

        # Arm joint indices
        self.arm_joint_names = robot_cfg["arm_joint_names"]
        self.arm_dof_idx = [
            self.robot.get_joint(name).dofs_idx_local[0]
            for name in self.arm_joint_names
        ]

        # Gripper joint index (drive_joint only, mimic handled by solver)
        self.gripper_joint_name = robot_cfg["gripper_joint_name"]
        self.gripper_dof_idx = [
            self.robot.get_joint(self.gripper_joint_name).dofs_idx_local[0],
        ]

        # All controlled DOF indices (arm + gripper drive)
        self.all_dof_idx = self.arm_dof_idx + self.gripper_dof_idx

        # PD gains for arm
        arm_kp = torch.tensor(robot_cfg["kp"], device=self.device, dtype=gs.tc_float)
        arm_kv = torch.tensor(robot_cfg["kv"], device=self.device, dtype=gs.tc_float)
        self.robot.set_dofs_kp(arm_kp, self.arm_dof_idx)
        self.robot.set_dofs_kv(arm_kv, self.arm_dof_idx)
        self.robot.set_dofs_force_range(
            torch.tensor(robot_cfg["force_lower"], device=self.device, dtype=gs.tc_float),
            torch.tensor(robot_cfg["force_upper"], device=self.device, dtype=gs.tc_float),
            self.arm_dof_idx,
        )

        # PD gains for gripper
        gripper_kp = torch.tensor([robot_cfg["gripper_kp"]], device=self.device, dtype=gs.tc_float)
        gripper_kv = torch.tensor([robot_cfg["gripper_kv"]], device=self.device, dtype=gs.tc_float)
        self.robot.set_dofs_kp(gripper_kp, self.gripper_dof_idx)
        self.robot.set_dofs_kv(gripper_kv, self.gripper_dof_idx)

        # Gripper force range (limits max torque to prevent penetration/flying)
        if "gripper_force_lower" in robot_cfg:
            self.robot.set_dofs_force_range(
                torch.tensor([robot_cfg["gripper_force_lower"]], device=self.device, dtype=gs.tc_float),
                torch.tensor([robot_cfg["gripper_force_upper"]], device=self.device, dtype=gs.tc_float),
                self.gripper_dof_idx,
            )

        # Override damping/frictionloss for all gripper DOFs (drive + mimic joints)
        if "all_gripper_joint_names" in robot_cfg:
            all_gripper_dof_idx = [
                self.robot.get_joint(n).dofs_idx_local[0]
                for n in robot_cfg["all_gripper_joint_names"]
            ]
            n_grip = len(all_gripper_dof_idx)
            self.robot.set_dofs_damping(
                torch.full((n_grip,), robot_cfg["gripper_damping"],
                           device=self.device, dtype=gs.tc_float),
                all_gripper_dof_idx,
            )
            self.robot.set_dofs_frictionloss(
                torch.full((n_grip,), robot_cfg["gripper_frictionloss"],
                           device=self.device, dtype=gs.tc_float),
                all_gripper_dof_idx,
            )

        self.default_gripper_pos = robot_cfg["default_gripper_pos"]

        # Compute default arm qpos from Cartesian pose via IK
        # Target EE pose: [300mm, 0, 300mm] relative to robot base, roll=180° (gripper pointing down)
        ee_target_pos = torch.tensor(
            [[0.3, 0.0, self.table_height + 0.3]],
            device=self.device, dtype=gs.tc_float,
        ).expand(self.num_envs, 3)
        ee_target_quat = xyz_to_quat(
            torch.tensor([[math.pi, 0.0, 0.0]], device=self.device, dtype=gs.tc_float),
            rpy=True, degrees=False,
        ).expand(self.num_envs, 4)
        init_qpos = self.robot.inverse_kinematics(
            link=self.ik_link,
            pos=ee_target_pos,
            quat=ee_target_quat,
            dofs_idx_local=self.arm_dof_idx,
        )
        self.default_arm_qpos = init_qpos[0, self.arm_dof_idx].detach()
        self.gripper_open_pos = 0.0    # drive_joint=0 → fingers open (84mm)
        self.gripper_close_pos = 0.85  # drive_joint=0.85 → fingers closed (0mm)

        # Desired EE orientation: gripper pointing down (roll=π)
        self.desired_down_quat = xyz_to_quat(
            torch.tensor([[math.pi, 0.0, 0.0]], device=self.device, dtype=gs.tc_float),
            rpy=True, degrees=False,
        ).squeeze(0)  # (4,)
        self.desired_down_quat_batch = self.desired_down_quat.unsqueeze(0).expand(self.num_envs, 4)

        # Workspace bounds
        self.obj_spawn_lower = torch.tensor(
            env_cfg["obj_spawn_lower"], device=self.device, dtype=gs.tc_float,
        )
        self.obj_spawn_upper = torch.tensor(
            env_cfg["obj_spawn_upper"], device=self.device, dtype=gs.tc_float,
        )
        self.target_spawn_lower = torch.tensor(
            env_cfg["target_spawn_lower"], device=self.device, dtype=gs.tc_float,
        )
        self.target_spawn_upper = torch.tensor(
            env_cfg["target_spawn_upper"], device=self.device, dtype=gs.tc_float,
        )

        # === Reward functions ===
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                self.num_envs, device=self.device, dtype=gs.tc_float,
            )

        # === CSV logging ===
        self.csv_log_path = None  # set by train.py to enable CSV logging
        self._csv_file = None
        self._csv_writer = None

        # === Buffers ===
        self._init_buffers()

        # === Curriculum learning ===
        self.curriculum_stage = 0
        self.grasp_success_history = torch.zeros(2000, device=self.device)
        self.grasp_history_idx = 0
        self.grasp_history_count = 0
        self.place_success_history = torch.zeros(2000, device=self.device)
        self.place_history_idx = 0
        self.place_history_count = 0

        self.reset()

    def _init_buffers(self):
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_int,
        )
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self.target_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float,
        )
        self.grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self.ever_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self.episode_place_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
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

        n = len(envs_idx)

        # --- Curriculum: track grasp/place success and check for stage upgrade ---
        if self.grasp_history_count > 0 or n < self.num_envs:
            # Record grasp outcomes for finishing episodes (skip initial full reset)
            for i in range(n):
                self.grasp_success_history[self.grasp_history_idx] = self.ever_grasped[envs_idx[i]].float()
                self.grasp_history_idx = (self.grasp_history_idx + 1) % len(self.grasp_success_history)
                self.grasp_history_count = min(self.grasp_history_count + 1, len(self.grasp_success_history))

            # Record place outcomes (for stages 2+)
            if self.curriculum_stage >= 2:
                for i in range(n):
                    self.place_success_history[self.place_history_idx] = self.episode_place_success[envs_idx[i]].float()
                    self.place_history_idx = (self.place_history_idx + 1) % len(self.place_success_history)
                    self.place_history_count = min(self.place_history_count + 1, len(self.place_success_history))

            # Check for stage upgrade (min 500 episodes before checking)
            if self.grasp_history_count >= 500:
                grasp_rate = self.grasp_success_history[:self.grasp_history_count].mean().item()
                if self.curriculum_stage == 0 and grasp_rate > 0.50:
                    self.curriculum_stage = 1
                    self.grasp_history_count = 0  # reset to re-accumulate
                    print(f"[Curriculum] Stage 0 -> 1 (grasp_rate={grasp_rate:.2f}): narrow random spawn")
                elif self.curriculum_stage == 1 and grasp_rate > 0.70:
                    self.curriculum_stage = 2
                    self.grasp_history_count = 0
                    self.place_history_count = 0
                    print(f"[Curriculum] Stage 1 -> 2 (grasp_rate={grasp_rate:.2f}): close target placement")

            if self.curriculum_stage >= 2 and self.place_history_count >= 500:
                place_rate = self.place_success_history[:self.place_history_count].mean().item()
                if self.curriculum_stage == 2 and place_rate > 0.60:
                    self.curriculum_stage = 3
                    self.place_history_count = 0
                    print(f"[Curriculum] Stage 2 -> 3 (place_rate={place_rate:.2f}): medium target distance")
                elif self.curriculum_stage == 3 and place_rate > 0.50:
                    self.curriculum_stage = 4
                    print(f"[Curriculum] Stage 3 -> 4 (place_rate={place_rate:.2f}): full range target")

        self.episode_length_buf[envs_idx] = 0
        self.grasped[envs_idx] = False
        self.ever_grasped[envs_idx] = False
        self.episode_place_success[envs_idx] = False

        # Reset robot to default pose (arm + gripper open)
        default_qpos = torch.zeros(n, self.robot.n_dofs, device=self.device, dtype=gs.tc_float)
        for i, idx in enumerate(self.arm_dof_idx):
            default_qpos[:, idx] = self.default_arm_qpos[i]
        for idx in self.gripper_dof_idx:
            default_qpos[:, idx] = self.default_gripper_pos
        self.robot.set_qpos(default_qpos, envs_idx=envs_idx)  # zero_velocity=True by default

        # Randomize object position based on curriculum stage
        obj_half_z = self.obj_size[2] / 2
        if self.curriculum_stage == 0:
            # Fixed position: center of workspace
            obj_pos = torch.tensor(
                [0.30, 0.0, self.table_height + obj_half_z],
                device=self.device, dtype=gs.tc_float,
            ).unsqueeze(0).expand(n, 3).clone()
        else:
            rand_obj = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
            if self.curriculum_stage == 1:
                lower = self.obj_spawn_lower
                upper = self.obj_spawn_upper
            else:
                # Stage 2: full original range
                lower = torch.tensor([0.25, -0.10, 0.0], device=self.device, dtype=gs.tc_float)
                upper = torch.tensor([0.35, 0.10, 0.0], device=self.device, dtype=gs.tc_float)
            obj_pos = lower + rand_obj * (upper - lower)
            obj_pos[:, 2] = self.table_height + obj_half_z

        obj_quat = torch.tensor([1, 0, 0, 0], device=self.device, dtype=gs.tc_float).expand(n, 4)
        self.obj.set_pos(obj_pos, envs_idx=envs_idx)
        self.obj.set_quat(obj_quat, envs_idx=envs_idx)

        # Randomize target position based on curriculum stage
        if self.curriculum_stage <= 1:
            # Stage 0-1: Target near object (easy place, 10cm ahead)
            target_pos = obj_pos.clone()
            target_pos[:, 0] += 0.10
        elif self.curriculum_stage == 2:
            # Stage 2: Close target (5-15cm from object)
            rand_tgt = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
            target_pos = obj_pos.clone()
            target_pos[:, 0] += 0.05 + rand_tgt[:, 0] * 0.10
            target_pos[:, 1] += (rand_tgt[:, 1] - 0.5) * 0.06
        elif self.curriculum_stage == 3:
            # Stage 3: Medium target (10-25cm from object)
            rand_tgt = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
            target_pos = obj_pos.clone()
            target_pos[:, 0] += 0.10 + rand_tgt[:, 0] * 0.15
            target_pos[:, 1] += (rand_tgt[:, 1] - 0.5) * 0.10
        else:
            # Stage 4: Full range
            rand_tgt = torch.rand(n, 3, device=self.device, dtype=gs.tc_float)
            lower_t = torch.tensor([0.40, -0.10, 0.0], device=self.device, dtype=gs.tc_float)
            upper_t = torch.tensor([0.55, 0.10, 0.0], device=self.device, dtype=gs.tc_float)
            target_pos = lower_t + rand_tgt * (upper_t - lower_t)
            target_pos[:, 2] = self.table_height + obj_half_z

        self.target_pos[envs_idx] = target_pos
        self.target_marker.set_pos(target_pos, envs_idx=envs_idx)

        # Episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        self.extras["episode"]["curriculum_stage"] = self.curriculum_stage
        if self.grasp_history_count > 0:
            self.extras["episode"]["grasp_success_rate"] = (
                self.grasp_success_history[:self.grasp_history_count].mean().item()
            )
        if self.place_history_count > 0:
            self.extras["episode"]["place_success_rate"] = (
                self.place_success_history[:self.place_history_count].mean().item()
            )

        # Write CSV log
        if self.csv_log_path is not None:
            self._write_csv_log()

    def _write_csv_log(self):
        """Append current episode stats to CSV file."""
        ep = self.extras.get("episode", {})
        if not ep:
            return

        row = {
            "curriculum_stage": ep.get("curriculum_stage", self.curriculum_stage),
            "grasp_success_rate": ep.get("grasp_success_rate", 0.0),
            "place_success_rate": ep.get("place_success_rate", 0.0),
        }
        # Add all reward columns
        for key in self.reward_scales.keys():
            row["rew_" + key] = ep.get("rew_" + key, 0.0)

        fieldnames = list(row.keys())

        if self._csv_writer is None:
            write_header = not os.path.exists(self.csv_log_path)
            self._csv_file = open(self.csv_log_path, "a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            if write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1

        # Scale actions
        scaled = actions * self.action_scales

        # Cartesian control: delta EE position (3D) → joint positions via IK
        delta_pos = scaled[:, :3]
        gripper_cmd = actions[:, 3]  # raw, not scaled

        # Compute target IK pose (using link6 as IK link, orientation locked downward)
        current_pos = self.ik_link.get_pos()
        target_pos = current_pos + delta_pos

        # IK solve for arm joints
        arm_qpos = self.robot.inverse_kinematics(
            link=self.ik_link,
            pos=target_pos,
            quat=self.desired_down_quat_batch,  # fixed downward orientation
            dofs_idx_local=self.arm_dof_idx,
        )

        # Gripper control: cmd > 0 → open, cmd <= 0 → close
        gripper_target = torch.where(
            gripper_cmd > 0,
            torch.full_like(gripper_cmd, self.gripper_open_pos),
            torch.full_like(gripper_cmd, self.gripper_close_pos),
        )

        # Apply arm joint positions
        self.robot.control_dofs_position(arm_qpos[:, self.arm_dof_idx], self.arm_dof_idx)
        # Apply gripper position
        self.robot.control_dofs_position(gripper_target.unsqueeze(-1), self.gripper_dof_idx)

        self.scene.step()

        # Update grasp state
        self._update_grasp_state()

        # Update place success state (for curriculum tracking)
        obj_pos_now = self.obj.get_pos()
        dist_to_target = torch.norm(obj_pos_now - self.target_pos, dim=-1)
        self.episode_place_success = self.episode_place_success | (
            (dist_to_target < 0.04) & self.ever_grasped & (~self.grasped)
        )

        # Check termination (timeout only)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=gs.tc_float)
        time_out_idx = self.reset_buf.nonzero(as_tuple=True)[0]
        self.extras["time_outs"][time_out_idx] = 1.0

        # Compute reward
        reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # Save episode outcomes BEFORE reset clears them
        self.extras["episode_grasp_success"] = self.ever_grasped.clone()
        self.extras["episode_place_success"] = self.episode_place_success.clone()

        # Reset timed-out envs
        if len(time_out_idx) > 0:
            self.reset_idx(time_out_idx)

        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def _update_grasp_state(self):
        """Determine if object is grasped (lifted above table, gripper closed, EE near object)."""
        obj_pos = self.obj.get_pos()
        obj_z = obj_pos[:, 2]
        gripper_pos = self.robot.get_dofs_position(self.gripper_dof_idx).squeeze(-1)

        lifted = obj_z > (self.table_height + self.obj_size[2] / 2 + 0.02)  # 20mm lift
        gripper_closed = gripper_pos > 0.40  # 夹住 4cm 方块时 Grip≈0.51，阈值需低于此值

        # EE must be near the object
        ee_pos = self._finger_center_pos()
        ee_obj_dist = torch.norm(ee_pos - obj_pos, dim=-1)
        ee_near_obj = ee_obj_dist < 0.05

        self.grasped = lifted & gripper_closed & ee_near_obj
        self.ever_grasped = self.ever_grasped | self.grasped

    def _finger_center_pos(self) -> torch.Tensor:
        """Average position of left and right finger links."""
        left = self.left_finger_link.get_pos()
        right = self.right_finger_link.get_pos()
        return (left + right) / 2.0

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        ee_pos = self._finger_center_pos()                  # (B, 3)
        gripper_pos = self.robot.get_dofs_position(self.gripper_dof_idx)  # (B, 1)
        obj_pos = self.obj.get_pos()                        # (B, 3)
        obj_quat = self.obj.get_quat()                      # (B, 4)
        target_pos = self.target_pos                        # (B, 3)

        obj_to_ee = ee_pos - obj_pos                             # (B, 3)
        obj_to_target = target_pos - obj_pos                     # (B, 3)
        grasped_float = self.grasped.unsqueeze(-1).float()       # (B, 1)
        ever_grasped_float = self.ever_grasped.unsqueeze(-1).float()  # (B, 1)

        obs = torch.cat([
            ee_pos,              # 3
            gripper_pos,         # 1
            obj_pos,             # 3
            obj_quat,            # 4
            target_pos,          # 3
            obj_to_ee,           # 3
            obj_to_target,       # 3
            grasped_float,       # 1
            ever_grasped_float,  # 1
        ], dim=-1)               # total: 22

        self.extras["observations"] = {"critic": obs}
        return obs, self.extras

    def get_privileged_observations(self) -> None:
        return None

    # ------------ Reward functions ------------

    def _reward_reach(self) -> torch.Tensor:
        """Stage 1: Encourage EE to approach the object (before grasp).
        Uses 1/(1+k*d) instead of exp(-k*d) to maintain gradient at long range."""
        ee_pos = self._finger_center_pos()
        obj_pos = self.obj.get_pos()
        dist = torch.norm(ee_pos - obj_pos, dim=-1)
        return (1.0 / (1.0 + 5.0 * dist)) * (~self.ever_grasped).float()

    def _reward_align(self) -> torch.Tensor:
        """Reward for aligning EE height with object when close in XY."""
        ee_pos = self._finger_center_pos()
        obj_pos = self.obj.get_pos()
        xy_dist = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=-1)
        near_xy = torch.exp(-20.0 * xy_dist)
        height_diff = torch.abs(ee_pos[:, 2] - obj_pos[:, 2])
        height_align = torch.exp(-20.0 * height_diff)
        return near_xy * height_align * (~self.ever_grasped).float()

    def _reward_close_gripper(self) -> torch.Tensor:
        """Encourage closing gripper when near object (smooth proximity)."""
        ee_pos = self._finger_center_pos()
        obj_pos = self.obj.get_pos()
        dist = torch.norm(ee_pos - obj_pos, dim=-1)
        proximity = torch.clamp(1.0 - dist / 0.10, min=0.0)
        gripper_pos = self.robot.get_dofs_position(self.gripper_dof_idx).squeeze(-1)
        gripper_closing = (gripper_pos / self.gripper_close_pos).clamp(0, 1)
        return proximity * gripper_closing * (~self.ever_grasped).float()

    def _reward_lift(self) -> torch.Tensor:
        """Continuous reward for lifting the object off the table."""
        obj_z = self.obj.get_pos()[:, 2]
        rest_z = self.table_height + self.obj_size[2] / 2
        height_gain = (obj_z - rest_z).clamp(0, 0.10) / 0.10
        gripper_pos = self.robot.get_dofs_position(self.gripper_dof_idx).squeeze(-1)
        gripper_closed = (gripper_pos > 0.35).float()
        return height_gain * gripper_closed

    def _reward_grasp(self) -> torch.Tensor:
        """Bonus when object is grasped, disabled near target to incentivize release."""
        obj_pos = self.obj.get_pos()
        xy_dist = torch.norm(obj_pos[:, :2] - self.target_pos[:, :2], dim=-1)
        near_target = xy_dist < 0.05
        return self.grasped.float() * (~near_target).float()

    def _reward_place(self) -> torch.Tensor:
        """Encourage transporting grasped object to target XY, then lowering to table."""
        obj_pos = self.obj.get_pos()
        # Transport: XY proximity to target
        xy_dist = torch.norm(obj_pos[:, :2] - self.target_pos[:, :2], dim=-1)
        transport = (1.0 / (1.0 + 5.0 * xy_dist)) * self.grasped.float()
        # Lower: descend to table height when XY is close
        near_xy = (xy_dist < 0.05).float()
        rest_z = self.table_height + self.obj_size[2] / 2
        z_diff = torch.abs(obj_pos[:, 2] - rest_z)
        lower = torch.exp(-20.0 * z_diff) * near_xy * self.grasped.float()
        return transport + lower

    def _reward_release(self) -> torch.Tensor:
        """Reward opening gripper when object is at target position on table surface."""
        obj_pos = self.obj.get_pos()
        xy_dist = torch.norm(obj_pos[:, :2] - self.target_pos[:, :2], dim=-1)
        near_target = (xy_dist < 0.05).float()
        rest_z = self.table_height + self.obj_size[2] / 2
        at_table = torch.exp(-20.0 * torch.abs(obj_pos[:, 2] - rest_z))
        gripper_pos = self.robot.get_dofs_position(self.gripper_dof_idx).squeeze(-1)
        gripper_open = (1.0 - gripper_pos / self.gripper_close_pos).clamp(0, 1)
        return near_target * at_table * gripper_open * self.ever_grasped.float()

    def _reward_success(self) -> torch.Tensor:
        """Large bonus when object is placed at target, released, and stable."""
        obj_pos = self.obj.get_pos()
        dist = torch.norm(obj_pos - self.target_pos, dim=-1)
        near = (dist < 0.04).float()
        released = (~self.grasped).float() * self.ever_grasped.float()
        obj_vel = self.obj.get_vel()
        vel_mag = torch.norm(obj_vel, dim=-1)
        stability = torch.exp(-10.0 * vel_mag)
        return near * released * stability * 10.0

    def _reward_action_penalty(self) -> torch.Tensor:
        """Penalize large joint velocities for smooth motion."""
        joint_vel = self.robot.get_dofs_velocity(self.arm_dof_idx)
        return -torch.sum(joint_vel ** 2, dim=-1)

    def _reward_table_collision(self) -> torch.Tensor:
        """Penalize arm links (link3-link5) approaching the table surface.
        Linear penalty normalized by margin: 0 at threshold, 1 at table surface."""
        penalty = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        margin = 0.04  # start penalizing within 40mm of table surface
        for link in self.collision_monitor_links:
            link_z = link.get_pos()[:, 2]
            violation = torch.clamp((self.table_height + margin - link_z) / margin, min=0.0)
            penalty += violation
        return -penalty
