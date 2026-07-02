"""Real-robot session for reach deploy (servo joint streaming)."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from ufactory.deploy.action_postprocess import ReachActionCommand
from ufactory.deploy.action_adapter import apply_reach_joint_delta
from ufactory.deploy.obs_adapter import build_reach_obs
from ufactory.deploy.reach_config import EXECUTOR_ONLINE_JOINT, EXECUTOR_SERVO_J, ReachDeployConfig
from ufactory.deploy.sdk_units import sdk_position_m
from ufactory.deploy.safety import SafetyGuard
from ufactory.real_robot_session import RealRobotSession
from ufactory.robot_params import RobotRuntimeProfile, get_robot_runtime_profile
from ufactory.xarm_control import (
    MODE_JOINT_ONLINE_PLANNING,
    MODE_POSITION,
    MODE_SERVO,
    prepare_arm_for_motion,
    prime_online_joint_planning,
    prime_servo_angle_j,
    wait_for_servo_motion_ready,
)

DEFAULT_GRAVITY_DIRECTION = [0.0, 0.0, -1.0]


class ReachDeploySession(RealRobotSession):
    """xArm session configured for reach-policy deployment."""

    def __init__(
        self,
        ip: str,
        *,
        robot_key: str = "xarm6",
        runtime: RobotRuntimeProfile | None = None,
        z_min_mm: float = 0.0,
        executor: str = EXECUTOR_SERVO_J,
    ) -> None:
        runtime = runtime or get_robot_runtime_profile(robot_key)
        config = ReachDeployConfig.from_runtime(runtime, z_min_mm=z_min_mm, executor=executor)
        motion_mode = MODE_JOINT_ONLINE_PLANNING if config.executor == EXECUTOR_ONLINE_JOINT else MODE_SERVO
        super().__init__(
            ip,
            dof=runtime.model.dof,
            home_qpos=runtime.arm.default_qpos,
            motion_mode=motion_mode,
        )
        self.runtime = runtime
        self.config = config
        self.safety = SafetyGuard.from_runtime(runtime, self.config)
        self._target_pos_m = np.zeros(3, dtype=np.float64)
        self.last_action_command: ReachActionCommand | None = None
        self.action_command_log: list[ReachActionCommand] = []
        self.deploy_configured = False

    def configure_for_deploy(self) -> None:
        """Servo mode after dynamics-style setup in position mode."""
        self.deploy_configured = False
        arm = self.arm
        # TCP/gravity in position mode (same proven path as dynamics/replay).
        self._motion_mode = MODE_POSITION
        self.ensure_ready(MODE_POSITION)
        arm.set_gravity_direction(DEFAULT_GRAVITY_DIRECTION)
        code = arm.set_tcp_load(0, [0, 0, 0])
        if code != 0:
            print(f"[WARN] set_tcp_load(0) returned code={code}")
        arm.set_report_tau_or_i(0)

        if self.config.executor == EXECUTOR_ONLINE_JOINT:
            prepare_arm_for_motion(arm, mode=MODE_JOINT_ONLINE_PLANNING)
            self._motion_mode = MODE_JOINT_ONLINE_PLANNING
            time.sleep(1.0)
            q, _, _ = self.get_joint_states()
            prime_online_joint_planning(
                arm,
                q.tolist(),
                speed=self.config.online_joint_speed_rad_s,
                mvacc=self.config.online_joint_mvacc_rad_s2,
            )
            self.deploy_configured = True
            return

        prepare_arm_for_motion(arm, mode=MODE_SERVO)
        self._motion_mode = MODE_SERVO
        time.sleep(0.15)
        wait_for_servo_motion_ready(arm)
        q, _, _ = self.get_joint_states()
        prime_servo_angle_j(
            arm,
            q.tolist(),
            speed=self.config.servo_speed_rad_s,
            mvacc=self.config.servo_mvacc_rad_s2,
        )
        self.deploy_configured = True

    def set_target_position(self, target_pos_m: Sequence[float]) -> None:
        self._target_pos_m = np.asarray(target_pos_m, dtype=np.float64).reshape(3)

    @property
    def target_pos_m(self) -> np.ndarray:
        return self._target_pos_m.copy()

    def use_action_contract(self, contract: ReachDeployConfig) -> None:
        """Apply checkpoint action semantics while preserving the runtime executor."""
        self.config = self.config.with_action_contract(contract)
        self.safety = SafetyGuard.from_runtime(self.runtime, self.config)

    def get_ee_pos_m(self) -> np.ndarray:
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed with code {code}")
        ee = sdk_position_m(pose)
        self.safety.check_ee_position(ee)
        return ee

    def read_reach_obs(self) -> np.ndarray:
        q, qvel, _ = self.get_joint_states()
        ee = self.get_ee_pos_m()
        return build_reach_obs(q, qvel, ee, self._target_pos_m)

    def step_servo(self, action: np.ndarray) -> np.ndarray:
        self.safety.check_arm_ready(self.arm)
        q, _, _ = self.get_joint_states()
        q_cmd = apply_reach_joint_delta(
            self.arm,
            q,
            action,
            config=self.config,
            safety=self.safety,
            on_command=self._record_action_command,
        )
        return q_cmd

    def _record_action_command(self, command: ReachActionCommand) -> None:
        self.last_action_command = command
        self.action_command_log.append(command)

    def run_control_loop(
        self,
        step_fn,
        *,
        steps: int,
        ctrl_dt: float | None = None,
    ) -> None:
        """Run ``step_fn(session, step_idx) -> action`` at fixed rate."""
        dt = self.config.ctrl_dt if ctrl_dt is None else float(ctrl_dt)
        for step in range(steps):
            t0 = time.monotonic()
            action = step_fn(self, step)
            if action is not None:
                self.step_servo(np.asarray(action, dtype=np.float64))
            elapsed = time.monotonic() - t0
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
