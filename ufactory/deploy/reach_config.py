"""Reach task deployment parameters aligned with simulation training."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ufactory.deploy.action_postprocess import effective_max_joint_delta_rad
from ufactory.robots.runtime import RobotRuntimeProfile, get_robot_runtime_profile

# Servo streaming defaults (rad / rad/s / rad/s^2).
DEFAULT_SERVO_SPEED_RAD_S = 0.5
DEFAULT_SERVO_MVACC_RAD_S2 = 5.0
# Mode-6 online joint planning defaults (rad/s / rad/s^2).
DEFAULT_ONLINE_JOINT_SPEED_RAD_S = 0.5
DEFAULT_ONLINE_JOINT_MVACC_RAD_S2 = 5.0
# Clip normalized policy actions before scaling (rsl-rl ActorCritic is unbounded at inference).
DEFAULT_ACTION_CLIP = 1.0
EXECUTOR_SERVO_J = "servo_j"
EXECUTOR_ONLINE_JOINT = "online-joint"
REACH_EXECUTORS = (EXECUTOR_SERVO_J, EXECUTOR_ONLINE_JOINT)


def normalize_reach_executor(executor: str) -> str:
    """Validate and normalize a reach deployment executor name."""
    value = str(executor).strip().lower()
    if value not in REACH_EXECUTORS:
        raise ValueError(f"Unknown reach executor {executor!r}; expected one of {REACH_EXECUTORS}")
    return value


@dataclass(frozen=True)
class ReachDeployConfig:
    """Runtime parameters shared by obs/action adapters and deploy loop."""

    dof: int
    num_obs: int
    num_actions: int
    action_scale: float
    ctrl_dt: float
    z_min_m: float
    action_clip: float
    max_joint_delta_rad: float
    servo_speed_rad_s: float
    servo_mvacc_rad_s2: float
    ee_link: str
    executor: str = EXECUTOR_SERVO_J
    online_joint_speed_rad_s: float = DEFAULT_ONLINE_JOINT_SPEED_RAD_S
    online_joint_mvacc_rad_s2: float = DEFAULT_ONLINE_JOINT_MVACC_RAD_S2

    def with_action_contract(self, contract: ReachDeployConfig) -> ReachDeployConfig:
        """Return this runtime config with action semantics copied from ``contract``.

        The executor and motion parameters stay runtime-owned. This lets a
        checkpoint trained with one delta limit run through a different xArm
        transport, such as mode-6 online joint planning, without changing the
        policy's action scale.
        """
        if self.dof != contract.dof:
            raise ValueError(f"Action contract dof mismatch: runtime={self.dof}, contract={contract.dof}")
        if self.num_actions != contract.num_actions:
            raise ValueError(
                f"Action contract action dim mismatch: runtime={self.num_actions}, contract={contract.num_actions}"
            )
        if self.num_obs != contract.num_obs:
            raise ValueError(f"Action contract obs dim mismatch: runtime={self.num_obs}, contract={contract.num_obs}")
        return replace(
            self,
            action_scale=contract.action_scale,
            ctrl_dt=contract.ctrl_dt,
            action_clip=contract.action_clip,
            max_joint_delta_rad=contract.max_joint_delta_rad,
            servo_speed_rad_s=contract.servo_speed_rad_s,
        )

    @classmethod
    def from_runtime(
        cls,
        runtime: RobotRuntimeProfile,
        *,
        z_min_mm: float = 0.0,
        action_clip: float = DEFAULT_ACTION_CLIP,
        servo_speed_rad_s: float = DEFAULT_SERVO_SPEED_RAD_S,
        servo_mvacc_rad_s2: float = DEFAULT_SERVO_MVACC_RAD_S2,
        executor: str = EXECUTOR_SERVO_J,
        online_joint_speed_rad_s: float = DEFAULT_ONLINE_JOINT_SPEED_RAD_S,
        online_joint_mvacc_rad_s2: float = DEFAULT_ONLINE_JOINT_MVACC_RAD_S2,
    ) -> ReachDeployConfig:
        executor = normalize_reach_executor(executor)
        defaults = runtime.task.reach_env_defaults
        action_scale = float(defaults["action_scale"])
        ctrl_dt = float(defaults["ctrl_dt"])
        servo_limit_speed = servo_speed_rad_s if executor == EXECUTOR_SERVO_J else None
        max_joint_delta_rad = effective_max_joint_delta_rad(
            action_scale=action_scale,
            action_clip=action_clip,
            ctrl_dt=ctrl_dt,
            servo_speed_rad_s=servo_limit_speed,
            max_joint_delta_rad=defaults.get("max_joint_delta_rad"),
        )
        return cls(
            dof=runtime.model.dof,
            num_obs=int(defaults["num_obs"]),
            num_actions=int(defaults["num_actions"]),
            action_scale=action_scale,
            ctrl_dt=ctrl_dt,
            z_min_m=float(z_min_mm) / 1000.0,
            action_clip=float(action_clip),
            max_joint_delta_rad=max_joint_delta_rad,
            servo_speed_rad_s=float(servo_speed_rad_s),
            servo_mvacc_rad_s2=float(servo_mvacc_rad_s2),
            ee_link=runtime.arm.ee_link,
            executor=executor,
            online_joint_speed_rad_s=float(online_joint_speed_rad_s),
            online_joint_mvacc_rad_s2=float(online_joint_mvacc_rad_s2),
        )

    @classmethod
    def for_robot(
        cls,
        robot_key: str = "xarm6",
        **kwargs,
    ) -> ReachDeployConfig:
        return cls.from_runtime(get_robot_runtime_profile(robot_key), **kwargs)
