"""Real xArm connection helpers for static torque verification.

Unit policy (real-robot motion):
  - joint angles q: rad
  - joint speed: rad/s
  - joint acceleration (mvacc, if used): rad/s^2
Aligns with XArmAPI(is_radian=True) and set_servo_angle(..., is_radian=True).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ufactory.robot_params import DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
from ufactory.xarm_control import (
    MODE_POSITION,
    format_arm_status,
    prepare_arm_for_motion,
)

DEFAULT_GRAVITY_DIRECTION = [0.0, 0.0, -1.0]
HOME_Q = np.zeros(6, dtype=np.float64)
MOVE_STRATEGY_AXIS_SEQUENTIAL = "axis-sequential"
MOVE_STRATEGY_DIRECT = "direct"
MOVE_STRATEGIES = (MOVE_STRATEGY_AXIS_SEQUENTIAL, MOVE_STRATEGY_DIRECT)

VEL_TOL_RAD_S = 0.01
SETTLE_TIMEOUT_S = 5.0
SETTLE_POLL_S = 0.1


@dataclass
class RealRobotSample:
    q: np.ndarray
    qvel: np.ndarray
    tau: np.ndarray
    settled: bool
    tau_median: np.ndarray | None = None
    tau_std: np.ndarray | None = None
    tau_min: np.ndarray | None = None
    tau_max: np.ndarray | None = None
    tau_direct: np.ndarray | None = None
    tau_direct_std: np.ndarray | None = None
    n_samples: int = 1
    duration_s: float = 0.0
    temperature: np.ndarray | None = None
    runtime_s: float | None = None


class RobotMotionError(RuntimeError):
    """Raised when a real robot motion command fails."""

    def __init__(
        self,
        message: str,
        *,
        code: int,
        waypoint_index: int,
        waypoint_count: int,
        target_q: Sequence[float],
        waypoint_q: Sequence[float],
        actual_q: Sequence[float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.waypoint_index = waypoint_index
        self.waypoint_count = waypoint_count
        self.target_q = np.asarray(target_q, dtype=np.float64)
        self.waypoint_q = np.asarray(waypoint_q, dtype=np.float64)
        self.actual_q = None if actual_q is None else np.asarray(actual_q, dtype=np.float64)


def build_axis_sequential_waypoints(
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    atol: float = 1e-9,
) -> list[np.ndarray]:
    """Build waypoints that move J1..J6 one axis at a time."""
    start = np.asarray(start_q, dtype=np.float64).reshape(-1)
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if start.shape != target.shape:
        raise ValueError(f"start_q shape {start.shape} does not match target_q shape {target.shape}")

    current = start.copy()
    waypoints: list[np.ndarray] = []
    for idx in range(target.size):
        if abs(float(current[idx] - target[idx])) <= atol:
            continue
        current = current.copy()
        current[idx] = target[idx]
        waypoints.append(current.copy())
    return waypoints


def build_motion_waypoints(
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    strategy: str = MOVE_STRATEGY_DIRECT,
) -> list[np.ndarray]:
    """Build real-robot motion waypoints for a named movement strategy."""
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if strategy == MOVE_STRATEGY_AXIS_SEQUENTIAL:
        return build_axis_sequential_waypoints(start_q, target)
    if strategy == MOVE_STRATEGY_DIRECT:
        return [target.copy()]
    raise ValueError(f"Unknown move strategy: {strategy}")


class RealRobotSession:
    """Non-simulation xArm session for slow position holds and torque readback."""

    def __init__(
        self,
        ip: str,
        *,
        dof: int = 6,
        home_qpos: Sequence[float] | None = None,
        is_radian: bool = True,
        motion_mode: int = MODE_POSITION,
    ):
        from xarm.wrapper import XArmAPI

        self.ip = ip
        self.dof = int(dof)
        self.home_qpos = np.asarray(
            home_qpos if home_qpos is not None else np.zeros(self.dof, dtype=np.float64),
            dtype=np.float64,
        )
        self._motion_mode = motion_mode
        self.arm = XArmAPI(ip, is_radian=is_radian)
        time.sleep(0.5)
        if not self.arm.connected:
            raise ConnectionError(f"Failed to connect to xArm at {ip}")

    def ensure_ready(self, mode: int | None = None) -> None:
        """Prepare arm for motion (SDK: motion_enable -> set_mode -> set_state(0))."""
        prepare_arm_for_motion(self.arm, mode=mode if mode is not None else self._motion_mode)

    def configure_for_dynamics(self) -> None:
        """Position mode, bare flange, torque reporting in Nm."""
        self._motion_mode = MODE_POSITION
        self.ensure_ready(MODE_POSITION)
        arm = self.arm
        arm.set_gravity_direction(DEFAULT_GRAVITY_DIRECTION)
        code = arm.set_tcp_load(0, [0, 0, 0])
        if code != 0:
            print(f"[WARN] set_tcp_load(0) returned code={code} (tcp_load may already be zero)")
        arm.set_report_tau_or_i(0)

    def configure_for_simulation_collision_check(self) -> None:
        """Dynamics setup plus xArm simulation mode and self-collision detection."""
        self.configure_for_dynamics()
        self.arm.set_simulation_robot(on_off=True)
        self.arm.set_self_collision_detection(True)

    def recover_after_motion_error(self, *, retries: int = 5) -> bool:
        """Clear stop/collision state so the next pose can be checked."""
        for _ in range(retries):
            try:
                self.arm.clean_warn()
                time.sleep(0.1)
                if self.arm.error_code != 0:
                    self.arm.clean_error()
                    time.sleep(0.3)
                self.ensure_ready(MODE_POSITION)
                if self.arm.error_code == 0 and self.arm.state != 4:
                    return True
            except Exception:
                pass
            time.sleep(0.2)
        return False

    def print_config(self) -> None:
        arm = self.arm
        print(f"SDK firmware   : {arm.version}")
        code, reported_state = arm.get_state()
        gs = f"{reported_state}" if code == 0 else f"err({code})"
        print(f"robot state    : {arm.state} (report; 4=stop)  get_state()={gs}")
        print(f"robot mode     : {arm.mode} (report; 0=position)")
        print(f"error/warn     : {arm.error_code} / {arm.warn_code}")
        print(f"tcp_offset     : {list(arm.tcp_offset)}")
        print(f"tcp_load       : {arm.tcp_load}")
        print(f"gravity_dir    : {list(arm.gravity_direction)}")
        code, tau_or_i = arm.get_report_tau_or_i()
        if code == 0:
            print(f"report_tau_or_i: {tau_or_i} (0=torque Nm)")

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        if code != 0:
            raise RuntimeError(f"get_joint_states failed with code {code}")
        pos, vel, effort = states
        return (
            np.asarray(pos[: self.dof], dtype=np.float64),
            np.asarray(vel[: self.dof], dtype=np.float64),
            np.asarray(effort[: self.dof], dtype=np.float64),
        )

    def get_joints_torque(self) -> np.ndarray | None:
        """Read direct joint torque API when available.

        Some firmware/SDK combinations expose both ``get_joint_states(...,
        effort)`` and ``get_joints_torque()``.  The validation report records
        both so bias in one channel is visible instead of hidden.
        """
        getter = getattr(self.arm, "get_joints_torque", None)
        if getter is None:
            return None
        code, tau = getter()
        if code != 0:
            return None
        return np.asarray(tau[: self.dof], dtype=np.float64)

    def wait_until_settled(
        self,
        *,
        vel_tol: float = VEL_TOL_RAD_S,
        timeout_s: float = SETTLE_TIMEOUT_S,
        poll_s: float = SETTLE_POLL_S,
    ) -> RealRobotSample:
        deadline = time.monotonic() + timeout_s
        last_sample: RealRobotSample | None = None
        while time.monotonic() < deadline:
            q, qvel, tau = self.get_joint_states()
            last_sample = RealRobotSample(q=q, qvel=qvel, tau=tau, settled=False)
            if float(np.abs(qvel).max()) <= vel_tol:
                last_sample.settled = True
                return last_sample
            time.sleep(poll_s)
        if last_sample is None:
            raise RuntimeError("No joint state samples received")
        return last_sample

    def collect_hold_samples(
        self,
        *,
        duration_s: float = 3.0,
        poll_s: float = SETTLE_POLL_S,
        vel_tol: float = VEL_TOL_RAD_S,
    ) -> RealRobotSample:
        """Collect multiple steady-state samples and return robust statistics."""
        deadline = time.monotonic() + max(0.0, duration_s)
        qs: list[np.ndarray] = []
        qvels: list[np.ndarray] = []
        efforts: list[np.ndarray] = []
        direct_torques: list[np.ndarray] = []

        while True:
            q, qvel, tau = self.get_joint_states()
            qs.append(q)
            qvels.append(qvel)
            efforts.append(tau)
            tau_direct = self.get_joints_torque()
            if tau_direct is not None:
                direct_torques.append(tau_direct)

            if time.monotonic() >= deadline:
                break
            time.sleep(poll_s)

        q_arr = np.vstack(qs)
        qvel_arr = np.vstack(qvels)
        tau_arr = np.vstack(efforts)
        settled = bool(float(np.abs(qvel_arr).max()) <= vel_tol)
        tau_direct_mean = tau_direct_std = None
        if direct_torques:
            direct_arr = np.vstack(direct_torques)
            tau_direct_mean = direct_arr.mean(axis=0)
            tau_direct_std = direct_arr.std(axis=0)

        return RealRobotSample(
            q=q_arr.mean(axis=0),
            qvel=qvel_arr.mean(axis=0),
            tau=tau_arr.mean(axis=0),
            settled=settled,
            tau_median=np.median(tau_arr, axis=0),
            tau_std=tau_arr.std(axis=0),
            tau_min=tau_arr.min(axis=0),
            tau_max=tau_arr.max(axis=0),
            tau_direct=tau_direct_mean,
            tau_direct_std=tau_direct_std,
            n_samples=len(efforts),
            duration_s=duration_s,
            temperature=None,
            runtime_s=None,
        )

    def move_to(
        self,
        q: Sequence[float],
        *,
        speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        wait: bool = True,
        move_strategy: str = MOVE_STRATEGY_DIRECT,
    ) -> int:
        target = np.asarray(q, dtype=np.float64).reshape(-1)
        q_start, _, _ = self.get_joint_states()
        waypoints = build_motion_waypoints(q_start, target, strategy=move_strategy)
        if not waypoints:
            return 0

        for i, waypoint in enumerate(waypoints, start=1):
            self.ensure_ready(MODE_POSITION)
            print(
                f"    waypoint {i}/{len(waypoints)} [{move_strategy}] "
                f"speed={float(speed_rad_s):.4f} rad/s "
                f"q=[{', '.join(f'{v:.4f}' for v in waypoint)}]"
            )
            code = self.arm.set_servo_angle(
                angle=waypoint.tolist(),
                speed=float(speed_rad_s),
                mvtime=0,
                wait=wait,
                is_radian=True,
                radius=None,
            )
            if code != 0:
                try:
                    q_now, _, _ = self.get_joint_states()
                except Exception:
                    q_now = None
                actual = "unknown" if q_now is None else list(np.asarray(q_now, dtype=np.float64))
                raise RobotMotionError(
                    f"set_servo_angle failed code={code} waypoint {i}/{len(waypoints)} "
                    f"move_strategy={move_strategy} target_q={target.tolist()} "
                    f"waypoint_q={waypoint.tolist()} actual_q={actual} {format_arm_status(self.arm)}",
                    code=code,
                    waypoint_index=i,
                    waypoint_count=len(waypoints),
                    target_q=target,
                    waypoint_q=waypoint,
                    actual_q=q_now,
                )
        return 0

    def sample_at_hold(
        self,
        q: Sequence[float],
        *,
        speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy: str = MOVE_STRATEGY_DIRECT,
        post_move_wait_s: float = 1.5,
        sample_duration_s: float = 3.0,
        sample_poll_s: float = SETTLE_POLL_S,
    ) -> RealRobotSample:
        self.move_to(q, speed_rad_s=speed_rad_s, wait=True, move_strategy=move_strategy)
        time.sleep(post_move_wait_s)
        settled_sample = self.wait_until_settled()
        stats = self.collect_hold_samples(duration_s=sample_duration_s, poll_s=sample_poll_s)
        stats.settled = bool(settled_sample.settled and stats.settled)
        return stats

    def return_home(
        self,
        *,
        speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy: str = MOVE_STRATEGY_DIRECT,
    ) -> None:
        try:
            self.move_to(self.home_qpos, speed_rad_s=speed_rad_s, wait=True, move_strategy=move_strategy)
            time.sleep(1.0)
        except RuntimeError as exc:
            print(f"[WARN] return_home failed: {exc}")

    def disconnect(self) -> None:
        self.arm.disconnect()


def sdk_fk_ee_z_mm(arm, q: Sequence[float]) -> float:
    """EE z in mm from SDK FK (simulation mode not required)."""
    code, pose = arm.get_forward_kinematics(
        angles=list(q),
        input_is_radian=True,
        return_is_radian=True,
    )
    if code != 0:
        raise RuntimeError(f"SDK FK failed with code {code}")
    return float(pose[2])
