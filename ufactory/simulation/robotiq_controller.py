"""Rate-limited position/velocity servo with conditional PI anti-windup for Robotiq 2F."""

from __future__ import annotations

from typing import Any

import torch

from ufactory.grippers.robotiq import RobotiqLinkageGeometry
from ufactory.simulation.g2_controller import ControlOutput
from ufactory.simulation.g2_model import checked
from ufactory.simulation.robotiq_model import (
    ROBOTIQ_KI,
    ROBOTIQ_KP,
    ROBOTIQ_KV,
    RobotiqForceLookup,
    RobotiqModel,
)


def robotiq_drive_for_gap_m(gap_m: torch.Tensor, geometry: RobotiqLinkageGeometry, max_gap_m: float) -> torch.Tensor:
    g = geometry
    cosine = (gap_m.clamp(0.0, max_gap_m) / 2.0 - g.base_joint_y_m - g.pad_inner_y_m) / g.radius_m
    return (cosine.clamp(-1.0, 1.0).acos() - g.phase_rad).clamp(g.drive_lower_rad, g.drive_upper_rad)


def robotiq_pad_gap_m(drive_rad: torch.Tensor, geometry: RobotiqLinkageGeometry) -> torch.Tensor:
    g = geometry
    return 2.0 * (
        g.base_joint_y_m + g.finger_joint_y_m * drive_rad.cos() - g.finger_joint_z_m * drive_rad.sin() + g.pad_inner_y_m
    )


class RobotiqController:
    """Calibrated Robotiq 2F-85 / 2F-140 servo; every state and output is ``(num_envs,)``."""

    def __init__(
        self,
        model: RobotiqModel,
        *,
        num_envs: int = 1,
        device: Any = "cpu",
        dtype: torch.dtype = torch.float64,
        initial_position_mm: float | None = None,
        kp: float = ROBOTIQ_KP,
        kv: float = ROBOTIQ_KV,
        ki: float = ROBOTIQ_KI,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.model, self.num_envs = model, int(num_envs)
        self.device, self.dtype = torch.device(device), dtype
        self.kp = checked(kp, 1e-9, 1e6, "kp")
        self.kv = checked(kv, 0, 1e6, "kv")
        self.ki = checked(ki, 0, 1e6, "ki")
        self.open_gap_mm = float(model.spec.open_gap_mm)
        self.open_gap_m = float(model.spec.open_gap_m)
        initial_raw = self.open_gap_mm if initial_position_mm is None else initial_position_mm
        initial = checked(initial_raw, 0.0, self.open_gap_mm, "initial_position_mm")
        self.reference_gap_mm = self._full(initial)
        self.position_mm = self._full(initial)
        self.speed_percent = self._full(50.0)
        self.force_percent = self._full(50.0)
        self.last_nonzero_speed_percent = self._full(50.0)
        self.integral_torque_nm = self._full(0.0)
        self._pause_pending = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._reverse_pending = self._pause_pending.clone()
        self._pause_start_limit_nm = self._full(0.0)
        self._pause_start_integral_nm = self._full(0.0)
        self._paused_hold_sign = self._full(0.0)

    def _full(self, value: float) -> torch.Tensor:
        return torch.full((self.num_envs,), float(value), device=self.device, dtype=self.dtype)

    def _vector(self, value: Any, name: str, *, size: int | None = None, bounds: tuple | None = None) -> torch.Tensor:
        size = self.num_envs if size is None else size
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(size)
        if tensor.shape != (size,) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite with shape ({size},) or scalar")
        if bounds is not None and not bool(((tensor >= bounds[0]) & (tensor <= bounds[1])).all()):
            raise ValueError(f"{name} must be within {bounds}")
        return tensor

    def speed_mm_s(self, percent: torch.Tensor) -> torch.Tensor:
        lo = self.model.spec.speed_min_mm_s
        hi = self.model.spec.speed_max_mm_s
        scaled = lo + (percent / 100.0) * (hi - lo)
        return torch.where(percent > 0.0, scaled, 0.0)

    @property
    def requested_speed_mm_s(self) -> torch.Tensor:
        return self.speed_mm_s(self.speed_percent)

    def force_n(self) -> torch.Tensor:
        lo = self.model.spec.force_min_n
        hi = self.model.spec.force_max_n
        return lo + ((self.force_percent - 1.0) / 99.0) * (hi - lo)

    def force_lookup(self, env: int = 0) -> RobotiqForceLookup:
        """Calibration lookup of one environment, for telemetry."""
        speed = float(self.speed_mm_s(self.last_nonzero_speed_percent)[env])
        return self.model.curve.lookup(speed, float(self.force_percent[env]))

    @property
    def torque_limit_nm(self) -> torch.Tensor:
        return self.force_n() * self.model.reference_jacobian_m_rad

    def drive_for_gap_mm(self, gap: torch.Tensor) -> torch.Tensor:
        return robotiq_drive_for_gap_m(gap / 1000.0, self.model.linkage, self.open_gap_m)

    def gap_mm(self, q: torch.Tensor) -> torch.Tensor:
        return 1000.0 * robotiq_pad_gap_m(q, self.model.linkage)

    def command(
        self,
        position_mm: Any,
        speed_percent: Any = 50.0,
        force_percent: Any = 50.0,
        *,
        envs_idx: Any = None,
    ) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        size = self.position_mm[idx].numel()
        position = self._vector(position_mm, "position_mm", size=size, bounds=(0.0, self.open_gap_mm))
        speed = self._vector(speed_percent, "speed_percent", size=size, bounds=(0.0, 100.0))
        force = self._vector(force_percent, "force_percent", size=size, bounds=(1.0, 100.0))
        old_limit = self.torque_limit_nm[idx]
        old_direction = self.position_mm[idx] - self.reference_gap_mm[idx]
        new_direction = position - self.reference_gap_mm[idx]
        reverse = (speed > 0.0) & (
            (old_direction * new_direction < 0.0) | (new_direction * self.integral_torque_nm[idx] > 0.0)
        )
        pause = (speed == 0.0) & (self.speed_percent[idx] > 0.0)
        self._reverse_pending[idx] |= reverse
        self.integral_torque_nm[idx] = torch.where(reverse, 0.0, self.integral_torque_nm[idx])
        self._pause_pending[idx] |= pause
        self._pause_start_limit_nm[idx] = torch.where(pause, old_limit, self._pause_start_limit_nm[idx])
        self._pause_start_integral_nm[idx] = torch.where(
            pause, self.integral_torque_nm[idx], self._pause_start_integral_nm[idx]
        )
        self.last_nonzero_speed_percent[idx] = torch.where(speed > 0.0, speed, self.last_nonzero_speed_percent[idx])
        self._paused_hold_sign[idx] = torch.where(speed > 0.0, 0.0, self._paused_hold_sign[idx])
        self.position_mm[idx], self.speed_percent[idx], self.force_percent[idx] = position, speed, force
        limit = self.torque_limit_nm[idx]
        self.integral_torque_nm[idx] = self.integral_torque_nm[idx].clamp(-limit, limit)

    def reset(
        self,
        envs_idx: Any = None,
        *,
        position_mm: Any = None,
        target_mm: Any = None,
        prime: bool = False,
    ) -> None:
        """Reset the reference; optionally seed a saturated pre-held grasp."""
        idx = slice(None) if envs_idx is None else envs_idx
        size = self.position_mm[idx].numel()
        pos_val = self.open_gap_mm if position_mm is None else position_mm
        position = self._vector(pos_val, "position_mm", size=size, bounds=(0.0, self.open_gap_mm))
        target = (
            position
            if target_mm is None
            else self._vector(target_mm, "target_mm", size=size, bounds=(0.0, self.open_gap_mm))
        )
        self.reference_gap_mm[idx], self.position_mm[idx] = position, target
        self.integral_torque_nm[idx] = -torch.sign(target - position) * self.torque_limit_nm[idx] if prime else 0.0
        self._pause_pending[idx] = self._reverse_pending[idx] = False
        self._pause_start_limit_nm[idx] = self._pause_start_integral_nm[idx] = self._paused_hold_sign[idx] = 0.0

    def update(
        self,
        position_rad: Any,
        velocity_rad_s: Any,
        dt: float = 0.02,
        *,
        measured_gap_mm: Any = None,
    ) -> ControlOutput:
        q = self._vector(position_rad, "position_rad")
        velocity = self._vector(velocity_rad_s, "velocity_rad_s")
        dt = checked(dt, 1e-12, 1e6, "dt")
        measured = self.gap_mm(q) if measured_gap_mm is None else self._vector(measured_gap_mm, "measured_gap_mm")
        measured = measured.clamp(0.0, self.open_gap_mm)
        limit = self.torque_limit_nm
        old_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        self.reference_gap_mm = torch.where(self._reverse_pending, measured, self.reference_gap_mm)
        self.integral_torque_nm = torch.where(self._reverse_pending, 0.0, self.integral_torque_nm)
        self._reverse_pending.zero_()
        preload = self._pause_start_integral_nm + self.kp * (old_q_ref - q)
        self.integral_torque_nm = torch.where(self._pause_pending, preload, self.integral_torque_nm)
        hold_sign = torch.where(
            (velocity.abs() <= 0.02) & (preload.abs() >= 0.95 * self._pause_start_limit_nm),
            preload.sign(),
            0.0,
        )
        self._paused_hold_sign = torch.where(self._pause_pending, hold_sign, self._paused_hold_sign)
        paused_gap = torch.where(velocity.abs() > 0.02, measured, self.gap_mm(q).clamp(0.0, self.open_gap_mm))
        self.reference_gap_mm = torch.where(self._pause_pending, paused_gap, self.reference_gap_mm)
        self._pause_pending.zero_()
        previous_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        travel = self.speed_mm_s(self.speed_percent) * dt
        self.reference_gap_mm = self.reference_gap_mm + (self.position_mm - self.reference_gap_mm).clamp(
            -travel, travel
        )
        q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        v_ref = (q_ref - previous_q_ref) / dt
        error = q_ref - q
        integral = self.integral_torque_nm.clamp(-limit, limit)
        candidate = (integral + self.ki * error * dt).clamp(-limit, limit)
        candidate_total = self.kp * error + self.kv * (v_ref - velocity) + candidate
        integral = torch.where(
            (candidate_total.abs() <= limit) | (error * candidate_total <= 0.0),
            candidate,
            integral,
        )
        holding = (self.speed_percent == 0.0) & (self._paused_hold_sign != 0.0)
        self.integral_torque_nm = torch.where(holding, self._paused_hold_sign * limit, integral)
        target = torch.where(holding, q, q_ref) + self.integral_torque_nm / self.kp
        return ControlOutput(
            target,
            torch.where(holding, 0.0, v_ref),
            self.reference_gap_mm.clone(),
            self.integral_torque_nm.clone(),
            limit,
        )
