"""Torch implementation of the calibrated G2 servo with independent state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ufactory.grippers.g2_torch import drive_for_gap_m, pad_gap_m
from ufactory.simulation.g2_model import G2Model, G2_KP, G2_KV, G2_KI, checked


@dataclass(frozen=True)
class BatchControlOutput:
    position_rad: torch.Tensor
    velocity_rad_s: torch.Tensor
    reference_gap_mm: torch.Tensor
    integral_torque_nm: torch.Tensor
    torque_limit_nm: torch.Tensor


class G2BatchController:
    """Same control law as G2Controller, vectorized without environment loops."""

    def __init__(
        self,
        model: G2Model,
        num_envs: int,
        *,
        device: Any = "cpu",
        dtype: torch.dtype = torch.float32,
        initial_position_mm: float = 84,
        kp: float = G2_KP,
        kv: float = G2_KV,
        ki: float = G2_KI,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.model, self.num_envs = model, int(num_envs)
        self.device, self.dtype = torch.device(device), dtype
        self.kp = checked(kp, 1e-9, 1e6, "kp")
        self.kv = checked(kv, 0, 1e6, "kv")
        self.ki = checked(ki, 0, 1e6, "ki")
        initial = checked(initial_position_mm, 0, 84, "initial_position_mm")
        self.reference_gap_mm = self._full(initial)
        self.position_mm = self._full(initial)
        self.speed_percent = self._full(50)
        self.force_percent = self._full(50)
        self.last_nonzero_speed_percent = self._full(50)
        self.integral_torque_nm = self._full(0)
        self._pause_pending = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._reverse_pending = self._pause_pending.clone()
        self._pause_start_limit_nm = self._full(0)
        self._pause_start_integral_nm = self._full(0)
        self._paused_hold_sign = self._full(0)
        self._speeds = torch.tensor(model.curve.speeds, device=self.device, dtype=dtype)
        # Use the union of knots; interpolation at every original knot retains
        # all sharp changes and plateaus even if speed curves have unequal knots.
        currents = sorted({c for points in model.curve.points.values() for c in points})
        self._currents = torch.tensor(currents, device=self.device, dtype=dtype)
        self._forces = torch.tensor(
            [[model.curve._at_speed(rpm, current) for current in currents] for rpm in model.curve.speeds],
            device=self.device,
            dtype=dtype,
        )

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

    @staticmethod
    def speed_mm_s(percent: torch.Tensor) -> torch.Tensor:
        return torch.where(percent > 0, (2.25 * percent).clamp_min(15), 0.0)

    def force_n(self) -> torch.Tensor:
        speed = self.speed_mm_s(self.last_nonzero_speed_percent)
        rpm = torch.trunc((speed * 60 / 9.88235 + 140) / 0.4).clamp_max(3500)
        hi = torch.searchsorted(self._speeds, rpm.contiguous()).clamp(1, len(self._speeds) - 1)
        lo = hi - 1
        ratio = (rpm - self._speeds[lo]) / (self._speeds[hi] - self._speeds[lo])
        chi = torch.searchsorted(self._currents, self.force_percent.contiguous()).clamp(1, len(self._currents) - 1)
        clo = chi - 1
        cr = (self.force_percent - self._currents[clo]) / (self._currents[chi] - self._currents[clo])
        a = torch.lerp(self._forces[lo, clo], self._forces[lo, chi], cr)
        b = torch.lerp(self._forces[hi, clo], self._forces[hi, chi], cr)
        return torch.lerp(a, b, ratio)

    @property
    def torque_limit_nm(self) -> torch.Tensor:
        return self.force_n() * self.model.reference_jacobian_m_rad

    def drive_for_gap_mm(self, gap: torch.Tensor) -> torch.Tensor:
        return drive_for_gap_m(gap / 1000, self.model.linkage)

    def gap_mm(self, q: torch.Tensor) -> torch.Tensor:
        return 1000 * pad_gap_m(q, self.model.linkage)

    def command(
        self, position_mm: Any, speed_percent: Any = 50, force_percent: Any = 50, *, envs_idx: Any = None
    ) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        size = self.position_mm[idx].numel()
        position = self._vector(position_mm, "position_mm", size=size, bounds=(0, 84))
        speed = self._vector(speed_percent, "speed_percent", size=size, bounds=(0, 100))
        force = self._vector(force_percent, "force_percent", size=size, bounds=(1, 100))
        old_limit = self.torque_limit_nm[idx]
        old_direction = self.position_mm[idx] - self.reference_gap_mm[idx]
        new_direction = position - self.reference_gap_mm[idx]
        reverse = (speed > 0) & (
            (old_direction * new_direction < 0) | (new_direction * self.integral_torque_nm[idx] > 0)
        )
        pause = (speed == 0) & (self.speed_percent[idx] > 0)
        self._reverse_pending[idx] |= reverse
        self.integral_torque_nm[idx] = torch.where(reverse, 0.0, self.integral_torque_nm[idx])
        self._pause_pending[idx] |= pause
        self._pause_start_limit_nm[idx] = torch.where(pause, old_limit, self._pause_start_limit_nm[idx])
        self._pause_start_integral_nm[idx] = torch.where(
            pause, self.integral_torque_nm[idx], self._pause_start_integral_nm[idx]
        )
        self.last_nonzero_speed_percent[idx] = torch.where(speed > 0, speed, self.last_nonzero_speed_percent[idx])
        self._paused_hold_sign[idx] = torch.where(speed > 0, 0.0, self._paused_hold_sign[idx])
        self.position_mm[idx], self.speed_percent[idx], self.force_percent[idx] = position, speed, force
        limit = self.torque_limit_nm[idx]
        self.integral_torque_nm[idx] = self.integral_torque_nm[idx].clamp(-limit, limit)

    def reset(self, envs_idx: Any = None, *, position_mm: Any = 84, target_mm: Any = None, prime: bool = False) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        size = self.position_mm[idx].numel()
        position = self._vector(position_mm, "position_mm", size=size, bounds=(0, 84))
        target = position if target_mm is None else self._vector(target_mm, "target_mm", size=size, bounds=(0, 84))
        self.reference_gap_mm[idx], self.position_mm[idx] = position, target
        self.integral_torque_nm[idx] = -torch.sign(target - position) * self.torque_limit_nm[idx] if prime else 0
        self._pause_pending[idx] = self._reverse_pending[idx] = False
        self._pause_start_limit_nm[idx] = self._pause_start_integral_nm[idx] = self._paused_hold_sign[idx] = 0

    def update(
        self, position_rad: Any, velocity_rad_s: Any, dt: float = 0.02, *, measured_gap_mm: Any = None
    ) -> BatchControlOutput:
        q = self._vector(position_rad, "position_rad")
        velocity = self._vector(velocity_rad_s, "velocity_rad_s")
        dt = checked(dt, 1e-12, 1e6, "dt")
        measured = self.gap_mm(q) if measured_gap_mm is None else self._vector(measured_gap_mm, "measured_gap_mm")
        measured = measured.clamp(0, 84)
        limit = self.torque_limit_nm
        old_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        self.reference_gap_mm = torch.where(self._reverse_pending, measured, self.reference_gap_mm)
        self.integral_torque_nm = torch.where(self._reverse_pending, 0.0, self.integral_torque_nm)
        self._reverse_pending.zero_()
        preload = self._pause_start_integral_nm + self.kp * (old_q_ref - q)
        self.integral_torque_nm = torch.where(self._pause_pending, preload, self.integral_torque_nm)
        hold_sign = torch.where(
            (velocity.abs() <= 0.02) & (preload.abs() >= 0.95 * self._pause_start_limit_nm), preload.sign(), 0
        )
        self._paused_hold_sign = torch.where(self._pause_pending, hold_sign, self._paused_hold_sign)
        paused_gap = torch.where(velocity.abs() > 0.02, measured, self.gap_mm(q).clamp(0, 84))
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
        integral = torch.where((candidate_total.abs() <= limit) | (error * candidate_total <= 0), candidate, integral)
        holding = (self.speed_percent == 0) & (self._paused_hold_sign != 0)
        self.integral_torque_nm = torch.where(holding, self._paused_hold_sign * limit, integral)
        target = torch.where(holding, q, q_ref) + self.integral_torque_nm / self.kp
        return BatchControlOutput(
            target,
            torch.where(holding, 0.0, v_ref),
            self.reference_gap_mm.clone(),
            self.integral_torque_nm.clone(),
            limit,
        )
