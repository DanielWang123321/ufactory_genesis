"""Rate-limited position/velocity servo with conditional PI anti-windup.

Only joint position/velocity and optional pad position are inputs.
Contact force is not fed back.
The output position bias lets Genesis evaluate the PD term every substep.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ufactory.simulation.g2_model import G2Model, ForceLookup, checked, sdk_speed_rpm, speed_mm_s


@dataclass(frozen=True)
class ControlOutput:
    position_rad: float
    velocity_rad_s: float
    reference_gap_mm: float
    integral_torque_nm: float
    torque_limit_nm: float
    force_lookup: ForceLookup


class G2Controller:
    def __init__(
        self, model: G2Model, initial_position_mm: float = 84, *, kp: float = 40, kv: float = 5, ki: float = 40
    ):
        self.model = model
        self.kp = checked(kp, 1e-9, 1e6, "kp")
        self.kv = checked(kv, 0, 1e6, "kv")
        self.ki = checked(ki, 0, 1e6, "ki")
        self.reference_gap_mm = checked(initial_position_mm, 0, 84, "initial_position_mm")
        self.position_mm = self.reference_gap_mm
        self.speed_percent = 50.0
        self.force_percent = 50.0
        self.last_nonzero_speed_percent = 50.0
        self.integral_torque_nm = 0.0
        self._pause_pending = False
        self._reverse_pending = False
        self._pause_start_limit_nm = 0.0
        self._pause_start_integral_nm = 0.0
        self._paused_hold_sign = 0.0

    @property
    def requested_speed_mm_s(self) -> float:
        return speed_mm_s(self.speed_percent)

    @property
    def requested_speed_rpm(self) -> int:
        return sdk_speed_rpm(self.requested_speed_mm_s)

    @property
    def force_lookup(self) -> ForceLookup:
        rpm = sdk_speed_rpm(speed_mm_s(self.last_nonzero_speed_percent))
        return self.model.curve.lookup(rpm, self.force_percent)

    def command(self, position_mm: float, speed_percent: float = 50, force_percent: float = 50) -> None:
        # Validate the whole command before changing any controller state.
        position = checked(position_mm, 0, 84, "position_mm")
        speed = checked(speed_percent, 0, 100, "speed_percent")
        force = checked(force_percent, 1, 100, "force_percent")
        old_limit = self.model.torque_limit_nm(self.force_lookup)
        old_direction = self.position_mm - self.reference_gap_mm
        new_direction = position - self.reference_gap_mm
        if speed > 0 and (old_direction * new_direction < 0 or new_direction * self.integral_torque_nm > 0):
            self._reverse_pending = True
            self.integral_torque_nm = 0.0
        if speed == 0 and self.speed_percent > 0:
            self._pause_pending = True
            self._pause_start_limit_nm = old_limit
            self._pause_start_integral_nm = self.integral_torque_nm
        if speed > 0:
            self.last_nonzero_speed_percent = speed
            self._paused_hold_sign = 0.0
        self.position_mm, self.speed_percent, self.force_percent = position, speed, force
        limit = self.model.torque_limit_nm(self.force_lookup)
        self.integral_torque_nm = max(-limit, min(limit, self.integral_torque_nm))

    def update(
        self,
        position_rad: float,
        velocity_rad_s: float,
        dt: float = 0.02,
        *,
        measured_gap_mm: float | None = None,
    ) -> ControlOutput:
        q, velocity, dt = float(position_rad), float(velocity_rad_s), float(dt)
        if not all(math.isfinite(value) for value in (q, velocity, dt)) or dt <= 0:
            raise ValueError("joint state and dt must be finite; dt must be positive")
        linkage = self.model.linkage
        if measured_gap_mm is None:
            measured_gap_mm = linkage.pad_gap_m(q) * 1000
        if not math.isfinite(measured_gap_mm):
            raise ValueError("measured_gap_mm must be finite")
        measured_gap_mm = max(0.0, min(84.0, measured_gap_mm))
        lookup = self.force_lookup
        limit = self.model.torque_limit_nm(lookup)
        old_q_ref = linkage.drive_for_gap_m(self.reference_gap_mm / 1000)
        if self._reverse_pending:
            self.reference_gap_mm = measured_gap_mm
            self.integral_torque_nm = 0.0
            self._reverse_pending = False
        if self._pause_pending:
            # Transfer the proportional preload into the integral before
            # freezing at the measured position. This preserves grasp effort
            # and also stops a moving, unloaded position reference.
            preload = self._pause_start_integral_nm + self.kp * (old_q_ref - q)
            self.integral_torque_nm = preload
            self._paused_hold_sign = (
                math.copysign(1.0, preload)
                if abs(velocity) <= 0.02 and abs(preload) >= 0.95 * self._pause_start_limit_nm
                else 0.0
            )
            # Moving mimic joints can have a transient pad-position offset.
            # Freeze the physical gap during a stroke; a stopped grasp keeps
            # its joint posture and preload rather than unloading this offset.
            self.reference_gap_mm = (
                measured_gap_mm if abs(velocity) > 0.02 else max(0, min(84, linkage.pad_gap_m(q) * 1000))
            )
            self._pause_pending = False
        previous_q_ref = linkage.drive_for_gap_m(self.reference_gap_mm / 1000)
        travel = self.requested_speed_mm_s * dt
        error_mm = self.position_mm - self.reference_gap_mm
        self.reference_gap_mm += max(-travel, min(travel, error_mm))
        q_ref = linkage.drive_for_gap_m(self.reference_gap_mm / 1000)
        v_ref = (q_ref - previous_q_ref) / dt
        error = q_ref - q
        self.integral_torque_nm = max(-limit, min(limit, self.integral_torque_nm))
        if self.speed_percent == 0 and self._paused_hold_sign:
            # A stalled saturated grasp needs its holding effort after pause.
            # Preserve the frozen trajectory, and refer the torque preload to
            # the current joint posture. A force change then changes the cap
            # without a fixed position reference cancelling the added effort.
            self.integral_torque_nm = self._paused_hold_sign * limit
            return ControlOutput(
                q + self.integral_torque_nm / self.kp,
                0.0,
                self.reference_gap_mm,
                self.integral_torque_nm,
                limit,
                lookup,
            )
        pd_torque = self.kp * error + self.kv * (v_ref - velocity)
        candidate = max(-limit, min(limit, self.integral_torque_nm + self.ki * error * dt))
        candidate_total = pd_torque + candidate
        # Permit integration that unwinds saturation; stop integration which
        # pushes an already saturated command further against the limit.
        if abs(candidate_total) <= limit or error * candidate_total <= 0:
            self.integral_torque_nm = candidate
        return ControlOutput(
            q_ref + self.integral_torque_nm / self.kp,
            v_ref,
            self.reference_gap_mm,
            self.integral_torque_nm,
            limit,
            lookup,
        )

    def reset(self, position_mm: float = 84, *, target_mm: float | None = None, prime: bool = False) -> None:
        """Reset the reference; optionally seed a saturated pre-held grasp."""
        position = checked(position_mm, 0, 84, "position_mm")
        target = position if target_mm is None else checked(target_mm, 0, 84, "target_mm")
        self.reference_gap_mm = position
        self.position_mm = target
        self.integral_torque_nm = 0.0
        self._pause_pending = self._reverse_pending = False
        self._paused_hold_sign = 0.0
        self._pause_start_limit_nm = self._pause_start_integral_nm = 0.0
        if prime and target != position:
            self.integral_torque_nm = -math.copysign(self.model.torque_limit_nm(self.force_lookup), target - position)
