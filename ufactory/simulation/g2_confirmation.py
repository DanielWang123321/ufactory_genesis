"""G2 phase confirmation, grip/release deadlines, program preparation and acceptance summary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import math
import statistics

import numpy as np

from ufactory.simulation.g2_model import checked, speed_mm_s


CONFIRMATION_FRAMES = 2


@dataclass
class ConsecutiveFrameGate(ABC):
    rate_hz: float = 50.0
    consecutive_frames: int = 0
    previous_time_s: float | None = None
    confirmed_time_s: float | None = None

    @abstractmethod
    def matches(self, row: dict) -> bool:
        """Whether this sample satisfies the phase's completion condition."""

    def observe(self, row: dict) -> bool:
        if self.confirmed_time_s is not None:
            return True
        now = row["time_s"]
        adjacent = self.previous_time_s is not None and math.isclose(
            now - self.previous_time_s, 1 / self.rate_hz, rel_tol=0, abs_tol=1e-9
        )
        self.consecutive_frames = (self.consecutive_frames if adjacent else 0) + 1 if self.matches(row) else 0
        self.previous_time_s = now
        if self.consecutive_frames >= CONFIRMATION_FRAMES:
            self.confirmed_time_s = now
        return self.confirmed_time_s is not None


CONTACT_THRESHOLD_N = 0.2
FORCE_RELATIVE_TOLERANCE = 0.05


def grip_timeout_s(reference_gap_mm: float, position_mm: float, speed_percent: float) -> float:
    reference = checked(reference_gap_mm, 0, 84, "reference_gap_mm")
    position = checked(position_mm, 0, 84, "position_mm")
    speed = speed_mm_s(checked(speed_percent, 1e-9, 100, "speed_percent"))
    return max(2.0, abs(reference - position) / speed + 1.0)


def force_at_target(row: dict) -> bool:
    left, right, target = (row[key] for key in ("left_normal_n", "right_normal_n", "expected_static_force_n"))
    return (
        all(math.isfinite(value) for value in (left, right, target))
        and min(left, right) > CONTACT_THRESHOLD_N
        and target > 0
        and abs((left + right) / 2 - target) <= target * FORCE_RELATIVE_TOLERANCE + 1e-12
    )


class GripForceGate(ConsecutiveFrameGate):
    def matches(self, row: dict) -> bool:
        return force_at_target(row)


OPEN_POSITION_MM = 84.0
OPEN_TOLERANCE_MM = 1.0


def release_timeout_s(reference_gap_mm: float, speed_percent: float) -> float:
    # The reference can lag the physical pads at reversal under high force.
    # Allow its remaining stroke; this deadline never imposes a minimum wait.
    return grip_timeout_s(reference_gap_mm, OPEN_POSITION_MM, speed_percent)


def release_at_target(row: dict) -> bool:
    gap, left, right = (row[key] for key in ("gap_mm", "left_normal_n", "right_normal_n"))
    return (
        all(math.isfinite(value) for value in (gap, left, right))
        and abs(gap - OPEN_POSITION_MM) <= OPEN_TOLERANCE_MM
        and 0 <= min(left, right)
        and max(left, right) <= CONTACT_THRESHOLD_N
    )


class ReleaseGate(ConsecutiveFrameGate):
    def matches(self, row: dict) -> bool:
        return release_at_target(row)


def prepare_g2_program(program, settings):
    """Keep the arm path; gripper durations become state-confirmed deadlines."""
    if settings.speed_percent <= 0:
        raise ValueError("automatic G2 manipulation requires speed_percent > 0")
    close = settings.position_mm / 1000
    segments = []
    for segment in program.segments:
        if segment.label in {"grip-settle", "pre-release-relax"}:
            continue
        if segment.label == "grip":
            segment = replace(
                segment,
                gap_start=0.084,
                gap_end=close,
                duration=grip_timeout_s(84, settings.position_mm, settings.speed_percent),
            )
        elif segment.label == "release":
            segment = replace(
                segment,
                gap_start=close,
                gap_end=0.084,
                duration=release_timeout_s(settings.position_mm, settings.speed_percent),
            )
        elif segment.kind == "gripper" and segment.label == "pre-release-settle":
            segment = replace(segment, gap_start=close, gap_end=close)
        _, count = segment.samples(program.rate)
        segments.append(replace(segment, samples_count=count))
    return replace(program, segments=segments)


@dataclass(frozen=True)
class G2Telemetry:
    """One sampled control period, shared by trajectory replay and standalone runs.

    Field names and order are the recorded schema; both producers build their
    CSV rows from this dataclass so a new field cannot drift between them.
    """

    time_s: float
    phase: str
    position_mm: float
    position_raw_counts: int
    speed_percent: float
    force_percent: float
    speed_command_mm_s: float
    motor_speed_command_rpm: int
    calibration_speed_rpm: float
    outside_calibration: bool
    curve_force_n: float
    expected_static_force_n: float
    reference_gap_mm: float
    gap_mm: float
    gap_velocity_mm_s: float
    drive_rad: float
    drive_velocity_rad_s: float
    drive_torque_nm: float
    torque_limit_nm: float
    integral_torque_nm: float
    motor_torque_estimate_nm: float
    left_normal_n: float
    right_normal_n: float
    sensor_force_n: float


def summarize(rows: list[dict], config, *, packaging: bool = False) -> dict:
    if not rows:
        raise ValueError("no pick-place samples")
    params = config.task.parameters
    target = np.array(params["fixed_target_position_m"]) * 1000
    initial_z = float(params["fixed_object_position_m"][2]) * 1000
    height = float(params["object_size_m"][2]) * 1000
    final = rows[-1]
    object_final = np.array([final[f"object_base_{axis}_mm"] for axis in "xyz"])
    rate = config.motion.rate_hz
    gate = GripForceGate(rate)
    confirmation_index = None
    confirmation = []
    grip_indices = [i for i, row in enumerate(rows) if row["phase"] == "grip"]
    for i in grip_indices:
        if gate.observe(rows[i]):
            confirmation_index = i
            confirmation = rows[i - CONFIRMATION_FRAMES + 1 : i + 1]
            break
    lift_index = next((i for i, row in enumerate(rows) if row["phase"] == "lift"), None)
    confirmed_before_lift = (
        confirmation_index is not None and lift_index is not None and confirmation_index < lift_index
    )
    delay = rows[lift_index]["time_s"] - gate.confirmed_time_s if confirmed_before_lift else None
    release_gate = ReleaseGate(rate)
    release_index = None
    release_confirmation = []
    release_indices = [i for i, row in enumerate(rows) if row["phase"] == "release"]
    for i in release_indices:
        if release_gate.observe(rows[i]):
            release_index = i
            release_confirmation = rows[i - CONFIRMATION_FRAMES + 1 : i + 1]
            break
    retreat_index = next(
        (i for i, row in enumerate(rows) if row["phase"] == ("post-release-settle" if packaging else "retreat")), None
    )
    confirmed_before_retreat = release_index is not None and retreat_index is not None and release_index < retreat_index
    release_delay = rows[retreat_index]["time_s"] - release_gate.confirmed_time_s if confirmed_before_retreat else None
    timing_checks = {
        "grip_force_confirmed": confirmation_index is not None,
        "lift_next_control_tick": (
            confirmed_before_lift
            and lift_index == confirmation_index + 1
            and math.isclose(delay, 1 / rate, rel_tol=0, abs_tol=1e-9)
        ),
        "no_fixed_grip_wait": (
            (confirmation_index is None or grip_indices[-1] == confirmation_index)
            and not any(row["phase"] == "grip-settle" for row in rows)
        ),
        "unconfirmed_grip_prevents_lift": lift_index is None or confirmed_before_lift,
        "release_confirmed": release_index is not None,
        "retreat_next_control_tick": (
            confirmed_before_retreat
            and retreat_index == release_index + 1
            and math.isclose(release_delay, 1 / rate, rel_tol=0, abs_tol=1e-9)
        ),
        "no_fixed_release_wait": release_index is None or release_indices[-1] == release_index,
        "unconfirmed_release_prevents_retreat": (
            not any(row["phase"] in {"retreat", "return-home"} for row in rows) or confirmed_before_retreat
        ),
    }
    lift = [row for row in rows if row["phase"] == "lift"]
    transit = [row for row in rows if row["phase"] == "transit"]
    carry = lift + transit
    slip = tool_displacement = None
    if confirmation and carry:
        baseline = np.mean([[row[f"object_in_pad_{axis}_mm"] for axis in "xyz"] for row in confirmation], axis=0)
        slip = max(
            float(np.linalg.norm(np.array([row[f"object_in_pad_{axis}_mm"] for axis in "xyz"]) - baseline))
            for row in carry
        )
        tool_baseline = np.mean([[row[f"object_in_tool_{axis}_mm"] for axis in "xyz"] for row in confirmation], axis=0)
        tool_displacement = max(
            float(np.linalg.norm(np.array([row[f"object_in_tool_{axis}_mm"] for axis in "xyz"]) - tool_baseline))
            for row in carry
        )
    place_error = float(np.linalg.norm(object_final - target))
    lift_height = lift[-1]["object_base_z_mm"] - initial_z if lift else None
    transit_clearance = min((row["object_base_z_mm"] - height / 2 for row in transit), default=None)
    contact_ratio = (
        statistics.fmean(min(row["left_normal_n"], row["right_normal_n"]) > CONTACT_THRESHOLD_N for row in transit)
        if transit
        else None
    )
    completed = bool(
        lift and transit and release_index is not None and any(row["phase"] == "return-home" for row in rows)
    )
    released = completed and all(
        max(row["left_normal_n"], row["right_normal_n"]) <= CONTACT_THRESHOLD_N for row in rows[-10:]
    )
    finite = all(math.isfinite(value) for row in rows for value in row.values() if isinstance(value, (float, int)))
    torque_ok = all(abs(row["drive_torque_nm"]) <= row["torque_limit_nm"] + 1e-5 for row in rows)
    transport_checks = {
        "finite_state": finite,
        "lifted_50mm": lift_height is not None and lift_height >= 50,
        "transit_clearance_20mm": transit_clearance is not None and transit_clearance >= 20,
        "transit_bilateral_95percent": contact_ratio is not None and contact_ratio >= 0.95,
        "carry_slip_within_5mm": slip is not None and slip <= 5,
        **({} if packaging else {"placed_within_10mm": completed and place_error <= 10}),
        "released": released,
        "opened_within_1mm": completed and abs(final["gap_mm"] - 84) <= 1,
        "object_settled": completed and final["object_speed_mm_s"] <= 5,
        "home_within_10mm": completed and final["home_error_mm"] <= 10,
        "torque_within_limit": torque_ok,
    }
    checks = timing_checks | transport_checks
    return {
        "passed": all(checks.values()),
        "timing_passed": all(timing_checks.values()),
        "transport_passed": all(transport_checks.values()),
        "timing_checks": timing_checks,
        "transport_checks": transport_checks,
        "source_example_thresholds_passed": completed
        and (packaging or place_error < 50)
        and final["home_error_mm"] < 10,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "grip_duration_s": (
            rows[grip_indices[-1]]["time_s"] - (rows[grip_indices[0] - 1]["time_s"] if grip_indices[0] else 0)
            if grip_indices
            else None
        ),
        "grip_force_confirmation_time_s": gate.confirmed_time_s,
        "grip_target_force_n": confirmation[-1]["expected_static_force_n"] if confirmation else None,
        "grip_actual_force_n": (
            (confirmation[-1]["left_normal_n"] + confirmation[-1]["right_normal_n"]) / 2 if confirmation else None
        ),
        "grip_confirmation_samples": [
            {
                key: row[key]
                for key in ("time_s", "expected_static_force_n", "left_normal_n", "right_normal_n", "gap_mm")
            }
            for row in confirmation
        ],
        "lift_start_time_s": rows[lift_index]["time_s"] if lift_index is not None else None,
        "force_to_lift_delay_s": delay,
        "release_duration_s": (
            rows[release_indices[-1]]["time_s"] - (rows[release_indices[0] - 1]["time_s"] if release_indices[0] else 0)
            if release_indices
            else None
        ),
        "release_confirmation_time_s": release_gate.confirmed_time_s,
        "release_confirmed_gap_mm": release_confirmation[-1]["gap_mm"] if release_confirmation else None,
        "release_confirmation_samples": [
            {key: row[key] for key in ("time_s", "gap_mm", "left_normal_n", "right_normal_n")}
            for row in release_confirmation
        ],
        "retreat_start_time_s": rows[retreat_index]["time_s"] if retreat_index is not None else None,
        "release_to_retreat_delay_s": release_delay,
        "place_error_mm": place_error,
        "home_error_mm": final["home_error_mm"],
        "lift_height_mm": lift_height,
        "transit_min_clearance_mm": transit_clearance,
        "transit_bilateral_ratio": contact_ratio,
        "carry_max_slip_mm": slip,
        "carry_max_tool_displacement_mm": tool_displacement,
        "contact_peak_n": max(row["sensor_force_n"] for row in rows),
        "drive_torque_peak_nm": max(abs(row["drive_torque_nm"]) for row in rows),
        "motor_torque_estimate_peak_nm": max(abs(row["motor_torque_estimate_nm"]) for row in rows),
        "final_gap_mm": final["gap_mm"],
        "final_object_speed_mm_s": final["object_speed_mm_s"],
        "duration_s": final["time_s"],
        "samples": len(rows),
    }
