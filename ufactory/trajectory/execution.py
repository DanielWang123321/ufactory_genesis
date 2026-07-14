"""Approved-only simulation and real execution application services."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

import numpy as np

from ufactory.safety import ApprovedProgram, ArmTransport, Clock, FaultClass, SafetyViolation, SystemClock
from ufactory.safety.gate import program_sha256
from ufactory.trajectory.segments import Segment
from ufactory.types import FloatArray


@dataclass(frozen=True)
class ExecutionBindings:
    """Current runtime identities compared against the approval report."""

    robot_key: str
    serial_number: str
    config_sha256: str
    urdf_sha256: str
    calibration_sha256: str
    scene_sha256: str


@dataclass(frozen=True)
class ExecutionFault:
    fault_class: FaultClass
    message: str
    sample_index: int
    state_confirmed: bool


@dataclass(frozen=True)
class ExecutionReport:
    completed: bool
    sent_samples: int
    minor_lateness_count: int
    max_lateness_ns: int
    fault: ExecutionFault | None = None
    warnings: tuple[str, ...] = ()


class ExecutionRejected(RuntimeError):
    """Approval, identity, or explicit confirmation was invalid."""


RuntimeMonitor = Callable[[FloatArray, str], SafetyViolation | None]
TickCallback = Callable[[Segment, int], None]


@dataclass(frozen=True)
class _Command:
    kind: Literal["joint", "cartesian", "gripper"]
    target: FloatArray
    stage: str
    segment: Segment
    tick_idx: int


def _commands(approved: ApprovedProgram) -> tuple[_Command, ...]:
    commands: list[_Command] = []
    for segment in approved.program.segments:
        samples, _ = segment.samples(approved.program.rate)
        stage = segment.label or segment.kind
        if segment.kind == "movej":
            kind: Literal["joint", "cartesian", "gripper"] = "joint"
        elif segment.kind == "movel":
            kind = "cartesian"
        else:
            kind = "gripper"
        for tick_idx, row in enumerate(samples):
            commands.append(_Command(kind, row.copy(), stage, segment, tick_idx))
    return tuple(commands)


def _validate_bindings(approved: ApprovedProgram, bindings: ExecutionBindings) -> None:
    report = approved.report
    current = {
        "program": program_sha256(approved.program),
        "robot": bindings.robot_key,
        "serial": bindings.serial_number,
        "config": bindings.config_sha256,
        "urdf": bindings.urdf_sha256,
        "calibration": bindings.calibration_sha256,
        "scene": bindings.scene_sha256,
    }
    expected = {
        "program": report.program_sha256,
        "robot": report.robot_key,
        "serial": approved.expected_serial_number,
        "config": report.config_sha256,
        "urdf": report.urdf_sha256,
        "calibration": report.calibration_sha256,
        "scene": report.scene_sha256,
    }
    changed = [name for name in expected if current[name] != expected[name]]
    if changed:
        raise ExecutionRejected(f"approval binding mismatch: {', '.join(changed)}")


def _confirm_fault_state(
    transport: ArmTransport,
    clock: Clock,
    fault_class: FaultClass,
    *,
    timeout_ns: int = 1_000_000_000,
) -> bool:
    expected_state = 3 if fault_class is FaultClass.PAUSE else 4
    rc = transport.pause() if fault_class is FaultClass.PAUSE else transport.stop()
    if rc != 0:
        return False
    deadline = clock.monotonic_ns() + timeout_ns
    while clock.monotonic_ns() < deadline:
        state = transport.read_state()
        if state.state_code == expected_state:
            return True
        clock.wait_until_ns(min(deadline, clock.monotonic_ns() + 10_000_000))
    return False


def _fault_report(
    transport: ArmTransport,
    clock: Clock,
    fault_class: FaultClass,
    message: str,
    *,
    sample_index: int,
    sent_samples: int,
    minor_lateness_count: int,
    max_lateness_ns: int,
) -> ExecutionReport:
    confirmed = _confirm_fault_state(transport, clock, fault_class)
    warnings: list[str] = []
    if not confirmed:
        transport.disconnect()
        warnings.append(
            "controller did not confirm pause/stop within 1 s; use the physical E-stop and inspect UFACTORY Studio"
        )
    return ExecutionReport(
        completed=False,
        sent_samples=sent_samples,
        minor_lateness_count=minor_lateness_count,
        max_lateness_ns=max_lateness_ns,
        fault=ExecutionFault(fault_class, message, sample_index, confirmed),
        warnings=tuple(warnings),
    )


def execute_real(
    approved: ApprovedProgram,
    *,
    transport: ArmTransport,
    bindings: ExecutionBindings,
    confirm_real: bool,
    clock: Clock | None = None,
    runtime_monitor: RuntimeMonitor | None = None,
    gripper_sender: Callable[[float], int] | None = None,
    on_tick: TickCallback | None = None,
    not_ready_codes: frozenset[int] = frozenset((1,)),
    max_joint_feedback_error_rad: float = 0.05,
) -> ExecutionReport:
    """Execute only an unchanged, identity-bound approved program.

    The scheduler never catches up with a burst.  After a minor miss the next
    deadline is reset to ``now + dt``.  No path in this function clears errors,
    warnings, or automatically resumes motion.

    ``on_tick`` runs after a successful send and runtime checks, before the
    period wait. Keep it non-blocking (e.g. ``AsyncMirrorBridge.on_tick``);
    exceptions become a STOP fault.
    """

    if not isinstance(approved, ApprovedProgram):
        raise TypeError("execute_real accepts ApprovedProgram only")
    if not confirm_real:
        raise ExecutionRejected("real motion requires explicit --confirm-real")
    _validate_bindings(approved, bindings)
    timing = clock or SystemClock()
    initial = transport.read_state()
    if initial.robot_key and initial.robot_key != bindings.robot_key:
        raise ExecutionRejected("transport robot model does not match approval")
    if initial.serial_number and initial.serial_number != bindings.serial_number:
        raise ExecutionRejected("transport serial number does not match approval")
    if initial.error_code != 0:
        raise ExecutionRejected(f"robot has an existing error: {initial.error_code}")
    if not initial.ready:
        raise ExecutionRejected("robot is not ready before motion enable")

    commands = _commands(approved)
    dt_ns = int(round(1_000_000_000.0 / approved.program.rate))
    minor_threshold_ns = int(round(dt_ns * 0.25))
    feedback_stale_ns = dt_ns * 2
    minor_window: deque[int] = deque()
    sent = 0
    minor_total = 0
    max_lateness = 0
    deadline = timing.monotonic_ns() + dt_ns

    for index, command in enumerate(commands):
        if not np.all(np.isfinite(command.target)):
            return _fault_report(
                transport,
                timing,
                FaultClass.STOP,
                "non-finite approved command encountered",
                sample_index=index,
                sent_samples=sent,
                minor_lateness_count=minor_total,
                max_lateness_ns=max_lateness,
            )

        while True:
            if command.kind == "joint":
                rc = transport.send_joint_target(command.target)
            elif command.kind == "cartesian":
                rc = transport.send_cartesian_target(command.target)
            else:
                rc = 0 if gripper_sender is None else int(gripper_sender(float(command.target.reshape(-1)[0])))
            now = timing.monotonic_ns()
            if rc == 0:
                break
            if rc not in not_ready_codes:
                return _fault_report(
                    transport,
                    timing,
                    FaultClass.STOP,
                    f"SDK send failed with code {rc}",
                    sample_index=index,
                    sent_samples=sent,
                    minor_lateness_count=minor_total,
                    max_lateness_ns=max_lateness,
                )
            if now >= deadline:
                return _fault_report(
                    transport,
                    timing,
                    FaultClass.PAUSE,
                    "not-ready persisted until the current deadline",
                    sample_index=index,
                    sent_samples=sent,
                    minor_lateness_count=minor_total,
                    max_lateness_ns=max_lateness,
                )
            timing.wait_until_ns(min(deadline, now + max(1, dt_ns // 20)))

        sent += 1
        state = transport.read_state()
        now = timing.monotonic_ns()
        feedback_q = np.asarray(state.q_rad, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(feedback_q)):
            return _fault_report(
                transport,
                timing,
                FaultClass.STOP,
                "feedback contains NaN or infinity",
                sample_index=index,
                sent_samples=sent,
                minor_lateness_count=minor_total,
                max_lateness_ns=max_lateness,
            )
        if state.monotonic_ns > now or now - state.monotonic_ns > feedback_stale_ns:
            return _fault_report(
                transport,
                timing,
                FaultClass.STOP,
                "feedback is stale or has an invalid timestamp",
                sample_index=index,
                sent_samples=sent,
                minor_lateness_count=minor_total,
                max_lateness_ns=max_lateness,
            )
        if state.error_code != 0 or not state.ready:
            return _fault_report(
                transport,
                timing,
                FaultClass.STOP,
                f"robot feedback is not safe (error={state.error_code}, ready={state.ready})",
                sample_index=index,
                sent_samples=sent,
                minor_lateness_count=minor_total,
                max_lateness_ns=max_lateness,
            )
        if command.kind == "joint" and feedback_q.shape == command.target.shape:
            feedback_error = float(np.max(np.abs(feedback_q - command.target), initial=0.0))
            if feedback_error > max_joint_feedback_error_rad:
                return _fault_report(
                    transport,
                    timing,
                    FaultClass.STOP,
                    f"joint feedback error {feedback_error:.6g} rad exceeds {max_joint_feedback_error_rad:.6g} rad",
                    sample_index=index,
                    sent_samples=sent,
                    minor_lateness_count=minor_total,
                    max_lateness_ns=max_lateness,
                )
        if runtime_monitor is not None:
            violation = runtime_monitor(feedback_q, command.stage)
            if violation is not None:
                return _fault_report(
                    transport,
                    timing,
                    violation.fault_class,
                    violation.message,
                    sample_index=index,
                    sent_samples=sent,
                    minor_lateness_count=minor_total,
                    max_lateness_ns=max_lateness,
                )
        if on_tick is not None:
            try:
                on_tick(command.segment, command.tick_idx)
            except Exception as exc:
                return _fault_report(
                    transport,
                    timing,
                    FaultClass.STOP,
                    f"on_tick failed: {exc}",
                    sample_index=index,
                    sent_samples=sent,
                    minor_lateness_count=minor_total,
                    max_lateness_ns=max_lateness,
                )

        timing.wait_until_ns(deadline)
        now = timing.monotonic_ns()
        lateness = max(0, now - deadline)
        max_lateness = max(max_lateness, lateness)
        if lateness > dt_ns:
            return _fault_report(
                transport,
                timing,
                FaultClass.STOP,
                "scheduler missed more than one full control period",
                sample_index=index,
                sent_samples=sent,
                minor_lateness_count=minor_total,
                max_lateness_ns=max_lateness,
            )
        if lateness > minor_threshold_ns:
            minor_total += 1
            minor_window.append(now)
            cutoff = now - 1_000_000_000
            while minor_window and minor_window[0] < cutoff:
                minor_window.popleft()
            if len(minor_window) >= 3:
                return _fault_report(
                    transport,
                    timing,
                    FaultClass.PAUSE,
                    "three minor deadline misses occurred within one second",
                    sample_index=index,
                    sent_samples=sent,
                    minor_lateness_count=minor_total,
                    max_lateness_ns=max_lateness,
                )
            deadline = now + dt_ns
        else:
            deadline += dt_ns

    confirmed = _confirm_fault_state(transport, timing, FaultClass.PAUSE)
    warnings = (
        ()
        if confirmed
        else ("normal completion pause was not confirmed; use the physical E-stop and inspect UFACTORY Studio",)
    )
    if not confirmed:
        transport.disconnect()
    return ExecutionReport(
        completed=confirmed,
        sent_samples=sent,
        minor_lateness_count=minor_total,
        max_lateness_ns=max_lateness,
        warnings=warnings,
    )


T = TypeVar("T")


def execute_sim(approved: ApprovedProgram, runner: Callable[[object], T]) -> T:
    """Run an unchanged approved program through an injected simulator."""

    if not isinstance(approved, ApprovedProgram):
        raise TypeError("execute_sim accepts ApprovedProgram only")
    if program_sha256(approved.program) != approved.report.program_sha256:
        raise ExecutionRejected("program changed after preflight")
    return runner(approved.program)
