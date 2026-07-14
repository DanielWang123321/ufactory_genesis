from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.safety import ArmState, CollisionResult, FaultClass
from ufactory.trajectory.execution import ExecutionBindings, execute_real
from ufactory.trajectory.preflight import create_safety_gate
from ufactory.trajectory.segments import Program, Segment


class FakeKinematics:
    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        return np.asarray((0.3, 0.0, 0.3))

    def inverse(self, pose: np.ndarray, seed_q_rad: np.ndarray) -> np.ndarray:
        return np.asarray(seed_q_rad)


class ClearCollision:
    def check(self, q_rad, *, stage, gripper_drive=None):
        return CollisionResult(False, 1.0, "link1", "link3")

    def check_all(self, q_rad, *, stage, gripper_drive=None):
        return (self.check(q_rad, stage=stage, gripper_drive=gripper_drive),)


class FakeClock:
    def __init__(self, wait_lateness_ns=()):
        self.now = 0
        self.wait_lateness_ns = deque(wait_lateness_ns)

    def monotonic_ns(self) -> int:
        return self.now

    def wait_until_ns(self, deadline_ns: int) -> None:
        self.now = max(self.now, deadline_ns)
        if self.wait_lateness_ns:
            self.now += self.wait_lateness_ns.popleft()


class FakeTransport:
    def __init__(self, clock: FakeClock, *, send_codes=(), stale=False):
        self.clock = clock
        self.send_codes = deque(send_codes)
        self.stale = stale
        self.state_code = 2
        self.send_times: list[int] = []
        self.targets: list[np.ndarray] = []
        self.pause_calls = 0
        self.stop_calls = 0
        self.disconnect_calls = 0

    def read_state(self) -> ArmState:
        stamp = self.clock.now - 100_000_000 if self.stale else self.clock.now
        return ArmState(
            q_rad=np.zeros(6),
            monotonic_ns=stamp,
            ready=self.state_code not in (3, 4),
            state_code=self.state_code,
            serial_number="XI130506XXXXXX",
            robot_key="xarm6_1305",
        )

    def send_joint_target(self, q_rad: np.ndarray) -> int:
        self.send_times.append(self.clock.now)
        self.targets.append(np.asarray(q_rad).copy())
        return self.send_codes.popleft() if self.send_codes else 0

    def send_cartesian_target(self, pose: np.ndarray) -> int:
        raise AssertionError("joint-only timing fixture")

    def pause(self) -> int:
        self.pause_calls += 1
        self.state_code = 3
        return 0

    def stop(self) -> int:
        self.stop_calls += 1
        self.state_code = 4
        return 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def _approved(samples: int = 4):
    config = load_runtime_config("xarm6")
    q0 = np.zeros(6)
    rows = np.stack([q0 + (index + 1) * 0.0005 for index in range(samples)])
    program = Program(
        [
            Segment(
                "movej",
                samples / 50.0,
                1.0,
                12.0,
                "approach",
                q_start=q0,
                q_end=rows[-1],
                q_samples=rows,
                samples_count=samples,
            )
        ],
        rate=50.0,
        robot_key="xarm6_1305",
    )
    urdf = Path(config.robot.assets_dir) / config.robot.urdf
    gate = create_safety_gate(
        config,
        kinematics=FakeKinematics(),
        collision=ClearCollision(),
        calibration_sha256="a" * 64,
        scene_sha256="b" * 64,
        urdf_path=urdf,
    )
    report = gate.preflight(program, executor="servo_j")
    assert report.passed
    approved = gate.approve(program, report, expected_serial_number="XI130506XXXXXX")
    bindings = ExecutionBindings(
        robot_key="xarm6_1305",
        serial_number="XI130506XXXXXX",
        config_sha256=report.config_sha256,
        urdf_sha256=report.urdf_sha256,
        calibration_sha256=report.calibration_sha256,
        scene_sha256=report.scene_sha256,
    )
    return approved, bindings


def test_minor_lateness_resets_deadline_and_never_catches_up():
    approved, bindings = _approved(3)
    dt = 20_000_000
    clock = FakeClock((dt * 3 // 10, 0, 0))
    transport = FakeTransport(clock)
    result = execute_real(approved, transport=transport, bindings=bindings, confirm_real=True, clock=clock)
    assert result.completed
    assert result.minor_lateness_count == 1
    assert transport.send_times == [0, 26_000_000, 46_000_000]


def test_three_minor_misses_pause_without_auto_resume():
    approved, bindings = _approved(4)
    dt = 20_000_000
    clock = FakeClock((dt * 3 // 10,) * 3)
    transport = FakeTransport(clock)
    result = execute_real(approved, transport=transport, bindings=bindings, confirm_real=True, clock=clock)
    assert not result.completed
    assert result.fault is not None and result.fault.fault_class is FaultClass.PAUSE
    assert transport.pause_calls == 1
    assert transport.stop_calls == 0


def test_one_full_period_miss_stops():
    approved, bindings = _approved(2)
    clock = FakeClock((20_000_001,))
    transport = FakeTransport(clock)
    result = execute_real(approved, transport=transport, bindings=bindings, confirm_real=True, clock=clock)
    assert result.fault is not None and result.fault.fault_class is FaultClass.STOP
    assert transport.stop_calls == 1


def test_not_ready_retries_the_same_target_within_deadline():
    approved, bindings = _approved(1)
    clock = FakeClock()
    transport = FakeTransport(clock, send_codes=(9, 9, 0))
    result = execute_real(
        approved,
        transport=transport,
        bindings=bindings,
        confirm_real=True,
        clock=clock,
        not_ready_codes=frozenset((9,)),
    )
    assert result.completed
    assert len(transport.targets) == 3
    assert all(np.array_equal(transport.targets[0], target) for target in transport.targets[1:])


def test_not_ready_deadline_and_stale_feedback_have_expected_faults():
    approved, bindings = _approved(1)
    clock = FakeClock()
    transport = FakeTransport(clock, send_codes=(9,) * 30)
    result = execute_real(
        approved,
        transport=transport,
        bindings=bindings,
        confirm_real=True,
        clock=clock,
        not_ready_codes=frozenset((9,)),
    )
    assert result.fault is not None and result.fault.fault_class is FaultClass.PAUSE

    clock = FakeClock()
    stale_transport = FakeTransport(clock, stale=True)
    stale = execute_real(
        approved,
        transport=stale_transport,
        bindings=bindings,
        confirm_real=True,
        clock=clock,
    )
    assert stale.fault is not None and stale.fault.fault_class is FaultClass.STOP


def test_real_execution_requires_explicit_confirmation():
    approved, bindings = _approved(1)
    with pytest.raises(RuntimeError, match="confirm-real"):
        execute_real(
            approved,
            transport=FakeTransport(FakeClock()),
            bindings=bindings,
            confirm_real=False,
        )


def test_on_tick_receives_segment_and_tick_index():
    approved, bindings = _approved(3)
    clock = FakeClock()
    transport = FakeTransport(clock)
    seen: list[tuple[str, int]] = []

    def on_tick(segment, tick_idx: int) -> None:
        seen.append((segment.label or segment.kind, tick_idx))

    result = execute_real(
        approved,
        transport=transport,
        bindings=bindings,
        confirm_real=True,
        clock=clock,
        on_tick=on_tick,
    )
    assert result.completed
    assert seen == [("approach", 0), ("approach", 1), ("approach", 2)]


def test_on_tick_exception_becomes_stop_fault():
    approved, bindings = _approved(2)
    clock = FakeClock()
    transport = FakeTransport(clock)

    def on_tick(segment, tick_idx: int) -> None:
        raise RuntimeError("mirror boom")

    result = execute_real(
        approved,
        transport=transport,
        bindings=bindings,
        confirm_real=True,
        clock=clock,
        on_tick=on_tick,
    )
    assert not result.completed
    assert result.fault is not None
    assert result.fault.fault_class is FaultClass.STOP
    assert "on_tick failed" in result.fault.message
    assert transport.stop_calls == 1
    assert result.sent_samples == 1
