"""Real-robot streaming executor for the trajectory pipeline (xArm MODE_SERVO).

The xArm servo APIs are high-frequency target refresh commands: the controller
executes the latest target and the effective speed is the host-side finite
difference ``delta_target / dt``. SDK ``speed`` / ``mvacc`` / ``mvtime`` fields
are reserved for ``set_servo_angle_j`` / ``set_servo_cartesian`` and must not be
treated as safety limits. This executor therefore validates every streamed
target sequence before sending and reports the commanded finite-difference
speed/acceleration in dry-run and SDK-simulation modes.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ufactory.hardware import xarm as xc
from ufactory.dynamics import parse_joint_limits
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sdk_pos_mm
from ufactory.robots.paths import xarm6_urdf
from ufactory.trajectory.segments import Program, Segment

EXECUTOR_SERVO_J = "servo-j"
EXECUTOR_SERVO_CART = "servo-cartesian"
REAL_EXECUTORS = (EXECUTOR_SERVO_J, EXECUTOR_SERVO_CART)

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# xArm SDK set_gripper_g2_position valid speed range (mm/s).
GRIPPER_G2_MIN_SPEED_MM_S = 15.0
GRIPPER_G2_MAX_SPEED_MM_S = 225.0
GRIPPER_G2_POSITION_WARN_TOLERANCE_MM = 5.0


class TrajectorySafetyError(RuntimeError):
    """Raised when a segment target would violate real-robot safety limits."""


@dataclass(frozen=True)
class ServoLimits:
    """Host-side limits for xArm servo target streams.

    Units:
    * joint: rad, rad/s, rad/s^2
    * Cartesian xyz: mm, mm/s, mm/s^2
    * Cartesian rpy: rad, rad/s, rad/s^2
    """

    joint_speed_rad_s: float = 0.7
    joint_acc_rad_s2: float = 5.0
    cart_speed_mm_s: float = 200.0
    cart_acc_mm_s2: float = 1000.0
    orient_speed_rad_s: float = 0.7
    orient_acc_rad_s2: float = 5.0


@dataclass(frozen=True)
class ServoStreamStats:
    """Finite-difference statistics for one streamed servo segment."""

    kind: str
    label: str
    samples: int
    duration_s: float
    max_step: float = 0.0
    max_speed: float = 0.0
    max_acc: float = 0.0
    max_orient_step: float = 0.0
    max_orient_speed: float = 0.0
    max_orient_acc: float = 0.0

    def digest(self) -> str:
        if self.kind == "j":
            return (
                f"max_step={self.max_step:.5f}rad ({math.degrees(self.max_step):.2f}deg) "
                f"max_speed={self.max_speed:.3f}rad/s ({math.degrees(self.max_speed):.1f}deg/s) "
                f"max_acc={self.max_acc:.3f}rad/s^2 ({math.degrees(self.max_acc):.1f}deg/s^2)"
            )
        return (
            f"max_xyz_step={self.max_step:.2f}mm max_xyz_speed={self.max_speed:.1f}mm/s "
            f"max_xyz_acc={self.max_acc:.1f}mm/s^2 "
            f"max_rpy_step={self.max_orient_step:.5f}rad "
            f"max_rpy_speed={self.max_orient_speed:.3f}rad/s "
            f"max_rpy_acc={self.max_orient_acc:.3f}rad/s^2"
        )


@dataclass
class RealExecutorConfig:
    executor: str = EXECUTOR_SERVO_J
    rate: float = 50.0
    # Used to generate the host-side LSPB trajectory. For servo APIs these are
    # not relied on as firmware safety caps.
    speed_rad_s: float = 0.698
    mvacc_rad_s2: float = 5.0
    speed_mm_s: float = 200.0
    mvacc_mm_s2: float = 1000.0
    z_min_mm: float = 0.0
    dry_run: bool = True
    ip: str | None = None
    kinematics_suffix: str | None = None
    joint_margin_rad: float = 0.02
    servo_limits: ServoLimits = field(default_factory=ServoLimits)
    preposition_joint_speed_rad_s: float = 0.35
    preposition_joint_acc_rad_s2: float = 2.0
    preposition_cart_speed_mm_s: float = 100.0
    preposition_cart_acc_mm_s2: float = 500.0
    preposition_joint_tolerance_rad: float = 0.02
    preposition_cart_tolerance_mm: float = 2.0
    preposition_orient_tolerance_rad: float = 0.05
    sdk_sim_validate: bool = False
    sdk_sim_report_csv: str | None = None

    @classmethod
    def normalize_executor(cls, executor: str) -> str:
        v = str(executor).strip().lower()
        if v not in REAL_EXECUTORS:
            raise ValueError(f"Unknown real executor {executor!r}; expected one of {REAL_EXECUTORS}")
        return v


@dataclass(frozen=True)
class _PreparedSegment:
    seg: Segment
    kind: str
    samples: np.ndarray
    start: np.ndarray
    stats: ServoStreamStats


def _arm_joint_limits():
    """Return (lower, upper) rad joint limits for the xArm6 arm joints."""
    urdf = xarm6_urdf("xarm6_with_gripper.urdf")
    return parse_joint_limits(urdf, list(ARM_JOINT_NAMES))


def _segment_servo_targets(seg: Segment, rate: float) -> tuple[str, np.ndarray, np.ndarray]:
    """Return ``(kind, samples, start)`` in the units sent to the SDK."""
    samples, _ = seg.samples(rate)
    if seg.kind == "movej":
        if seg.q_start is None:
            raise ValueError(f"MoveJ segment {seg.label!r} has no q_start")
        return "j", np.asarray(samples, dtype=np.float64), np.asarray(seg.q_start, dtype=np.float64).reshape(-1)
    if seg.kind == "movel":
        if seg.pose_start is None:
            raise ValueError(f"MoveL segment {seg.label!r} has no pose_start")
        cart = np.zeros((samples.shape[0], 6), dtype=np.float64)
        cart[:, 0:3] = np.asarray(samples[:, 0:3], dtype=np.float64) * 1000.0
        cart[:, 3] = math.pi
        start = np.array(
            [
                float(seg.pose_start[0]) * 1000.0,
                float(seg.pose_start[1]) * 1000.0,
                float(seg.pose_start[2]) * 1000.0,
                math.pi,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        return "c", cart, start
    raise ValueError(f"segment {seg.label!r} is not an arm servo segment")


def compute_servo_stream_stats(
    kind: str,
    samples: np.ndarray,
    start: np.ndarray,
    *,
    rate: float,
    label: str = "",
) -> ServoStreamStats:
    """Compute finite-difference speed/acceleration for a servo target stream."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError("servo samples must be a non-empty 2D array")
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    if start_arr.size != arr.shape[1]:
        raise ValueError(f"start/sample shape mismatch: {start_arr.size} vs {arr.shape[1]}")
    full = np.vstack([start_arr.reshape(1, -1), arr])
    dt_rate = float(rate)
    if dt_rate <= 0:
        raise ValueError("rate must be positive")
    delta = np.diff(full, axis=0)
    if kind == "j":
        speed = np.abs(delta) * dt_rate
        acc = np.abs(np.diff(speed, axis=0)) * dt_rate if speed.shape[0] > 1 else np.zeros_like(speed)
        return ServoStreamStats(
            kind=kind,
            label=label,
            samples=arr.shape[0],
            duration_s=arr.shape[0] / dt_rate,
            max_step=float(np.max(np.abs(delta))),
            max_speed=float(np.max(speed)),
            max_acc=float(np.max(acc)) if acc.size else 0.0,
        )
    if kind == "c":
        xyz_delta = delta[:, 0:3]
        rpy_delta = delta[:, 3:6]
        xyz_step = np.linalg.norm(xyz_delta, axis=1)
        xyz_speed = xyz_step * dt_rate
        xyz_acc = np.abs(np.diff(xyz_speed)) * dt_rate if xyz_speed.size > 1 else np.zeros(1)
        orient_step = np.linalg.norm(rpy_delta, axis=1)
        orient_speed = orient_step * dt_rate
        orient_acc = np.abs(np.diff(orient_speed)) * dt_rate if orient_speed.size > 1 else np.zeros(1)
        return ServoStreamStats(
            kind=kind,
            label=label,
            samples=arr.shape[0],
            duration_s=arr.shape[0] / dt_rate,
            max_step=float(np.max(xyz_step)),
            max_speed=float(np.max(xyz_speed)),
            max_acc=float(np.max(xyz_acc)) if xyz_acc.size else 0.0,
            max_orient_step=float(np.max(orient_step)),
            max_orient_speed=float(np.max(orient_speed)),
            max_orient_acc=float(np.max(orient_acc)) if orient_acc.size else 0.0,
        )
    raise ValueError(f"unknown servo stream kind {kind!r}")


def validate_servo_stream(
    kind: str,
    samples: np.ndarray,
    start: np.ndarray,
    *,
    rate: float,
    limits: ServoLimits,
    label: str = "",
) -> ServoStreamStats:
    """Validate host-side finite-difference speed/acceleration limits."""
    stats = compute_servo_stream_stats(kind, samples, start, rate=rate, label=label)
    if kind == "j":
        if stats.max_speed > limits.joint_speed_rad_s + 1e-12:
            raise TrajectorySafetyError(
                f"{label or 'servo-j'} joint speed {stats.max_speed:.3f} rad/s "
                f"({math.degrees(stats.max_speed):.1f} deg/s) exceeds "
                f"{limits.joint_speed_rad_s:.3f} rad/s"
            )
        if stats.max_acc > limits.joint_acc_rad_s2 + 1e-12:
            raise TrajectorySafetyError(
                f"{label or 'servo-j'} joint acceleration {stats.max_acc:.3f} rad/s^2 "
                f"({math.degrees(stats.max_acc):.1f} deg/s^2) exceeds "
                f"{limits.joint_acc_rad_s2:.3f} rad/s^2"
            )
        return stats
    if kind == "c":
        if stats.max_speed > limits.cart_speed_mm_s + 1e-12:
            raise TrajectorySafetyError(
                f"{label or 'servo-cartesian'} Cartesian speed {stats.max_speed:.1f} mm/s "
                f"exceeds {limits.cart_speed_mm_s:.1f} mm/s"
            )
        if stats.max_acc > limits.cart_acc_mm_s2 + 1e-12:
            raise TrajectorySafetyError(
                f"{label or 'servo-cartesian'} Cartesian acceleration {stats.max_acc:.1f} mm/s^2 "
                f"exceeds {limits.cart_acc_mm_s2:.1f} mm/s^2"
            )
        if stats.max_orient_speed > limits.orient_speed_rad_s + 1e-12:
            raise TrajectorySafetyError(
                f"{label or 'servo-cartesian'} orientation speed {stats.max_orient_speed:.3f} rad/s "
                f"exceeds {limits.orient_speed_rad_s:.3f} rad/s"
            )
        if stats.max_orient_acc > limits.orient_acc_rad_s2 + 1e-12:
            raise TrajectorySafetyError(
                f"{label or 'servo-cartesian'} orientation acceleration {stats.max_orient_acc:.3f} rad/s^2 "
                f"exceeds {limits.orient_acc_rad_s2:.3f} rad/s^2"
            )
        return stats
    raise ValueError(f"unknown servo stream kind {kind!r}")


def check_segment_safety(
    seg: Segment,
    *,
    rate: float,
    lower: np.ndarray,
    upper: np.ndarray,
    z_min_mm: float,
    margin: float,
) -> None:
    """Validate all streamed samples against joint limits or z-min."""
    if seg.kind == "movej":
        kind, samples, start = _segment_servo_targets(seg, rate)
        assert kind == "j"
        all_q = np.vstack([start.reshape(1, -1), samples])
        lo = lower + margin
        hi = upper - margin
        bad = np.argwhere((all_q < lo) | (all_q > hi))
        if bad.size:
            row, col = bad[0]
            raise TrajectorySafetyError(
                f"MoveJ sample {int(row)} joint{int(col) + 1}={all_q[row, col]:.4f} rad "
                f"outside [{lo[col]:.4f}, {hi[col]:.4f}]"
            )
    elif seg.kind == "movel":
        kind, samples, start = _segment_servo_targets(seg, rate)
        assert kind == "c"
        all_cart = np.vstack([start.reshape(1, -1), samples])
        bad = np.where(all_cart[:, 2] < float(z_min_mm))[0]
        if bad.size:
            row = int(bad[0])
            raise TrajectorySafetyError(
                f"MoveL sample {row} link6 z={all_cart[row, 2]:.1f} mm below minimum {z_min_mm:.1f} mm"
            )


def _prepare_segment(seg: Segment, cfg: RealExecutorConfig, lower: np.ndarray, upper: np.ndarray) -> _PreparedSegment | None:
    if seg.kind == "gripper":
        return None
    check_segment_safety(
        seg,
        rate=cfg.rate,
        lower=lower,
        upper=upper,
        z_min_mm=cfg.z_min_mm,
        margin=cfg.joint_margin_rad,
    )
    kind, samples, start = _segment_servo_targets(seg, cfg.rate)
    stats = validate_servo_stream(
        kind,
        samples,
        start,
        rate=cfg.rate,
        limits=cfg.servo_limits,
        label=seg.label or seg.kind,
    )
    return _PreparedSegment(seg=seg, kind=kind, samples=samples, start=start, stats=stats)


def _first_arm_segment(program: Program) -> Segment | None:
    for seg in program.segments:
        if seg.kind in ("movej", "movel"):
            return seg
    return None


def _connect_arm(cfg: RealExecutorConfig):
    """Create an XArmAPI arm. Motion mode is set after optional prepositioning."""
    from xarm.wrapper.xarm_api import XArmAPI  # local import: optional SDK path

    if not cfg.ip:
        raise SystemExit("--ip (or XARM_IP) is required for non-dry-run execution")
    arm = XArmAPI(cfg.ip, is_radian=True)
    if cfg.sdk_sim_validate:
        print("  [sdk-sim] set_simulation_robot(True)")
        code = arm.set_simulation_robot(True)
        if code != 0:
            raise RuntimeError(f"set_simulation_robot(True) failed code={code}")
    return arm


def replay_real(
    program: Program,
    cfg: RealExecutorConfig,
    *,
    on_phase: Callable[[Segment], None] | None = None,
    on_arm: Callable[..., None] | None = None,
    on_tick: Callable[[Segment, int], None] | None = None,
    on_preposition_complete: Callable[[], None] | None = None,
) -> None:
    """Replay ``program`` on the xArm, SDK simulation, or as a dry-run digest."""
    cfg.executor = RealExecutorConfig.normalize_executor(cfg.executor)
    lower, upper = _arm_joint_limits()
    prepared: dict[int, _PreparedSegment] = {}
    for seg in program.segments:
        prep = _prepare_segment(seg, cfg, lower, upper)
        if prep is not None:
            prepared[id(seg)] = prep

    mode_label = "DRY-RUN" if cfg.dry_run else ("SDK-SIM" if cfg.sdk_sim_validate else "STREAM")
    print(
        f"[real {mode_label}] executor={cfg.executor} rate={cfg.rate:.0f}Hz "
        f"host_speed={cfg.speed_rad_s:.3f}rad/s host_mvacc={cfg.mvacc_rad_s2:.3f}rad/s^2 "
        f"z_min={cfg.z_min_mm:.1f}mm segments={len(program.segments)} tot={program.total_duration:.2f}s"
    )

    arm = None
    csv_writer = None
    csv_file = None
    try:
        if not cfg.dry_run:
            arm = _connect_arm(cfg)
            if on_arm is not None:
                on_arm(arm)
            _preposition_to_first_arm_segment(arm, _first_arm_segment(program), cfg)
            if on_preposition_complete is not None:
                on_preposition_complete()
            xc.prepare_arm_for_motion(arm, mode=xc.MODE_SERVO)
            if _gripper_motion_enabled(cfg):
                xc.prepare_gripper_g2_for_motion(arm)

        if cfg.sdk_sim_report_csv:
            report_path = Path(cfg.sdk_sim_report_csv)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = report_path.open("w", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "segment",
                    "kind",
                    "tick",
                    "timestamp_s",
                    "target",
                    "reported",
                    "commanded_speed",
                    "reported_speed",
                    "code",
                ],
            )
            csv_writer.writeheader()

        for seg in program.segments:
            if seg.kind == "gripper":
                _run_gripper_segment(seg, cfg, arm, on_tick=on_tick)
            elif seg.kind in ("movej", "movel"):
                _run_arm_segment(prepared[id(seg)], cfg, arm, csv_writer=csv_writer, on_tick=on_tick)
            else:
                raise ValueError(f"unknown segment kind: {seg.kind}")
            if on_phase is not None:
                on_phase(seg)

        if arm is not None:
            xc.assert_motion_ready(arm)
    finally:
        if csv_file is not None:
            csv_file.close()
        if arm is not None:
            if cfg.sdk_sim_validate:
                try:
                    code = arm.set_simulation_robot(False)
                    print(f"  [sdk-sim] set_simulation_robot(False) code={code}")
                except Exception:
                    pass
            if hasattr(arm, "disconnect"):
                try:
                    arm.disconnect()
                except Exception:
                    pass


def _wrap_rpy_diff_rad(diff: np.ndarray) -> np.ndarray:
    """Wrap roll/pitch/yaw differences to (-pi, pi].

    ``get_position`` extracts Euler angles via ``atan2`` (range ``(-pi, pi]``).
    Our commanded roll sits exactly on that branch cut (``+pi``), so the SDK
    can report back ``-pi`` for the identical physical orientation. A raw
    linear subtraction then sees a false ~2*pi error; wrapping avoids that.
    """
    return (diff + math.pi) % (2.0 * math.pi) - math.pi


def _preposition_to_first_arm_segment(arm, first_seg: Segment | None, cfg: RealExecutorConfig) -> None:
    if first_seg is None:
        return
    xc.prepare_arm_for_motion(arm, mode=xc.MODE_POSITION)
    kind, _samples, start = _segment_servo_targets(first_seg, cfg.rate)
    if kind == "j":
        target = start.tolist()
        code = arm.set_servo_angle(
            angle=target,
            speed=min(cfg.preposition_joint_speed_rad_s, cfg.speed_rad_s),
            mvacc=min(cfg.preposition_joint_acc_rad_s2, cfg.mvacc_rad_s2),
            wait=True,
            is_radian=True,
        )
        if code != 0:
            raise RuntimeError(f"preposition set_servo_angle failed code={code}")
        code, reported = arm.get_servo_angle(is_radian=True)
        if code != 0:
            raise RuntimeError(f"preposition get_servo_angle failed code={code}")
        err = float(np.max(np.abs(np.asarray(reported[: len(target)], dtype=np.float64) - start)))
        if err > cfg.preposition_joint_tolerance_rad:
            raise TrajectorySafetyError(
                f"preposition joint error {err:.4f} rad exceeds {cfg.preposition_joint_tolerance_rad:.4f} rad"
            )
        print(f"  [preposition] mode=0 joint target reached max_err={err:.4f}rad")
        return

    x, y, z, roll, pitch, yaw = start.tolist()
    set_code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=min(cfg.preposition_cart_speed_mm_s, cfg.speed_mm_s),
        mvacc=min(cfg.preposition_cart_acc_mm_s2, cfg.mvacc_mm_s2),
        wait=True,
        is_radian=True,
    )
    if set_code != 0:
        raise RuntimeError(f"preposition set_position failed code={set_code}")
    get_code, reported = arm.get_position(is_radian=True)
    if get_code != 0:
        raise RuntimeError(f"preposition get_position failed code={get_code}")
    reported_arr = np.asarray(reported[:6], dtype=np.float64)
    xyz_err = float(np.linalg.norm(reported_arr[:3] - start[:3]))
    rpy_diff_wrapped = _wrap_rpy_diff_rad(reported_arr[3:6] - start[3:6])
    rpy_err = float(np.linalg.norm(rpy_diff_wrapped))
    if xyz_err > cfg.preposition_cart_tolerance_mm:
        raise TrajectorySafetyError(
            f"preposition Cartesian xyz error {xyz_err:.2f} mm exceeds {cfg.preposition_cart_tolerance_mm:.2f} mm"
        )
    if rpy_err > cfg.preposition_orient_tolerance_rad:
        raise TrajectorySafetyError(
            f"preposition Cartesian rpy error {rpy_err:.4f} rad exceeds {cfg.preposition_orient_tolerance_rad:.4f} rad"
        )
    print(f"  [preposition] mode=0 Cartesian target reached xyz_err={xyz_err:.2f}mm rpy_err={rpy_err:.4f}rad")


def _run_arm_segment(
    prep: _PreparedSegment,
    cfg: RealExecutorConfig,
    arm,
    *,
    csv_writer=None,
    on_tick: Callable[[Segment, int], None] | None = None,
) -> None:
    seg = prep.seg
    first = np.round(prep.samples[0], 4)
    last = np.round(prep.samples[-1], 4)
    name = "servo-j" if prep.kind == "j" else "servo-cartesian"
    print(
        f"  [{name}:{seg.label}] N={prep.samples.shape[0]} dur=({prep.stats.duration_s:.2f}s) "
        f"first={first.tolist()} last={last.tolist()} {prep.stats.digest()}"
    )
    if cfg.dry_run or arm is None:
        if on_tick is not None:
            _replay_arm_ticks_dry_run(prep, cfg, on_tick)
        return
    max_overrun, reported_max_speed = _stream_servo(prep, cfg, arm, csv_writer=csv_writer, on_tick=on_tick)
    suffix = f" max_overrun={max_overrun * 1000.0:.2f}ms"
    if cfg.sdk_sim_validate:
        if prep.kind == "j":
            suffix += (
                f" reported_max_speed={reported_max_speed:.3f}rad/s "
                f"({math.degrees(reported_max_speed):.1f}deg/s)"
            )
        else:
            suffix += f" reported_max_speed={reported_max_speed:.1f}mm/s"
    print(f"    sent {prep.samples.shape[0]} ticks{suffix}")


def _gripper_motion_enabled(cfg: RealExecutorConfig) -> bool:
    """Real Gripper G2 SDK commands fire only for true real motion.

    ``--sdk-sim-validate`` sets ``cfg.dry_run = False`` to stream the arm's
    servo targets against the controller's simulated-robot mode, but that flag
    does not cover the gripper: a real ``set_gripper_g2_position`` call would
    still move the physical gripper hardware. Gate on both flags so
    sdk-sim-validate stays free of physical side effects, matching its
    existing "arm does not move" contract.
    """
    return not cfg.dry_run and not cfg.sdk_sim_validate


def _gripper_g2_target_speed_mm_s(seg: Segment) -> float:
    """Derive an SDK speed (mm/s) from the segment's planned gap change/duration.

    Keeps the physical gripper's timing aligned with the host-planned/mirrored
    duration (same sim-to-real alignment philosophy as the arm streams),
    clamped to the SDK's accepted range.
    """
    if seg.gap_start is None or seg.gap_end is None or seg.duration <= 0:
        return GRIPPER_G2_MIN_SPEED_MM_S
    span_mm = abs(float(seg.gap_end) - float(seg.gap_start)) * 1000.0
    speed = span_mm / seg.duration
    return float(min(max(speed, GRIPPER_G2_MIN_SPEED_MM_S), GRIPPER_G2_MAX_SPEED_MM_S))


def _run_gripper_segment(
    seg: Segment,
    cfg: RealExecutorConfig,
    arm,
    *,
    on_tick: Callable[[Segment, int], None] | None = None,
) -> None:
    _samples, n = seg.samples(cfg.rate)
    motion_enabled = _gripper_motion_enabled(cfg) and arm is not None
    if not motion_enabled:
        if cfg.dry_run:
            reason = "dry-run"
        elif cfg.sdk_sim_validate:
            reason = "sdk-sim-validate keeps the physical gripper idle"
        else:
            reason = "no arm connection"
        print(
            f"  [{seg.label}] gripper gap {seg.gap_start * 1000:.1f}mm -> "
            f"{seg.gap_end * 1000:.1f}mm over {seg.duration:.2f}s "
            f"(N={n}; {reason}, no SDK gripper command sent)"
        )
        if on_tick is not None:
            _pace_ticks(n, cfg.rate, lambda t: on_tick(seg, t))
        else:
            _pace(n, cfg.rate)
        return

    target_mm = gripper_g2_gap_m_to_sdk_pos_mm(seg.gap_end)
    speed_mm_s = _gripper_g2_target_speed_mm_s(seg)
    code = arm.set_gripper_g2_position(target_mm, speed=speed_mm_s, wait=False)
    if code != 0:
        raise RuntimeError(
            f"set_gripper_g2_position failed code={code} segment={seg.label!r} target={target_mm:.1f}mm"
        )
    print(
        f"  [{seg.label}] gripper gap {seg.gap_start * 1000:.1f}mm -> {seg.gap_end * 1000:.1f}mm "
        f"over {seg.duration:.2f}s (N={n}; set_gripper_g2_position target={target_mm:.1f}mm "
        f"speed={speed_mm_s:.1f}mm/s)"
    )
    if on_tick is not None:
        _pace_ticks(n, cfg.rate, lambda t: on_tick(seg, t))
    else:
        _pace(n, cfg.rate)

    read_code, reported_mm = arm.get_gripper_g2_position()
    if read_code != 0:
        print(f"    WARNING: get_gripper_g2_position failed code={read_code}; cannot verify gripper reached target")
        return
    err_mm = abs(float(reported_mm) - target_mm)
    status = "OK" if err_mm <= GRIPPER_G2_POSITION_WARN_TOLERANCE_MM else "WARNING"
    print(f"    gripper reached {reported_mm:.1f}mm (target {target_mm:.1f}mm, err={err_mm:.1f}mm) {status}")


def _read_reported_target(arm, kind: str) -> np.ndarray | None:
    if kind == "j":
        code, reported = arm.get_servo_angle(is_radian=True)
    else:
        code, reported = arm.get_position(is_radian=True)
    if code != 0:
        return None
    return np.asarray(reported[: (6 if kind == "c" else 6)], dtype=np.float64)


def _reported_distance(kind: str, prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    if prev is None or cur is None:
        return 0.0
    if kind == "j":
        return float(np.max(np.abs(cur - prev)))
    return float(np.linalg.norm(cur[:3] - prev[:3]))


def _nearest_target_index(kind: str, targets: np.ndarray, reported: np.ndarray | None) -> int | None:
    """Best-effort target index for SDK simulation feedback.

    The firmware simulation can coalesce feedback and report every second
    streamed target. Mapping the reported pose back to the nearest command lets
    the velocity report divide by the number of command ticks actually skipped.
    """
    if reported is None:
        return None
    if kind == "j":
        dist = np.max(np.abs(targets - reported.reshape(1, -1)), axis=1)
    else:
        dist = np.linalg.norm(targets[:, :3] - reported[:3].reshape(1, -1), axis=1)
    idx = int(np.argmin(dist))
    tol = 1e-3 if kind == "j" else 0.05
    return idx if float(dist[idx]) <= tol else None


def _assert_stream_health(arm, prep: _PreparedSegment, tick: int, code: int) -> None:
    if getattr(arm, "state", None) == xc.REPORT_STATE_STOPPING or getattr(arm, "error_code", 0) != 0:
        raise RuntimeError(
            f"SDK stream fault segment={prep.seg.label!r} tick={tick} code={code}; "
            f"{xc.format_arm_status(arm)}"
        )


def _stream_servo(
    prep: _PreparedSegment,
    cfg: RealExecutorConfig,
    arm,
    *,
    csv_writer=None,
    on_tick: Callable[[Segment, int], None] | None = None,
) -> tuple[float, float]:
    kind = prep.kind
    send = arm.set_servo_angle_j if kind == "j" else arm.set_servo_cartesian
    prime = xc.prime_servo_angle_j if kind == "j" else xc.prime_servo_cartesian
    prime_args = dict(speed=cfg.speed_rad_s, mvacc=cfg.mvacc_rad_s2) if kind == "j" else dict(
        speed=cfg.speed_mm_s, mvacc=cfg.mvacc_mm_s2
    )
    first = prep.samples[0]
    prime(arm, first.tolist(), **prime_args)
    xc.wait_for_servo_motion_ready(arm)

    tick_s = 1.0 / float(cfg.rate)
    next_deadline = time.monotonic()
    max_overrun = 0.0
    prev_reported = _read_reported_target(arm, kind) if cfg.sdk_sim_validate or csv_writer is not None else None
    max_reported_speed = 0.0
    prev_report_t = time.monotonic()

    full_targets = np.vstack([prep.start.reshape(1, -1), prep.samples])
    prev_report_idx = _nearest_target_index(kind, full_targets, prev_reported)
    for t, target in enumerate(prep.samples):
        target_list = target.tolist()
        if kind == "j":
            code = send(angles=target_list, speed=cfg.speed_rad_s, mvacc=cfg.mvacc_rad_s2, mvtime=0, is_radian=True)
        else:
            code = send(mvpose=target_list, speed=cfg.speed_mm_s, mvacc=cfg.mvacc_mm_s2, mvtime=0, is_radian=True)
        if code not in (0, xc.STATE_NOT_READY_SDK):
            raise RuntimeError(f"set_servo_{kind} failed code={code} at segment={prep.seg.label} tick={t}")
        if code == xc.STATE_NOT_READY_SDK:
            time.sleep(xc.SERVO_CMD_RETRY_S)
        _assert_stream_health(arm, prep, t, code)

        now = time.monotonic()
        reported = _read_reported_target(arm, kind) if cfg.sdk_sim_validate or csv_writer is not None else None
        report_distance = _reported_distance(kind, prev_reported, reported)
        report_speed = 0.0
        if reported is not None and prev_reported is None:
            prev_reported = reported
            prev_report_idx = _nearest_target_index(kind, full_targets, reported)
            prev_report_t = now
        elif report_distance > 1e-6:
            report_idx = _nearest_target_index(kind, full_targets, reported)
            tick_floor = tick_s
            if report_idx is not None and prev_report_idx is not None:
                tick_floor = max(1, abs(report_idx - prev_report_idx)) * tick_s
            dt_report = max(now - prev_report_t, tick_floor)
            report_speed = report_distance / dt_report
            prev_reported = reported
            prev_report_idx = report_idx
            prev_report_t = now
        max_reported_speed = max(max_reported_speed, report_speed)

        command_delta = target - full_targets[t]
        command_speed = float(np.max(np.abs(command_delta)) / tick_s) if kind == "j" else float(
            np.linalg.norm(command_delta[:3]) / tick_s
        )
        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "segment": prep.seg.label,
                    "kind": kind,
                    "tick": t,
                    "timestamp_s": f"{now:.6f}",
                    "target": target_list,
                    "reported": None if reported is None else reported.tolist(),
                    "commanded_speed": f"{command_speed:.9f}",
                    "reported_speed": f"{report_speed:.9f}",
                    "code": code,
                }
            )
        if on_tick is not None:
            on_tick(prep.seg, t)
        next_deadline += tick_s
        after_send = time.monotonic()
        overrun = max(0.0, after_send - next_deadline)
        max_overrun = max(max_overrun, overrun)
        sleep_s = next_deadline - after_send
        if sleep_s > 0:
            time.sleep(sleep_s)
    return max_overrun, max_reported_speed


def _replay_arm_ticks_dry_run(
    prep: _PreparedSegment,
    cfg: RealExecutorConfig,
    on_tick: Callable[[Segment, int], None],
) -> None:
    seg = prep.seg
    tick_s = 1.0 / float(cfg.rate)
    next_deadline = time.monotonic()
    for t in range(prep.samples.shape[0]):
        on_tick(seg, t)
        next_deadline += tick_s
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _pace_ticks(n: int, rate: float, tick_fn: Callable[[int], None]) -> None:
    if n <= 0:
        return
    tick_s = 1.0 / float(rate)
    next_deadline = time.monotonic()
    for t in range(n):
        tick_fn(t)
        next_deadline += tick_s
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _pace(n: int, rate: float) -> None:
    if n <= 0:
        return
    tick_s = 1.0 / float(rate)
    next_deadline = time.monotonic()
    for _ in range(n):
        next_deadline += tick_s
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
