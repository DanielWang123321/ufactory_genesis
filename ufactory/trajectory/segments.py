"""Trajectory segments and program assembly for UFACTORY trajectory planning.

A :class:`Program` is an ordered list of :class:`Segment` instances. Each
segment resolves, at a given ``rate`` (Hz), to a dense stream of absolute
targets that the sim and real executors replay tick-for-tick:

* :class:`MoveJSegment`  - joint-space LSPB (trapezoidal) motion.
* :class:`MoveLSegment`  - Cartesian straight-line motion of the configured EE.
* :class:`GripperSegment` - gripper gap open/close at a fixed EE pose.

Cartesian targets are in the **robot base frame** (flange / ``tcp_offset=0``).
:func:`ufactory.trajectory.execution.execute_real` consumes these directly; the
sim executor converts base to world before Genesis IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ufactory.trajectory import profile as P
from ufactory.types import FloatArray

SegmentKind = Literal["movej", "movel", "gripper"]


@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    duration: float
    v_max: float
    a_max: float
    label: str = ""
    # MoveJ
    q_start: FloatArray | None = None
    q_end: FloatArray | None = None
    # Optional explicit MoveJ target stream. Same sampling contract as
    # ``joint_lspb_samples``: row 0 is the first target after ``q_start`` and
    # the final row equals ``q_end``.
    q_samples: FloatArray | None = None
    # MoveL (link6 base-frame xyz, m; orientation fixed gripper-down)
    pose_start: FloatArray | None = None
    pose_end: FloatArray | None = None
    # Gripper (physical two-finger gap, m)
    gap_start: float | None = None
    gap_end: float | None = None
    samples_count: int = 0

    def length(self) -> str:
        if self.kind == "movej" and self.q_start is not None and self.q_end is not None:
            return f"{np.abs(self.q_end - self.q_start).max():.4f} rad"
        if self.kind == "movel" and self.pose_start is not None and self.pose_end is not None:
            return f"{np.linalg.norm(self.pose_end - self.pose_start) * 1000:.1f} mm"
        if self.kind == "gripper" and self.gap_start is not None and self.gap_end is not None:
            return f"{abs(self.gap_end - self.gap_start) * 1000:.1f} mm"
        return "n/a"

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"<{self.kind}:{self.label} dur={self.duration:.3f}s "
            f"v={self.v_max:.3f} a={self.a_max:.3f} N={self.samples_count} {self.length()}>"
        )

    def samples(self, rate: float) -> tuple[FloatArray, int]:
        """Dense absolute targets at ``rate`` (Hz).

        Returns ``(samples[N, dim], N)``. MoveJ -> joints (rad), MoveL -> link6
        xyz (m), Gripper -> gap (m, single column).
        """
        if self.kind == "movej":
            assert self.q_start is not None and self.q_end is not None
            if self.q_samples is not None:
                q = np.asarray(self.q_samples, dtype=np.float64)
                if q.ndim != 2 or q.shape[0] < 1:
                    raise ValueError(f"MoveJ segment {self.label!r} q_samples must be a non-empty 2D array")
                return q.copy(), int(q.shape[0])
            q, n, _ = P.joint_lspb_samples(
                self.q_start,
                self.q_end,
                rate=rate,
                v_max=self.v_max,
                a_max=self.a_max,
            )
            return q, n
        if self.kind == "movel":
            assert self.pose_start is not None and self.pose_end is not None
            p, n, _ = P.linear_cartesian_samples(
                self.pose_start,
                self.pose_end,
                rate=rate,
                v_max=self.v_max,
                a_max=self.a_max,
            )
            return p, n
        if self.kind == "gripper":
            assert self.gap_start is not None and self.gap_end is not None
            g, n, _ = P.gap_lspb_samples(
                self.gap_start,
                self.gap_end,
                rate=rate,
                duration_s=self.duration,
                a_max=self.a_max,
            )
            return g, n
        raise ValueError(f"unknown segment kind: {self.kind}")


@dataclass
class JointLimits:
    """(speed, mvacc) -> (v_max, a_max) mapping shared across segments."""

    v_max_rad_s: float
    a_max_rad_s2: float
    # Derived limits for the linear/Cartesian profile over the working reach.
    v_max_lin_m_s: float
    a_max_lin_m_s2: float

    @classmethod
    def from_speed_mvacc(
        cls,
        speed_rad_s: float,
        mvacc_rad_s2: float,
        *,
        reach_m: float = P._DEFAULT_LINK_LINEAR_REACH_M,
        linear_speed_m_s: float | None = None,
        linear_acc_m_s2: float | None = None,
    ) -> JointLimits:
        v_j, a_j = P.joint_limits(speed_rad_s, mvacc_rad_s2)
        v_l, a_l = P.linear_limits_from_joint(speed_rad_s, mvacc_rad_s2, reach_m=reach_m)
        if linear_speed_m_s is not None:
            v_l = float(linear_speed_m_s) if float(linear_speed_m_s) > 0.0 else P._EPS
        if linear_acc_m_s2 is not None:
            a_l = float(linear_acc_m_s2) if float(linear_acc_m_s2) > 0.0 else P._EPS
        return cls(v_j, a_j, v_l, a_l)


def _finalize(seg: Segment, rate: float) -> Segment:
    _, n = seg.samples(rate)
    return _replace(seg, samples_count=n)


def _replace(seg: Segment, **kw: Any) -> Segment:
    return Segment(**{**seg.__dict__, **kw})


def make_movej(
    q_start: FloatArray,
    q_end: FloatArray,
    *,
    rate: float,
    limits: JointLimits,
    label: str = "",
) -> Segment:
    q0 = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q_end, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape:
        raise ValueError(f"MoveJ shape mismatch: {q0.shape} vs {q1.shape}")
    samples = P.joint_lspb_samples(q0, q1, rate=rate, v_max=limits.v_max_rad_s, a_max=limits.a_max_rad_s2)
    return _finalize(
        Segment(
            kind="movej",
            duration=samples[2],
            v_max=limits.v_max_rad_s,
            a_max=limits.a_max_rad_s2,
            label=label,
            q_start=q0,
            q_end=q1,
        ),
        rate,
    )


def make_movel(
    pose_start: FloatArray,
    pose_end: FloatArray,
    *,
    rate: float,
    limits: JointLimits,
    label: str = "",
) -> Segment:
    p0 = np.asarray(pose_start, dtype=np.float64).reshape(3)
    p1 = np.asarray(pose_end, dtype=np.float64).reshape(3)
    samples = P.linear_cartesian_samples(p0, p1, rate=rate, v_max=limits.v_max_lin_m_s, a_max=limits.a_max_lin_m_s2)
    return _finalize(
        Segment(
            kind="movel",
            duration=samples[2],
            v_max=limits.v_max_lin_m_s,
            a_max=limits.a_max_lin_m_s2,
            label=label,
            pose_start=p0,
            pose_end=p1,
        ),
        rate,
    )


def make_gripper(
    gap_start: float,
    gap_end: float,
    *,
    rate: float,
    duration_s: float,
    a_max_m_s2: float = 0.5,
    label: str = "",
) -> Segment:
    g0 = float(gap_start)
    g1 = float(gap_end)
    samples = P.gap_lspb_samples(g0, g1, rate=rate, duration_s=duration_s, a_max=a_max_m_s2)
    return _finalize(
        Segment(
            kind="gripper",
            duration=samples[2],
            v_max=float(samples[0].max()),
            a_max=a_max_m_s2,
            label=label,
            gap_start=g0,
            gap_end=g1,
        ),
        rate,
    )


@dataclass
class Program:
    segments: list[Segment] = field(default_factory=list)
    rate: float = 50.0
    limits: JointLimits | None = None
    robot_key: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        return float(sum(s.duration for s in self.segments))

    @property
    def total_ticks(self) -> int:
        return int(sum(s.samples_count for s in self.segments))

    def iter_samples(self) -> list[tuple[str, FloatArray]]:
        out: list[tuple[str, FloatArray]] = []
        for seg in self.segments:
            arr, _ = seg.samples(self.rate)
            out.append((seg.kind, arr))
        return out
