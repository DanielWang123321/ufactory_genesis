"""Time-parameterized trajectory profiles for UFACTORY trajectory programs.

The runtime kernel here is a self-contained NumPy implementation of the
*linear-segment-with-parabolic-blends* (LSPB / trapezoidal) velocity law and a
Cartesian-linear task-space profile. For xArm ``MODE_SERVO`` this host-side
sampling is the safety-critical time law: ``set_servo_angle_j`` and
``set_servo_cartesian`` refresh absolute targets, and their SDK
``speed``/``mvacc``/``mvtime`` fields are reserved. The same absolute target
stream replayed in simulation (Genesis PD) and on the real arm is aligned per
tick by construction.

``roboticstoolbox-python`` is *not* a runtime dependency: the optional
``[trajectory]`` extra declares it so users who want the reference toolbox's
``trapezoidal`` / ``mtraj`` / ``ctraj`` tooling can install it, but importing
``ufactory.trajectory`` or the RL paths never triggers its (heavy) import.
All profile math in this module is plain NumPy and has zero hard dependency.

Unit policy
------------
* joint space (``MoveJ``): rad / rad·s⁻¹ / rad·s⁻²
* Cartesian task space (``MoveL``, flange / configured EE in robot base frame):
  m / m·s⁻¹ / m·s⁻²
* gripper gap (``Gripper``): m (two-finger physical gap)
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12
_DEFAULT_LINK_LINEAR_REACH_M = 0.4


def lspb_duration(distance: float, v_max: float, a_max: float) -> float:
    """Time-optimal LSPB duration for a scalar displacement ``distance``.

    A trapezoidal profile (accel / cruise / decel) is used when the cruise
    velocity ``v_max`` is reachable within ``a_max``; otherwise the triangular
    bang-bang-accel limit is used. End-points have zero velocity.
    """
    d = abs(float(distance))
    v = float(v_max)
    a = float(a_max)
    if d < _EPS or v <= 0.0 or a <= 0.0:
        return 0.0
    d_a = v * v / (2.0 * a)
    if d >= 2.0 * d_a - _EPS:
        t_a = v / a
        t_c = (d - 2.0 * d_a) / v
        return 2.0 * t_a + t_c
    return 2.0 * math.sqrt(d / a)


def _lspb_alpha_for_duration(d: float, v_max: float, a_max: float, duration: float) -> float:
    """Normalized accel-time fraction (0, 0.5] fitting displacement ``d`` into
    a fixed ``duration`` under acceleration bound ``a_max`` and cruise bound
    ``v_max``."""
    dur = float(duration)
    if dur <= 0.0 or d < _EPS or a_max <= 0.0:
        return 0.5
    t_a_v = v_max / a_max
    if t_a_v <= dur * 0.5 + _EPS and (v_max * dur - v_max * t_a_v) >= d - _EPS:
        t_a = t_a_v
    else:
        t_a = dur * 0.5
    alpha = max(_EPS, min(0.5, t_a / dur))
    disc = 1.0 - 4.0 * d / (a_max * dur * dur)
    if disc >= 0.0:
        alpha_fit = 0.5 * (1.0 - math.sqrt(disc))
        if _EPS <= alpha_fit <= 0.5:
            alpha = alpha_fit
    return alpha


def _lspb_displacement(u: np.ndarray, d: float, alpha: float) -> np.ndarray:
    """LSPB displacement over normalized time ``u`` in ``[0, 1]``.

    Returns values in ``[0, d]`` with zero velocity at the boundaries. ``alpha``
    is the normalized accel/decel time fraction (triangular when ``alpha=0.5``).
    """
    a = max(alpha, _EPS)
    area = 1.0 - a
    out = np.empty_like(u, dtype=np.float64)
    accel = u <= a
    decel = u >= (1.0 - a)
    cruise = ~(accel | decel)
    ua = u[accel]
    out[accel] = d * (ua * ua) / (2.0 * a * area)
    uc = u[cruise]
    out[cruise] = d * (a * 0.5 + (uc - a)) / area
    ud = u[decel]
    w_decel = (1.0 - 1.5 * a) + (ud - 0.5 * ud * ud - ((1.0 - a) - 0.5 * (1.0 - a) ** 2)) / a
    out[decel] = d * w_decel / area
    return out


def _sample_grid(n: int) -> np.ndarray:
    n = max(1, int(n))
    return np.arange(1, n + 1, dtype=np.float64) / float(n)


def joint_lspb_samples(
    q0: np.ndarray,
    q1: np.ndarray,
    *,
    rate: float,
    v_max: float,
    a_max: float,
) -> tuple[np.ndarray, int, float]:
    """Synchronized multi-DOF LSPB joint trajectory sampled at ``rate``.

    All DOFs share one duration (the bottleneck DOF), so joint endpoints are
    reached simultaneously. Returns ``(Q[N, ndof], N, duration_s)`` where the
    last row equals ``q1`` and row 0 is the first streamed target after ``q0``.
    """
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape:
        raise ValueError(f"joint shape mismatch q0={q0.shape} q1={q1.shape}")
    delta = q1 - q0
    dist = np.abs(delta)
    if np.max(dist) < _EPS:
        return q1.reshape(1, -1).copy(), 1, 0.0
    durations = [lspb_duration(d_ii, v_max, a_max) for d_ii in dist]
    duration = max(durations)
    if duration <= _EPS:
        duration = max(_EPS, float(np.max(dist)) / max(v_max, _EPS))
    # Never round a time-optimal duration down: doing so silently violates the
    # requested velocity/acceleration limits on the discrete command stream.
    n = max(1, int(math.ceil(duration * rate - _EPS)))
    duration = float(n) / float(rate)
    u = _sample_grid(n)
    q = np.empty((n, q0.size), dtype=np.float64)
    for i in range(q0.size):
        d_i = float(dist[i])
        if d_i < _EPS:
            q[:, i] = q1[i]
            continue
        s = 1.0 if q1[i] >= q0[i] else -1.0
        alpha = _lspb_alpha_for_duration(d_i, v_max, a_max, duration)
        disp = _lspb_displacement(u, d_i, alpha)
        q[:, i] = q0[i] + s * disp
    return q, n, duration


def linear_cartesian_samples(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    rate: float,
    v_max: float,
    a_max: float,
) -> tuple[np.ndarray, int, float]:
    """Cartesian straight-line LSPB profile (EE xyz in robot base frame).

    Path is the straight segment ``p0 -> p1``; scalar progress along the line
    follows an LSPB velocity law (trapezoidal/triangular). Orientation is owned
    by the executor or task template, so only ``xyz`` is parameterized. Returns
    ``(P[N, 3], N, duration_s)`` with the last row equal to ``p1``.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < _EPS:
        return p1.reshape(1, -1).copy(), 1, 0.0
    duration = lspb_duration(length, v_max, a_max)
    if duration <= _EPS:
        duration = max(_EPS, length / max(v_max, _EPS))
    n = max(1, int(math.ceil(duration * rate - _EPS)))
    duration = float(n) / float(rate)
    u = _sample_grid(n)
    alpha = _lspb_alpha_for_duration(length, v_max, a_max, duration)
    disp = _lspb_displacement(u, length, alpha)
    direction = vec / length
    pts = p0 + disp[:, None] * direction[None, :]
    return pts, n, duration


def gap_lspb_samples(
    gap_start: float,
    gap_end: float,
    *,
    rate: float,
    duration_s: float,
    a_max: float = 0.5,
) -> tuple[np.ndarray, int, float]:
    """Gripper G2 gap interpolation (m) as a parabolic-blend profile.

    Returns gap targets (m) sampled at ``rate`` over ``duration_s``. The gap
    distance is fit into the fixed duration under the accel bound ``a_max`` (a
    generous cruise cap is used, so the blend is bounded by acceleration); the
    velocity is zero at both endpoints.
    """
    d = abs(float(gap_end) - float(gap_start))
    duration = float(duration_s)
    if duration <= _EPS:
        return np.array([[float(gap_end)]], dtype=np.float64), 1, 0.0
    if d < _EPS:
        n = max(1, int(math.ceil(duration * rate - _EPS)))
        duration = float(n) / float(rate)
        return np.full((n, 1), float(gap_end), dtype=np.float64), n, duration
    n = max(1, int(math.ceil(duration * rate - _EPS)))
    duration = float(n) / float(rate)
    u = _sample_grid(n)
    v_cap = d / duration + a_max * duration
    alpha = _lspb_alpha_for_duration(d, v_cap, a_max, duration)
    disp = _lspb_displacement(u, d, alpha)
    s = 1.0 if gap_end >= gap_start else -1.0
    gaps = float(gap_start) + s * disp
    return gaps.reshape(-1, 1), n, duration


def joint_limits(speed_rad_s: float, mvacc_rad_s2: float) -> tuple[float, float]:
    """Map (speed, mvacc) to (v_max, a_max) for joint-space (MoveJ) profiles."""
    v = float(speed_rad_s)
    a = float(mvacc_rad_s2)
    return (v if v > 0 else _EPS, a if a > 0 else _EPS)


def linear_limits_from_joint(
    speed_rad_s: float,
    mvacc_rad_s2: float,
    *,
    reach_m: float = _DEFAULT_LINK_LINEAR_REACH_M,
) -> tuple[float, float]:
    """Derive (v_lin m/s, a_lin m/s²) from joint (speed, mvacc) via a working reach.

    A task-relevant linear reach is used so
    a single ``--speed-rad-s`` / ``--mvacc-rad-s2`` CLI pair drives both MoveJ
    and MoveL while staying physically reasonable.
    """
    r = float(reach_m) if reach_m > 0 else _DEFAULT_LINK_LINEAR_REACH_M
    return joint_limits(speed_rad_s * r, mvacc_rad_s2 * r)
