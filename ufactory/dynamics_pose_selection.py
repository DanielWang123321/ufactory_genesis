"""Helpers for selecting dynamics validation poses with workspace coverage."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

EE_Y_SIDE_POS = "y+"
EE_Y_SIDE_NEG = "y-"
EE_Y_SIDE_NEUTRAL = "neutral"

T = TypeVar("T")


@dataclass(frozen=True)
class PoseCandidate:
    name: str
    q: np.ndarray
    ee_x_mm: float
    ee_y_mm: float
    ee_z_mm: float
    tau_norm: float
    ee_y_side: str
    collision_ok: bool = False
    collision_note: str = ""
    move_strategy: str = "direct"


def classify_ee_y_side(y_mm: float, *, y_tol_mm: float = 10.0) -> str:
    if y_mm > y_tol_mm:
        return EE_Y_SIDE_POS
    if y_mm < -y_tol_mm:
        return EE_Y_SIDE_NEG
    return EE_Y_SIDE_NEUTRAL


def _joint_diversity_key(q: np.ndarray) -> tuple[int, int, int]:
    j1_sign = 0 if q[0] >= 0 else 1
    j2_sign = 0 if q[1] >= 0 else 1
    j3_deep = 0 if q[2] < -1.0 else 1
    return (j1_sign, j2_sign, j3_deep)


def select_diverse_from_pool(
    candidates: Sequence[T],
    count: int,
    *,
    min_separation_rad: float,
    q_of: Callable[[T], np.ndarray],
    score_of: Callable[[T], float] | None = None,
    diversity_key_of: Callable[[T], tuple] | None = None,
    already_selected: Sequence[T] = (),
) -> list[T]:
    """Pick diverse items from one pool using torque bucketing and joint-space spacing."""
    if count <= 0:
        return []
    if len(candidates) <= count:
        return list(candidates)

    score_fn = score_of or (lambda _: 0.0)
    key_fn = diversity_key_of or (lambda item: _joint_diversity_key(q_of(item)))
    sorted_c = sorted(candidates, key=score_fn)
    n = len(sorted_c)
    bucket_size = max(1, n // 3)
    buckets = [
        sorted_c[:bucket_size],
        sorted_c[bucket_size : 2 * bucket_size],
        sorted_c[2 * bucket_size :],
    ]

    selected: list[T] = list(already_selected)

    def too_close(item: T) -> bool:
        q = q_of(item)
        for other in selected:
            if float(np.linalg.norm(q - q_of(other))) < min_separation_rad:
                return True
        return False

    while len(selected) < count + len(already_selected):
        progressed = False
        for bucket in buckets:
            if len(selected) >= count + len(already_selected):
                break
            for item in sorted(bucket, key=key_fn):
                if item in selected or too_close(item):
                    continue
                selected.append(item)
                progressed = True
                break
        if not progressed:
            for item in sorted_c:
                if item in selected or too_close(item):
                    continue
                selected.append(item)
                if len(selected) >= count + len(already_selected):
                    break
            break

    return selected[len(already_selected) : len(already_selected) + count]


def select_stratified_by_ee_y(
    candidates: Sequence[PoseCandidate],
    *,
    n_y_pos: int,
    n_y_neg: int,
    y_tol_mm: float = 10.0,
    min_separation_rad: float = 0.3,
) -> list[PoseCandidate]:
    """Select poses with balanced EE y+ / y- coverage."""
    y_pos_pool = [c for c in candidates if classify_ee_y_side(c.ee_y_mm, y_tol_mm=y_tol_mm) == EE_Y_SIDE_POS]
    y_neg_pool = [c for c in candidates if classify_ee_y_side(c.ee_y_mm, y_tol_mm=y_tol_mm) == EE_Y_SIDE_NEG]

    if len(y_pos_pool) < n_y_pos:
        raise ValueError(f"Not enough y+ candidates: need {n_y_pos}, have {len(y_pos_pool)}")
    if len(y_neg_pool) < n_y_neg:
        raise ValueError(f"Not enough y- candidates: need {n_y_neg}, have {len(y_neg_pool)}")

    selected_pos = select_diverse_from_pool(
        y_pos_pool,
        n_y_pos,
        min_separation_rad=min_separation_rad,
        q_of=lambda c: c.q,
        score_of=lambda c: c.tau_norm,
        diversity_key_of=lambda c: _joint_diversity_key(c.q),
    )
    selected_neg = select_diverse_from_pool(
        y_neg_pool,
        n_y_neg,
        min_separation_rad=min_separation_rad,
        q_of=lambda c: c.q,
        score_of=lambda c: c.tau_norm,
        diversity_key_of=lambda c: _joint_diversity_key(c.q),
        already_selected=selected_pos,
    )
    return selected_pos + selected_neg


def order_poses_greedy(candidates: Sequence[PoseCandidate]) -> list[PoseCandidate]:
    """Greedy order from home minimizing joint-space distance."""
    if not candidates:
        return []

    remaining = list(candidates)
    ordered: list[PoseCandidate] = []
    current = np.zeros(6, dtype=np.float64)

    while remaining:
        best_i = 0
        best_score = float("inf")
        for i, candidate in enumerate(remaining):
            dist = float(np.linalg.norm(candidate.q - current))
            score = abs(dist - 1.5)
            if score < best_score:
                best_score = score
                best_i = i
        chosen = remaining.pop(best_i)
        ordered.append(chosen)
        current = chosen.q.copy()
    return ordered
