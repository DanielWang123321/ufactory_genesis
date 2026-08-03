"""Genesis-free helpers for interpreting training metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean


def metrics_by_iteration(
    rows: Iterable[Mapping[str, str]],
) -> dict[int, list[Mapping[str, str]]]:
    """Group CSV rows by their explicit training iteration."""

    grouped: dict[int, list[Mapping[str, str]]] = {}
    for row in rows:
        if "training_iteration" not in row:
            raise ValueError("metrics CSV has no training_iteration column")
        iteration = int(float(row["training_iteration"]))
        if iteration > 0:
            grouped.setdefault(iteration, []).append(row)
    return grouped


def mean_iteration_metric(
    grouped: Mapping[int, list[Mapping[str, str]]],
    iteration: int,
    key: str,
) -> float:
    block = grouped[iteration]
    if not block or key not in block[0]:
        return float("nan")
    return mean(float(row[key]) for row in block)


def grasp_collapse_reason(
    grouped: Mapping[int, list[Mapping[str, str]]],
    *,
    floor: float,
    grace_iterations: int,
    failure_iterations: int,
) -> str | None:
    """Return an explicit stop reason only after a full consecutive low window."""

    if not 0.0 <= floor <= 1.0:
        raise ValueError("grasp floor must be in [0, 1]")
    if grace_iterations < 0 or failure_iterations <= 0:
        raise ValueError("monitor iteration counts are invalid")
    eligible = sorted(iteration for iteration in grouped if iteration > grace_iterations)
    if len(eligible) < failure_iterations:
        return None
    window_iterations = eligible[-failure_iterations:]
    window = [
        mean_iteration_metric(grouped, iteration, "learned_grasp_success_rate") for iteration in window_iterations
    ]
    if max(window) >= floor:
        return None
    latest = window_iterations[-1]
    return f"grasp_collapse iteration={latest} max_last_{failure_iterations}={max(window):.6f} floor={floor:.6f}"


__all__ = [
    "grasp_collapse_reason",
    "mean_iteration_metric",
    "metrics_by_iteration",
]
