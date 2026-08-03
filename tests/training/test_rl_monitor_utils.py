"""Tests for iteration-aware training monitoring."""

from __future__ import annotations

from examples.rl.pick_place.monitor_utils import grasp_collapse_reason, metrics_by_iteration


def _rows(rates: list[float]) -> list[dict[str, str]]:
    return [
        {
            "training_iteration": str(index),
            "learned_grasp_success_rate": str(rate),
        }
        for index, rate in enumerate(rates, start=1)
    ]


def test_monitor_waits_for_grace_and_full_failure_window() -> None:
    grouped = metrics_by_iteration(_rows([0.9] * 10 + [0.1] * 4))
    assert (
        grasp_collapse_reason(
            grouped,
            floor=0.75,
            grace_iterations=10,
            failure_iterations=5,
        )
        is None
    )


def test_monitor_records_clear_reason_for_consecutive_collapse() -> None:
    grouped = metrics_by_iteration(_rows([0.9] * 10 + [0.2] * 5))
    reason = grasp_collapse_reason(
        grouped,
        floor=0.75,
        grace_iterations=10,
        failure_iterations=5,
    )
    assert reason == "grasp_collapse iteration=15 max_last_5=0.200000 floor=0.750000"


def test_monitor_does_not_stop_when_one_recent_iteration_recovers() -> None:
    grouped = metrics_by_iteration(_rows([0.9] * 10 + [0.2, 0.2, 0.8, 0.2, 0.2]))
    assert (
        grasp_collapse_reason(
            grouped,
            floor=0.75,
            grace_iterations=10,
            failure_iterations=5,
        )
        is None
    )
