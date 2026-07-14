"""CLI argument checks for ufactory-grasp-place --visual."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ufactory.cli import grasp_place
from ufactory.safety import PreflightCheck, SafetyViolation, ViolationType
from ufactory.trajectory.mirror_executor import update_scene_visualizer
from ufactory.visualization.viewer import start_deferred_viewer


def test_visual_rejected_for_dry_run(capsys):
    with pytest.raises(SystemExit) as excinfo:
        grasp_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "dry-run",
                "--executor",
                "servo_j",
                "--visual",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--visual is only supported with --mode sim or --mode real" in err


def test_visual_rejected_for_sdk_sim(capsys):
    with pytest.raises(SystemExit) as excinfo:
        grasp_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "sdk-sim",
                "--executor",
                "servo_j",
                "--visual",
                "--ip",
                "127.0.0.1",
                "--calibration",
                "/tmp/fake.yaml",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--visual is only supported with --mode sim or --mode real" in err


def test_visual_start_hold_must_be_non_negative(capsys):
    with pytest.raises(SystemExit) as excinfo:
        grasp_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "real",
                "--executor",
                "servo_j",
                "--visual",
                "--visual-start-hold-s",
                "-0.1",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--visual-start-hold-s must be non-negative" in err


def test_preflight_failures_are_grouped_without_losing_counts(capsys):
    violations = (
        SafetyViolation(
            ViolationType.CLEARANCE,
            "descend",
            "unsafe",
            sample_index=4,
            link="finger/object",
            actual=0.004,
            limit=0.005,
        ),
        SafetyViolation(
            ViolationType.CLEARANCE,
            "descend",
            "unsafe",
            sample_index=5,
            link="finger/object",
            actual=0.003,
            limit=0.005,
        ),
        SafetyViolation(ViolationType.ACCELERATION, "program", "unsafe", actual=13.0, limit=12.0),
    )
    grasp_place._print_preflight_violation_summary(SimpleNamespace(violations=violations), None)
    err = capsys.readouterr().err
    assert "total=3 groups=2" in err
    assert "finger/object count=2 samples=4-5 min_distance_m=0.003 threshold_m=0.005" in err
    assert "rerun with --report PATH" in err


def test_preparation_progress_logs_include_samples_margin_and_timings(capsys):
    segment = SimpleNamespace(
        kind="movej",
        q_start=(0.0,) * 6,
        samples=lambda _rate: (None, 2),
    )
    program = SimpleNamespace(rate=50.0, segments=(segment,))
    config = SimpleNamespace(safety=SimpleNamespace(min_collision_distance_m=0.005))
    preflight = SimpleNamespace(
        passed=True,
        checks=(
            PreflightCheck("collision", True, 3, 625.0),
            PreflightCheck("total", True, 3, 675.0),
        ),
    )

    grasp_place._print_ik_compile_complete(program, elapsed_s=7.25)
    grasp_place._print_preflight_start(program, config)
    grasp_place._print_preflight_complete(preflight)

    assert capsys.readouterr().out.splitlines() == [
        "[ik-compile] complete samples=3 elapsed_s=7.25",
        "[preflight] checking samples=3 collision_margin_mm=5.0...",
        "[preflight] complete status=PASS collision_s=0.625 total_s=0.675",
    ]


def test_kinematic_mirror_disables_existing_viewer_scene_pacer():
    viewer = SimpleNamespace(realtime_factor=1.0)
    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))

    start_deferred_viewer(scene, kinematic_mirror=True)

    assert viewer.realtime_factor is None


def test_hold_viewer_runs_until_window_closes_and_paces_mirror(monkeypatch, capsys):
    alive_checks = 0
    step_calls = 0
    sleeps: list[float] = []

    class Viewer:
        def is_alive(self) -> bool:
            nonlocal alive_checks
            alive_checks += 1
            return alive_checks <= 3

    def on_step() -> None:
        nonlocal step_calls
        step_calls += 1

    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=Viewer()))
    monkeypatch.setattr(grasp_place.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(grasp_place.time, "sleep", lambda seconds: sleeps.append(seconds))

    grasp_place._hold_viewer(SimpleNamespace(scene=scene), on_step=on_step)

    assert step_calls == 3
    assert sleeps == pytest.approx([1.0 / 30.0] * 3)
    assert "Viewer open. Press Ctrl+C to exit..." in capsys.readouterr().out


def test_mirror_visual_update_drops_frame_when_render_lock_is_busy():
    class BusyLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return False

    context_updates: list[bool] = []
    pyrender_viewer = SimpleNamespace(render_lock=BusyLock(), update_on_sim_step=lambda: None)
    viewer = SimpleNamespace(
        _pyrender_viewer=pyrender_viewer,
        context=SimpleNamespace(update=lambda **kwargs: context_updates.append(kwargs["force_render"])),
    )
    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))

    assert update_scene_visualizer(scene) is False
    assert context_updates == []
