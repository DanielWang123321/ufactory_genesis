"""Robotiq grippers reach a real controller only in combinations validated on hardware."""

from __future__ import annotations

import importlib

import pytest

from ufactory.grippers.robotiq import ROBOTIQ_CONTROLLER_VALIDATED, robotiq_controller_mode_error


class _ReachedConnect(Exception):
    pass


def _run(cli: str, robot: str, gripper: str, mode: str, monkeypatch) -> None:
    if cli == "pick_place":
        # The pick-place CLI imports the Genesis IK stack at module level; public CI installs neither.
        pytest.importorskip("torch")
        pytest.importorskip("genesis")
    module = importlib.import_module(f"ufactory.cli.{cli}")

    def connect(ip: str):
        raise _ReachedConnect(ip)

    monkeypatch.setattr(module, "_connect", connect)
    argv = ["--robot", robot, "--gripper", gripper, "--mode", mode, "--executor", "servo_j"]
    argv += ["--ip", "192.0.2.1", "--calibration", "unused.yaml", "--confirm-real"]
    module.main(argv)


def test_only_pick_place_on_xarm6_with_2f85_is_validated() -> None:
    assert ROBOTIQ_CONTROLLER_VALIDATED == frozenset({("pick_place", "xarm6", "robotiq_2f_85")})
    for mode in ("sim", "dry-run"):
        assert robotiq_controller_mode_error("packaging", "uf850", "robotiq_2f_140", mode) is None
    assert robotiq_controller_mode_error("pick_place", "xarm6", "g2", "real") is None


@pytest.mark.parametrize(
    ("cli", "robot", "gripper"),
    [
        ("pick_place", "xarm6", "robotiq_2f_140"),
        ("pick_place", "xarm7", "robotiq_2f_85"),
        ("packaging", "xarm6", "robotiq_2f_85"),
        ("packaging", "xarm6", "robotiq_2f_140"),
    ],
)
@pytest.mark.parametrize("mode", ["sdk-sim", "real"])
def test_unvalidated_robotiq_controller_modes_are_refused_before_connecting(
    cli, robot, gripper, mode, monkeypatch, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(cli, robot, gripper, mode, monkeypatch)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"--gripper {gripper} currently requires --mode sim or dry-run for {cli} on {robot}" in err
    assert "pick_place on xarm6 with robotiq_2f_85" in err


@pytest.mark.parametrize("mode", ["sdk-sim", "real"])
def test_validated_combination_passes_the_gate(mode, monkeypatch) -> None:
    # Argument checks pass and the CLI goes on to connect (stubbed here).
    with pytest.raises(_ReachedConnect):
        _run("pick_place", "xarm6", "robotiq_2f_85", mode, monkeypatch)
