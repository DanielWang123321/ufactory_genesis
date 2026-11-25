"""Robotiq grippers stay off real controllers until controller TCP/load and SDK commands are validated on hardware."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("cli", ["pick_place", "packaging"])
@pytest.mark.parametrize("gripper", ["robotiq_2f_85", "robotiq_2f_140"])
@pytest.mark.parametrize("mode", ["sdk-sim", "real"])
def test_robotiq_controller_modes_are_refused_before_connecting(cli, gripper, mode, monkeypatch, capsys) -> None:
    module = importlib.import_module(f"ufactory.cli.{cli}")
    monkeypatch.setattr(module, "_connect", lambda ip: pytest.fail("must not connect to a controller"))
    argv = ["--robot", "xarm6", "--gripper", gripper, "--mode", mode]
    argv += ["--ip", "192.0.2.1", "--calibration", "unused.yaml", "--confirm-real"]
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"--gripper {gripper} currently requires --mode sim or dry-run" in err
    assert "11 mm flange adapter" in err
    assert "have not been validated on hardware" in err
