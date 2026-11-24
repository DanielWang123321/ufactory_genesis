"""Unit tests for the batched Torch Robotiq 2F-85 and 2F-140 controller."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ufactory.simulation.robotiq_controller import RobotiqController  # noqa: E402
from ufactory.simulation.robotiq_model import RobotiqModel  # noqa: E402


@pytest.mark.parametrize(("variant", "open_mm"), [("robotiq_2f_85", 85.0), ("robotiq_2f_140", 140.0)])
def test_robotiq_controller_stroke_reversal_and_pause_preload(variant: str, open_mm: float) -> None:
    model = RobotiqModel.load(variant)
    ctrl = RobotiqController(model, num_envs=2)
    q_open = float(ctrl.drive_for_gap_mm(torch.tensor(open_mm)))
    ctrl.command([20.0, open_mm], speed_percent=[100.0, 50.0], force_percent=[50.0, 80.0])
    out = ctrl.update([q_open, q_open], [0.0, 0.0], dt=0.02)
    assert out.reference_gap_mm[0].item() < open_mm
    assert out.reference_gap_mm[1].item() == pytest.approx(open_mm)
    assert out.torque_limit_nm[1].item() > out.torque_limit_nm[0].item()

    # Pause on a stalled grasp preserves nonzero preload sign.
    q_hold = float(ctrl.drive_for_gap_mm(torch.tensor(30.0)))
    ctrl.reset(envs_idx=[0], position_mm=25.0, target_mm=15.0, prime=True)
    ctrl.command([15.0, open_mm], speed_percent=[50.0, 50.0], force_percent=[40.0, 50.0])
    ctrl.update([q_hold, q_open], [0.0, 0.0], dt=0.02)
    ctrl.command([15.0, open_mm], speed_percent=[0.0, 50.0], force_percent=[90.0, 50.0])
    paused = ctrl.update([q_hold, q_open], [0.0, 0.0], dt=0.02)
    assert paused.velocity_rad_s[0].item() == pytest.approx(0.0)
    assert paused.integral_torque_nm[0].abs().item() == pytest.approx(paused.torque_limit_nm[0].item())

    # Reversing resets integral windup.
    ctrl.command([open_mm, open_mm], speed_percent=[50.0, 50.0], force_percent=[50.0, 50.0])
    reversed_out = ctrl.update([q_hold, q_open], [0.0, 0.0], dt=0.02)
    assert reversed_out.reference_gap_mm[0].item() >= 30.0


def test_robotiq_controller_bounds() -> None:
    model85 = RobotiqModel.load("robotiq_2f_85")
    ctrl85 = RobotiqController(model85, num_envs=1)
    with pytest.raises(ValueError):
        ctrl85.command(100.0)
    with pytest.raises(ValueError):
        ctrl85.command(20.0, force_percent=0.0)
