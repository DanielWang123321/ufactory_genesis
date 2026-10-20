"""Counterexamples for geometric release, touchdown and stable table support."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from examples.rl.pick_place.metrics import PickPlaceMetrics, cube_clearance, grasp_distance, SUCCESS_HOLD_STEPS


def sample(m, *, x=0.3, y=0.3, height=0.0, forces=(0.5, 0.5), ee=0.19, support=0.17, gap=0.03, speed=0.0, robot=0.0):
    n = m.n
    return m.update(
        torch.tensor([[x, y, height + 0.015]]).repeat(n, 1),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        torch.tensor([forces]).repeat(n, 1),
        torch.tensor([[0.0, 0.0, speed]]).repeat(n, 1),
        torch.full((n,), gap),
        ee_z=torch.full((n,), ee),
        table_force=torch.full((n,), support),
        robot_table_force=torch.full((n,), robot),
    )


def lifted(n=1):
    m = PickPlaceMetrics(n, "cpu")
    for _ in range(3):
        sample(m, y=0.0, ee=0.18)
    s = sample(m, y=0.0, height=0.085, ee=0.265, support=0.0)
    assert s["first_lift_clear"].all()
    assert m.ever_lifted_clear.all()
    return m


def settle(m, steps=15, **kwargs):
    for _ in range(steps):
        s = sample(m, forces=(0.0, 0.0), gap=0.05, **kwargs)
    return s


def test_ideal_release_above_table_and_supported_hold():
    m = lifted()
    sample(m, height=0.002, ee=0.182, support=0.0)
    sample(m, height=0.002, ee=0.182, support=0.0, forces=(0.0, 0.0), gap=0.05)
    for i in range(SUCCESS_HOLD_STEPS + 1):
        s = sample(m, forces=(0.0, 0.0), gap=0.05, ee=0.182)
        assert bool(s["success"]) == (i == SUCCESS_HOLD_STEPS)
    assert m.release_height.item() == pytest.approx(0.002)
    assert m.max_press.item() == 0


def test_gentle_release_after_touchdown_includes_previous_sample():
    m = lifted()
    sample(m, height=0.0003, ee=0.1810, support=0.0)
    sample(m, ee=0.1807)
    assert m.max_press.item() == pytest.approx(0.0003, abs=1e-7)
    s = settle(m, ee=0.1805)
    assert s["success"].item()
    assert m.max_press.item() == pytest.approx(0.0005, abs=1e-7)


def test_one_finger_keeps_press_tracking_until_detachment():
    m = lifted()
    sample(m, height=0.001, ee=0.182, support=0.0)
    sample(m, ee=0.1815)
    s = sample(m, ee=0.1805, forces=(0.0, 0.3), gap=0.05)
    assert s["press_failed"].item()
    assert not s["released"].item()
    assert m.max_press.item() == pytest.approx(0.0015, abs=1e-7)
    assert not settle(m, ee=0.1805)["success"].item()


def test_contact_jitter_does_not_reset_touchdown_or_confirm_release():
    m = lifted()
    sample(m, height=0.001, ee=0.182, support=0.0)
    sample(m, ee=0.1818)
    sample(m, ee=0.1815, forces=(0.0, 0.0), gap=0.05)
    sample(m, ee=0.1812, forces=(0.3, 0.0), gap=0.05)
    assert not m.release_confirmed.item()
    sample(m, ee=0.1808, support=0.0)
    assert m.max_press.item() == pytest.approx(0.0012, abs=1e-7)


def test_force_hysteresis_and_consecutive_grasp_confirmation():
    m = PickPlaceMetrics(1, "cpu")
    sample(m, y=0.0, forces=(0.3, 0.3))
    sample(m, y=0.0, forces=(0.1, 0.1))
    s = sample(m, y=0.0, forces=(0.1, 0.1))
    assert s["stable_contact"].item()
    sample(m, y=0.0, forces=(0.01, 0.3))
    assert m.contact_steps.item() == 0


def test_high_drop_cannot_become_success_after_landing():
    m = lifted()
    sample(m, height=0.05, ee=0.23, support=0.0)
    s = settle(m, ee=0.23)
    assert s["high_release"].item()
    assert m.release_height.item() == pytest.approx(0.05)
    assert not s["success"].item()


def test_high_release_uses_first_clear_sample_before_falling():
    m = lifted()
    sample(m, height=0.02, ee=0.20, support=0.0, forces=(0.0, 0.0), gap=0.05)
    s = settle(m, ee=0.20)
    assert s["high_release"].item()
    assert not s["success"].item()


def test_table_force_is_required_for_success():
    m = lifted()
    sample(m, height=0.002, ee=0.182, support=0.0)
    s = settle(m, support=0.0, ee=0.182)
    assert m.release_confirmed.item()
    assert not s["success"].item()
    assert m.hold_steps.item() == 0


def test_support_jitter_restarts_hold():
    m = lifted()
    sample(m, height=0.002, ee=0.182, support=0.0)
    settle(m, steps=10, ee=0.182)
    sample(m, support=0.0, ee=0.182, forces=(0.0, 0.0), gap=0.05)
    assert m.hold_steps.item() == 0
    assert not settle(m, steps=9, ee=0.182)["success"].item()
    assert settle(m, steps=1, ee=0.182)["success"].item()


def test_release_height_upper_boundary():
    m = lifted()
    sample(m, height=0.0031, ee=0.1831, support=0.0)
    assert not settle(m, ee=0.1831)["success"].item()
    assert m.high_release.item()


def test_release_at_exactly_three_mm_is_valid():
    m = lifted()
    sample(m, height=0.003, ee=0.183, support=0.0)
    assert settle(m, ee=0.183)["success"].item()


def test_post_release_arm_motion_does_not_inflate_press():
    m = lifted()
    sample(m, height=0.002, ee=0.182, support=0.0)
    settle(m, steps=4, ee=0.182)
    settle(m, steps=2, ee=0.17)
    assert m.max_press.item() == 0


def test_recontact_then_high_drop_cannot_reuse_previous_valid_release():
    m = lifted()
    sample(m, height=0.002, ee=0.182, support=0.0)
    settle(m, steps=4, ee=0.182)
    assert m.release_confirmed.item()
    sample(m, height=0.04, ee=0.22, support=0.0)
    s = settle(m, ee=0.22)
    assert s["high_release"].item()
    assert not s["success"].item()


def test_recontact_starts_another_press_interval():
    m = lifted()
    sample(m, height=0.002, ee=0.182, support=0.0)
    settle(m, steps=4, ee=0.182)
    sample(m, ee=0.1818)
    s = sample(m, ee=0.1805, forces=(0.5, 0.0))
    assert s["press_failed"].item()
    assert m.max_press.item() == pytest.approx(0.0015, abs=1e-7)


def test_batch_reset_does_not_change_other_histories():
    m = lifted(4)
    sample(m, height=0.001, ee=0.182, support=0.0)
    sample(m, ee=0.1815)
    tensors = {name: value[1:].clone() for name, value in vars(m).items() if isinstance(value, torch.Tensor)}
    m.reset(torch.tensor([0]), torch.tensor([[0.3, 0.0]]), torch.tensor([0.3]))
    for name, expected in tensors.items():
        torch.testing.assert_close(getattr(m, name)[1:], expected, rtol=0, atol=0)
    assert not m.ever_carried[0]
    assert m.max_press[0] == 0
    assert m.prev_ee_z[0] == pytest.approx(0.3)


def test_tilted_cube_clearance_uses_lowest_corner():
    q = torch.tensor([[math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0]])
    clearance = cube_clearance(torch.tensor([[0.3, 0.3, 0.023]]), q)
    assert clearance.item() == pytest.approx(0.023 - 0.015 * math.sqrt(2), abs=1e-7)


def test_open_pad_approach_does_not_require_center_height_alignment():
    obj = torch.tensor([[0.3, 0.0, 0.015]])
    high = grasp_distance(torch.tensor([[0.3, 0.0, 0.07]]), obj)
    band = grasp_distance(torch.tensor([[0.3, 0.0, 0.027]]), obj)
    center = grasp_distance(obj.clone(), obj)
    assert high > band
    torch.testing.assert_close(band, center, atol=1e-8, rtol=0)


def test_pre_lift_drag_and_tilt_are_terminal_failures():
    m = PickPlaceMetrics(1, "cpu")
    assert sample(m, x=0.315, y=0.0)["drag_failed"].item()
    m.reset(0)
    q = torch.tensor([[math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0, 0.0]])
    s = m.update(
        torch.tensor([[0.3, 0.0, 0.022]]),
        q,
        torch.zeros(1, 2),
        torch.zeros(1, 3),
        torch.tensor([0.05]),
        ee_z=torch.tensor([0.2]),
        table_force=torch.tensor([0.17]),
        robot_table_force=torch.zeros(1),
    )
    assert s["tilt_failed"].item()


def test_hold_duration_never_less_than_point_two_seconds():
    m = PickPlaceMetrics(1, "cpu", ctrl_dt=0.01, hold_steps=5)
    assert m.required_hold_steps == 20


@pytest.mark.parametrize("case", ["spawn", "pushed", "held", "off_target", "moving", "collision"])
def test_reject_original_failure_cases(case):
    m = PickPlaceMetrics(1, "cpu") if case in {"spawn", "pushed"} else lifted()
    if case not in {"spawn", "pushed"}:
        sample(m, height=0.002, ee=0.182, support=0.0)
    kwargs = {"ee": 0.182}
    if case == "spawn":
        kwargs["y"] = 0.0
    if case == "off_target":
        kwargs["y"] = 0.265
    if case == "moving":
        kwargs["speed"] = 0.05
    if case == "collision":
        kwargs["robot"] = 0.3
    if case == "held":
        for _ in range(15):
            s = sample(m, **kwargs)
    else:
        s = settle(m, **kwargs)
    assert not s["success"].item()
