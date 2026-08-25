from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ufactory.simulation.g2 import (
    G2_CONTACT_HOLD_POLICY,
    G2_MAX_RIGID_SUBSTEP_DT_S,
    G2_MIMIC_CONSTRAINT_SOL_PARAMS,
    G2_MIMIC_EQUALITY_NAMES,
    G2_PHYSICS_PROFILE,
    G2ContactHoldController,
    configure_g2_mimic_constraints,
    object_finger_contact_forces_n,
    validate_g2_contact_substeps,
)


class _Equality:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sol_params: np.ndarray | None = None

    def set_sol_params(self, value: np.ndarray) -> None:
        self.sol_params = value.copy()


def test_g2_contact_substep_boundary_accepts_32_at_50_hz() -> None:
    assert validate_g2_contact_substeps(dt=0.02, substeps=32) == pytest.approx(G2_MAX_RIGID_SUBSTEP_DT_S)


def test_g2_contact_substep_guard_rejects_31_at_50_hz() -> None:
    with pytest.raises(ValueError, match=rf"{G2_PHYSICS_PROFILE}.*0\.625 ms.*substeps=31.*at least 32"):
        validate_g2_contact_substeps(dt=0.02, substeps=31)


@pytest.mark.parametrize(
    ("dt", "substeps"),
    [(0.0, 32), (float("nan"), 32), (0.02, 0), (0.02, 31.5), (0.02, 32.0), (0.02, "32")],
)
def test_g2_contact_substep_guard_rejects_invalid_values(dt: float, substeps: object) -> None:
    with pytest.raises(ValueError):
        validate_g2_contact_substeps(dt=dt, substeps=substeps)


def test_shared_g2_mimic_policy_configures_all_five_linkage_equalities() -> None:
    equalities = [_Equality(name) for name in sorted(G2_MIMIC_EQUALITY_NAMES)] + [_Equality("unrelated")]

    assert configure_g2_mimic_constraints(SimpleNamespace(equalities=equalities)) == 5
    expected = np.asarray(G2_MIMIC_CONSTRAINT_SOL_PARAMS, dtype=np.float64)
    np.testing.assert_array_equal(G2_MIMIC_CONSTRAINT_SOL_PARAMS, (0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0))
    for equality in equalities[:-1]:
        np.testing.assert_array_equal(equality.sol_params, expected)
    assert equalities[-1].sol_params is None


def test_shared_g2_mimic_policy_rejects_a_partial_linkage() -> None:
    equalities = [_Equality(name) for name in sorted(G2_MIMIC_EQUALITY_NAMES)[:-1]]

    with pytest.raises(RuntimeError, match="missing G2 mimic equalities"):
        configure_g2_mimic_constraints(SimpleNamespace(equalities=equalities))


def test_object_finger_contact_forces_filter_links_and_padding() -> None:
    contacts = {
        "link_a": torch.tensor([[10, 20, 12, 10], [10, -1, -1, -1]]),
        "link_b": torch.tensor([[20, 11, 20, 20], [20, -1, -1, -1]]),
        "force_a": torch.tensor(
            [
                [[3.0, 4.0, 0.0], [0.0, 0.0, -2.0], [9.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        "force_b": torch.tensor(
            [
                [[-3.0, -4.0, 0.0], [0.0, 0.0, 2.0], [-9.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, -1.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        "valid_mask": torch.tensor([[True, True, True, True], [True, False, False, False]]),
    }

    left, right = object_finger_contact_forces_n(contacts, left_link_idx=10, right_link_idx=11)

    # The two left-pad contacts vector-sum to (0, 4, 0), rather than adding
    # magnitudes to 8 N.  The right pad is deliberately contact side B.
    torch.testing.assert_close(left, torch.tensor([4.0, 1.5]))
    torch.testing.assert_close(right, torch.tensor([2.0, 0.0]))


def test_g2_contact_hold_latches_regulates_and_releases() -> None:
    controller = G2ContactHoldController(
        1,
        device="cpu",
        dtype=torch.float32,
        initial_gap_m=0.084,
    )

    def update(*, requested: float, measured: float, left: float, right: float, release: bool = False):
        return controller.update(
            requested_gap_m=torch.tensor([requested]),
            measured_gap_m=torch.tensor([measured]),
            left_force_n=torch.tensor([left]),
            right_force_n=torch.tensor([right]),
            closing=torch.tensor([not release]),
            release=torch.tensor([release]),
        )[0].item()

    assert update(requested=0.030, measured=0.040, left=0.0, right=0.0) == pytest.approx(0.030)
    update(requested=0.020, measured=0.030, left=1.0, right=1.0)
    held = update(requested=0.020, measured=0.029, left=1.0, right=1.0)
    assert controller.latched.item() is True
    assert held == pytest.approx(0.029 - G2_CONTACT_HOLD_POLICY.latch_window_m)

    relieved = update(requested=0.020, measured=0.029, left=14.0, right=14.0)
    assert relieved == pytest.approx(held + G2_CONTACT_HOLD_POLICY.max_gap_step_m)

    opened = update(requested=0.084, measured=0.029, left=14.0, right=14.0, release=True)
    assert opened == pytest.approx(0.084)
    assert controller.latched.item() is False


def test_g2_contact_hold_reacquires_only_after_confirmed_loss() -> None:
    controller = G2ContactHoldController(1, device="cpu", dtype=torch.float32, initial_gap_m=0.030)
    controller.prime(torch.tensor([True]), torch.tensor([0.030]))
    zero = torch.zeros(1)
    kwargs = {
        "requested_gap_m": torch.tensor([0.020]),
        "measured_gap_m": torch.tensor([0.030]),
        "left_force_n": zero,
        "right_force_n": zero,
        "closing": torch.tensor([True]),
        "release": torch.tensor([False]),
    }

    for _ in range(G2_CONTACT_HOLD_POLICY.contact_loss_steps - 1):
        assert controller.update(**kwargs).item() == pytest.approx(0.030)
    assert controller.update(**kwargs).item() == pytest.approx(0.030 - G2_CONTACT_HOLD_POLICY.max_gap_step_m)
