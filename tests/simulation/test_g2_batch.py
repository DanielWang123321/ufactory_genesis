import pytest

torch = pytest.importorskip("torch")

from ufactory.simulation.g2_batch import G2BatchController
from ufactory.simulation.g2_controller import G2Controller
from ufactory.simulation.g2_model import G2Model
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive, gripper_g2_sim_drive_to_gap_m
from ufactory.simulation.g2_model import REFERENCE_GAP_MM
from ufactory.training.logic.pick_place import gap_m_to_drive, drive_to_gap_m


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("trimesh")
    return G2Model.load()


@pytest.mark.parametrize("dtype,tolerance_mm", [(torch.float64, 1e-9), (torch.float32, 2e-5)])
def test_tensor_geometry_matches_scalar_and_rl_conversions(model, dtype, tolerance_mm):
    gaps = torch.tensor([0, 10, REFERENCE_GAP_MM, 30, 60, 84], dtype=dtype)
    batch = G2BatchController(model, len(gaps), dtype=dtype)
    expected = torch.tensor([gripper_g2_gap_m_to_sim_drive(g / 1000) for g in gaps.tolist()], dtype=dtype)
    q = batch.drive_for_gap_mm(gaps)
    torch.testing.assert_close(q, expected, rtol=0, atol=tolerance_mm / 80)
    torch.testing.assert_close(batch.gap_mm(q), gaps, rtol=0, atol=tolerance_mm)
    torch.testing.assert_close(gap_m_to_drive(gaps / 1000, 0, 0.084), q, rtol=0, atol=0)
    restored = drive_to_gap_m(q, 0.084)
    scalar_gaps = torch.tensor([gripper_g2_sim_drive_to_gap_m(float(a)) for a in q], dtype=dtype)
    torch.testing.assert_close(restored, scalar_gaps, rtol=0, atol=tolerance_mm / 1000)
    assert bool((q[:-1] > q[1:]).all())


def test_contact_preflight_preserves_actual_angles_and_object_envelope():
    from unittest.mock import MagicMock

    from ufactory.simulation.g2_adapter import G2ContactCollisionBackend

    backend = MagicMock()
    collision = G2ContactCollisionBackend(backend, object_width_m=0.030)
    for gap in (0, 0.010, 0.02415587200800565, 0.030, 0.060, 0.084):
        drive = gripper_g2_gap_m_to_sim_drive(gap)
        expected = gripper_g2_gap_m_to_sim_drive(max(gap, 0.030))
        for method in ("check", "check_all", "check_all_within_margin"):
            kwargs = {"stage": "grip", "gripper_drive": drive}
            if method == "check_all_within_margin":
                kwargs["margin_m"] = 0.005
            getattr(collision, method)([0] * 6, **kwargs)
            assert getattr(backend, method).call_args.kwargs["gripper_drive"] == pytest.approx(expected, abs=1e-14)
    collision.check([0] * 6, stage="approach")
    assert backend.check.call_args.kwargs["gripper_drive"] == pytest.approx(gripper_g2_gap_m_to_sim_drive(0.084))


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 2e-4)])
def test_batch_matches_scalar_pause_resume_reverse_and_force_changes(model, dtype, tolerance):
    batch = G2BatchController(model, 4, dtype=dtype)
    scalar = [G2Controller(model) for _ in range(4)]
    q = torch.tensor([model.linkage.drive_for_gap_m(g / 1000) for g in (24, 30, 50, 70)], dtype=dtype)
    velocity = torch.tensor([0, 1, -0.5, 0.01], dtype=dtype)
    for position, speeds, forces in [
        ([0, 10, 20, 30], [10, 50, 90, 100], [10, 30, 50, 100]),
        ([0, 10, 20, 30], [0, 0, 0, 0], [10, 30, 80, 100]),
        ([84, 84, 0, 0], [0, 50, 90, 100], [80, 30, 50, 10]),
        ([84, 84, 84, 84], [50, 90, 10, 50], [80, 50, 100, 30]),
    ]:
        batch.command(position, speeds, forces)
        for i, c in enumerate(scalar):
            c.command(position[i], speeds[i], forces[i])
        for _ in range(12):
            actual = batch.update(q, velocity, measured_gap_mm=[23, 29, 49, 69])
            expected = [
                c.update(float(q[i]), float(velocity[i]), measured_gap_mm=[23, 29, 49, 69][i])
                for i, c in enumerate(scalar)
            ]
            for field in (
                "position_rad",
                "velocity_rad_s",
                "reference_gap_mm",
                "integral_torque_nm",
                "torque_limit_nm",
            ):
                assert getattr(actual, field).tolist() == pytest.approx(
                    [getattr(out, field) for out in expected], abs=tolerance
                )


def test_reset_is_independent_and_preheld_reset_seeds_torque(model):
    batch = G2BatchController(model, 4, dtype=torch.float64)
    batch.command(0, [10, 50, 90, 100], [10, 30, 50, 100])
    batch.update(torch.zeros(4), torch.zeros(4))
    untouched = {
        key: value[[1, 3]].clone()
        for key, value in vars(batch).items()
        if isinstance(value, torch.Tensor) and value.shape == (4,)
    }
    batch.reset([0, 2], position_mm=[30, 40], target_mm=[0, 10], prime=True)
    for key, value in untouched.items():
        torch.testing.assert_close(getattr(batch, key)[[1, 3]], value, rtol=0, atol=0)
    assert torch.equal(batch.integral_torque_nm[[0, 2]], batch.torque_limit_nm[[0, 2]])
    batch.reset([2], position_mm=84)
    assert batch.integral_torque_nm[2] == 0
    assert batch.reference_gap_mm[2] == batch.position_mm[2] == 84


def test_independent_torque_limits_and_atomic_invalid_batch_command(model):
    batch = G2BatchController(model, 4)
    batch.command(0, [10, 50, 90, 100], [10, 30, 80, 100])
    expected = [G2Controller(model) for _ in range(4)]
    for i, c in enumerate(expected):
        c.command(0, [10, 50, 90, 100][i], [10, 30, 80, 100][i])
    assert batch.torque_limit_nm.tolist() == pytest.approx(
        [model.torque_limit_nm(c.force_lookup) for c in expected], rel=1e-6
    )
    before = batch.position_mm.clone()
    with pytest.raises(ValueError):
        batch.command([84, 84, 85, 84], 50, 50)
    assert torch.equal(before, batch.position_mm)
