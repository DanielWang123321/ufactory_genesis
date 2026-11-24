"""Unit tests for Robotiq mimic constraints, substep guards, and collision wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ufactory.grippers.robotiq import get_robotiq_spec  # noqa: E402
from ufactory.simulation.robotiq_adapter import (  # noqa: E402
    ROBOTIQ_MIMIC_CONSTRAINT_SOL_PARAMS,
    RobotiqContactCollisionBackend,
    configure_robotiq_drive,
    configure_robotiq_mimic_constraints,
    robotiq_mimic_equality_names,
    validate_robotiq_contact_substeps,
)


@dataclass
class _FakeEquality:
    name: str
    params: np.ndarray | None = None

    def set_sol_params(self, values: np.ndarray) -> None:
        self.params = np.asarray(values, dtype=np.float64)


class _FakeJoint:
    def __init__(self, idx: int) -> None:
        self.dofs_idx_local = [idx]


class _FakeRobot:
    def __init__(self, variant: str, *, drop_equality: bool = False) -> None:
        spec = get_robotiq_spec(variant)
        names = sorted(robotiq_mimic_equality_names(variant))
        if drop_equality:
            names = names[:-1]
        self.equalities = [_FakeEquality(n) for n in names]
        self._joints = {name: _FakeJoint(i) for i, name in enumerate(spec.joint_names)}
        self.force_calls: list = []
        self.kp_calls: list = []
        self.kv_calls: list = []

    def get_joint(self, name: str) -> _FakeJoint:
        return self._joints[name]

    def get_dofs_position(self, idx: list[int]) -> torch.Tensor:
        return torch.zeros((1, len(idx)), dtype=torch.float64)

    def control_dofs_force(self, vals: torch.Tensor, idx: list[int]) -> None:
        self.force_calls.append((vals, idx))

    def set_dofs_kp(self, vals: torch.Tensor, idx: list[int]) -> None:
        self.kp_calls.append((vals, idx))

    def set_dofs_kv(self, vals: torch.Tensor, idx: list[int]) -> None:
        self.kv_calls.append((vals, idx))

    def set_dofs_armature(self, vals: torch.Tensor, idx: list[int]) -> None:
        del vals, idx

    def set_dofs_damping(self, vals: torch.Tensor, idx: list[int]) -> None:
        del vals, idx

    def set_dofs_frictionloss(self, vals: torch.Tensor, idx: list[int]) -> None:
        del vals, idx


def test_substep_guard() -> None:
    assert validate_robotiq_contact_substeps(dt=0.02, substeps=32, variant="robotiq_2f_85") == pytest.approx(0.000625)
    with pytest.raises(ValueError, match="Use at least 32 substeps"):
        validate_robotiq_contact_substeps(dt=0.02, substeps=16, variant="robotiq_2f_140")
    with pytest.raises(ValueError):
        validate_robotiq_contact_substeps(dt=0.0, substeps=32)


@pytest.mark.parametrize("variant", ["robotiq_2f_85", "robotiq_2f_140"])
def test_mimic_constraints_and_drive_setup(variant: str) -> None:
    robot = _FakeRobot(variant)
    assert configure_robotiq_mimic_constraints(robot, variant) == 5
    for eq in robot.equalities:
        np.testing.assert_allclose(eq.params, ROBOTIQ_MIMIC_CONSTRAINT_SOL_PARAMS)

    incomplete = _FakeRobot(variant, drop_equality=True)
    with pytest.raises(RuntimeError, match="missing Robotiq mimic equalities"):
        configure_robotiq_mimic_constraints(incomplete, variant)

    drive, all_idx = configure_robotiq_drive(robot, variant)
    assert drive == [0]
    assert all_idx == list(range(6))
    assert len(robot.force_calls[0][1]) == 5


@pytest.mark.parametrize(("variant", "open_gap_m"), [("robotiq_2f_85", 0.085), ("robotiq_2f_140", 0.140)])
def test_contact_collision_backend_clamps_gap_to_object_width(variant: str, open_gap_m: float) -> None:
    inner = MagicMock()
    backend = RobotiqContactCollisionBackend(inner, variant=variant, object_width_m=0.030)
    spec = get_robotiq_spec(variant)
    q = np.zeros(6)
    backend.check(q, stage="grip", gripper_drive=spec.gap_m_to_sim_drive(0.010))
    passed_drive = inner.check.call_args.kwargs["gripper_drive"]
    assert passed_drive == pytest.approx(spec.gap_m_to_sim_drive(0.030), abs=1e-6)

    with pytest.raises(ValueError):
        RobotiqContactCollisionBackend(inner, variant=variant, object_width_m=open_gap_m)
