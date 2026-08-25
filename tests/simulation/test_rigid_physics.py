from __future__ import annotations

from types import SimpleNamespace

import pytest

from ufactory.simulation.physics import make_rigid_options, validate_rigid_physics


class _RigidOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_genesis():
    return SimpleNamespace(
        constraint_solver=SimpleNamespace(Newton="NEWTON"),
        friction_cone=SimpleNamespace(pyramidal="PYRAMIDAL", elliptic="ELLIPTIC"),
        contact_resolution=SimpleNamespace(convex="CONVEX", signorini="SIGNORINI"),
        options=SimpleNamespace(RigidOptions=_RigidOptions),
    )


def test_make_rigid_options_resolves_project_defaults() -> None:
    options = make_rigid_options(_fake_genesis())
    assert options.kwargs == {
        "constraint_solver": "NEWTON",
        "friction_cone": "PYRAMIDAL",
        "contact_resolution": "CONVEX",
        "iterations": 100,
        "noslip_iterations": 0,
        "constraint_timeconst": 0.005,
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"iterations": 0}, "iterations"),
        ({"constraint_timeconst": 0.0}, "constraint_timeconst"),
    ],
)
def test_make_rigid_options_rejects_invalid_numeric_defaults(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_rigid_options(_fake_genesis(), **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"constraint_solver": "cg"}, "constraint_solver"),
        ({"friction_cone": "future"}, "friction_cone"),
        ({"contact_resolution": "future"}, "contact_resolution"),
        ({"friction_cone": "elliptic", "noslip_iterations": 5}, "noslip"),
        ({"friction_cone": "pyramidal", "contact_resolution": "signorini"}, "signorini"),
    ],
)
def test_invalid_rigid_physics_is_rejected(kwargs: dict, match: str) -> None:
    values = {
        "constraint_solver": "newton",
        "friction_cone": "elliptic",
        "contact_resolution": "signorini",
        "noslip_iterations": 0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        validate_rigid_physics(**values)
