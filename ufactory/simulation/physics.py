"""Shared rigid-contact policy for every Genesis scene in the project.

The public runtime configuration stores plain strings so configuration parsing
does not require Genesis to be installed.  This module resolves those strings
to Genesis enums only when a scene is actually constructed.
"""

from __future__ import annotations

from typing import Any

from ufactory.config.models import (
    CONSTRAINT_SOLVERS,
    CONTACT_RESOLUTIONS,
    FRICTION_CONES,
    validate_rigid_physics,
)

DEFAULT_CONSTRAINT_SOLVER = "newton"
DEFAULT_FRICTION_CONE = "pyramidal"
DEFAULT_CONTACT_RESOLUTION = "convex"
DEFAULT_SOLVER_ITERATIONS = 100
DEFAULT_NOSLIP_ITERATIONS = 0
DEFAULT_CONSTRAINT_TIMECONST = 0.005
DEFAULT_IMPRATIO = 100.0


def make_rigid_options(
    gs_module: Any,
    *,
    constraint_solver: str = DEFAULT_CONSTRAINT_SOLVER,
    friction_cone: str = DEFAULT_FRICTION_CONE,
    contact_resolution: str = DEFAULT_CONTACT_RESOLUTION,
    iterations: int = DEFAULT_SOLVER_ITERATIONS,
    noslip_iterations: int = DEFAULT_NOSLIP_ITERATIONS,
    constraint_timeconst: float = DEFAULT_CONSTRAINT_TIMECONST,
    impratio: float = DEFAULT_IMPRATIO,
    **options: Any,
) -> Any:
    """Create ``gs.options.RigidOptions`` from the project's named policy."""

    options.pop("dt", None)
    constraint_solver = str(constraint_solver).lower()
    friction_cone = str(friction_cone).lower()
    contact_resolution = str(contact_resolution).lower()
    iterations = int(iterations)
    noslip_iterations = int(noslip_iterations)
    constraint_timeconst = float(constraint_timeconst)
    impratio = float(impratio)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if constraint_timeconst <= 0.0:
        raise ValueError("constraint_timeconst must be positive")
    if impratio <= 0.0:
        raise ValueError("impratio must be positive")
    validate_rigid_physics(
        constraint_solver=constraint_solver,
        friction_cone=friction_cone,
        contact_resolution=contact_resolution,
        noslip_iterations=noslip_iterations,
    )
    solver_enum = {"newton": gs_module.constraint_solver.Newton}[constraint_solver]
    cone_enum = {
        "pyramidal": gs_module.friction_cone.pyramidal,
        "elliptic": gs_module.friction_cone.elliptic,
    }[friction_cone]
    resolution_enum = {
        "convex": gs_module.contact_resolution.convex,
        "signorini": gs_module.contact_resolution.signorini,
    }[contact_resolution]
    return gs_module.options.RigidOptions(
        constraint_solver=solver_enum,
        friction_cone=cone_enum,
        contact_resolution=resolution_enum,
        iterations=iterations,
        noslip_iterations=noslip_iterations,
        constraint_timeconst=constraint_timeconst,
        impratio=impratio,
        **options,
    )


__all__ = [
    "CONSTRAINT_SOLVERS",
    "CONTACT_RESOLUTIONS",
    "DEFAULT_CONSTRAINT_SOLVER",
    "DEFAULT_CONSTRAINT_TIMECONST",
    "DEFAULT_CONTACT_RESOLUTION",
    "DEFAULT_FRICTION_CONE",
    "DEFAULT_IMPRATIO",
    "DEFAULT_NOSLIP_ITERATIONS",
    "DEFAULT_SOLVER_ITERATIONS",
    "FRICTION_CONES",
    "make_rigid_options",
    "validate_rigid_physics",
]
