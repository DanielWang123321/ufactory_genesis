"""Shared rigid-contact policy for every Genesis scene in the project.

The public runtime configuration stores plain strings so configuration parsing
does not require Genesis to be installed.  This module resolves those strings
to Genesis enums only when a scene is actually constructed.
"""

from __future__ import annotations

from typing import Any


DEFAULT_CONSTRAINT_SOLVER = "newton"
DEFAULT_FRICTION_CONE = "pyramidal"
DEFAULT_CONTACT_RESOLUTION = "convex"
DEFAULT_SOLVER_ITERATIONS = 100
DEFAULT_NOSLIP_ITERATIONS = 0
DEFAULT_CONSTRAINT_TIMECONST = 0.005

CONSTRAINT_SOLVERS = (DEFAULT_CONSTRAINT_SOLVER,)
FRICTION_CONES = ("pyramidal", "elliptic")
CONTACT_RESOLUTIONS = ("convex", "signorini")


def validate_rigid_physics(
    *,
    constraint_solver: str,
    friction_cone: str,
    contact_resolution: str,
    noslip_iterations: int,
) -> None:
    """Reject unsupported or internally inconsistent rigid-contact settings."""

    if constraint_solver not in CONSTRAINT_SOLVERS:
        raise ValueError(f"constraint_solver must be one of {CONSTRAINT_SOLVERS}")
    if friction_cone not in FRICTION_CONES:
        raise ValueError(f"friction_cone must be one of {FRICTION_CONES}")
    if contact_resolution not in CONTACT_RESOLUTIONS:
        raise ValueError(f"contact_resolution must be one of {CONTACT_RESOLUTIONS}")
    if noslip_iterations < 0:
        raise ValueError("noslip_iterations must be non-negative")
    if friction_cone == "elliptic" and noslip_iterations:
        raise ValueError("elliptic friction is incompatible with the noslip solver")
    if contact_resolution == "signorini" and (
        constraint_solver != "newton" or friction_cone != "elliptic" or noslip_iterations != 0
    ):
        raise ValueError("signorini contact requires Newton, elliptic friction, and noslip_iterations=0")


def make_rigid_options(
    gs_module: Any,
    *,
    constraint_solver: str = DEFAULT_CONSTRAINT_SOLVER,
    friction_cone: str = DEFAULT_FRICTION_CONE,
    contact_resolution: str = DEFAULT_CONTACT_RESOLUTION,
    iterations: int = DEFAULT_SOLVER_ITERATIONS,
    noslip_iterations: int = DEFAULT_NOSLIP_ITERATIONS,
    constraint_timeconst: float = DEFAULT_CONSTRAINT_TIMECONST,
    **options: Any,
) -> Any:
    """Create ``gs.options.RigidOptions`` from the project's named policy."""

    constraint_solver = str(constraint_solver).lower()
    friction_cone = str(friction_cone).lower()
    contact_resolution = str(contact_resolution).lower()
    iterations = int(iterations)
    noslip_iterations = int(noslip_iterations)
    constraint_timeconst = float(constraint_timeconst)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if constraint_timeconst <= 0.0:
        raise ValueError("constraint_timeconst must be positive")
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
        **options,
    )


__all__ = [
    "CONSTRAINT_SOLVERS",
    "CONTACT_RESOLUTIONS",
    "DEFAULT_CONSTRAINT_SOLVER",
    "DEFAULT_CONSTRAINT_TIMECONST",
    "DEFAULT_CONTACT_RESOLUTION",
    "DEFAULT_FRICTION_CONE",
    "DEFAULT_NOSLIP_ITERATIONS",
    "DEFAULT_SOLVER_ITERATIONS",
    "FRICTION_CONES",
    "make_rigid_options",
    "validate_rigid_physics",
]
