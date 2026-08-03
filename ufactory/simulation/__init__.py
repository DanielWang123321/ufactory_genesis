"""Genesis runtime lifecycle and scene factories."""

from ufactory.simulation.physics import (
    DEFAULT_CONSTRAINT_SOLVER,
    DEFAULT_CONSTRAINT_TIMECONST,
    DEFAULT_CONTACT_RESOLUTION,
    DEFAULT_FRICTION_CONE,
    DEFAULT_NOSLIP_ITERATIONS,
    DEFAULT_SOLVER_ITERATIONS,
    make_rigid_options,
    validate_rigid_physics,
)
from ufactory.simulation.runtime import (
    BACKEND_INIT_HINT,
    GenesisRuntimeError,
    GenesisRuntimeManager,
    genesis_backend_constant,
    override_simulation_backend,
)

__all__ = [
    "BACKEND_INIT_HINT",
    "DEFAULT_CONSTRAINT_SOLVER",
    "DEFAULT_CONSTRAINT_TIMECONST",
    "DEFAULT_CONTACT_RESOLUTION",
    "DEFAULT_FRICTION_CONE",
    "DEFAULT_NOSLIP_ITERATIONS",
    "DEFAULT_SOLVER_ITERATIONS",
    "GenesisRuntimeError",
    "GenesisRuntimeManager",
    "genesis_backend_constant",
    "make_rigid_options",
    "override_simulation_backend",
    "validate_rigid_physics",
]
