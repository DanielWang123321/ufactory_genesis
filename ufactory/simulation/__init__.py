"""Genesis runtime lifecycle and scene factories."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ufactory.config.models import G2_PHYSICS_PROFILE
from ufactory.simulation.g2_model import G2Model, g2_physics_summary
from ufactory.simulation.g2_controller import G2Controller
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
from ufactory.simulation.compat import (
    forward_kinematics,
)
from ufactory.simulation.runtime import (
    BACKEND_INIT_HINT,
    GenesisRuntimeError,
    GenesisRuntimeManager,
    genesis_backend_constant,
    override_simulation_backend,
)

# Torch-dependent exports are loaded on first access so that importing this
# package (e.g. for ``physics`` or ``runtime``) works without the ``sim`` extra.
_LAZY_EXPORTS = {
    "G2BatchController": "ufactory.simulation.g2_batch",
    "G2_MAX_RIGID_SUBSTEP_DT_S": "ufactory.simulation.g2",
    "G2_MIMIC_CONSTRAINT_SOL_PARAMS": "ufactory.simulation.g2",
    "G2_MIMIC_EQUALITY_NAMES": "ufactory.simulation.g2",
    "configure_g2_mimic_constraints": "ufactory.simulation.g2",
    "object_finger_contact_forces_n": "ufactory.simulation.g2",
    "validate_g2_contact_substeps": "ufactory.simulation.g2",
}

if TYPE_CHECKING:
    from ufactory.simulation.g2 import (
        G2_MAX_RIGID_SUBSTEP_DT_S,
        G2_MIMIC_CONSTRAINT_SOL_PARAMS,
        G2_MIMIC_EQUALITY_NAMES,
        configure_g2_mimic_constraints,
        object_finger_contact_forces_n,
        validate_g2_contact_substeps,
    )
    from ufactory.simulation.g2_batch import G2BatchController


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "BACKEND_INIT_HINT",
    "DEFAULT_CONSTRAINT_SOLVER",
    "DEFAULT_CONSTRAINT_TIMECONST",
    "DEFAULT_CONTACT_RESOLUTION",
    "DEFAULT_FRICTION_CONE",
    "DEFAULT_NOSLIP_ITERATIONS",
    "DEFAULT_SOLVER_ITERATIONS",
    "G2_MAX_RIGID_SUBSTEP_DT_S",
    "G2_MIMIC_CONSTRAINT_SOL_PARAMS",
    "G2_MIMIC_EQUALITY_NAMES",
    "G2_PHYSICS_PROFILE",
    "G2Model",
    "G2Controller",
    "G2BatchController",
    "GenesisRuntimeError",
    "GenesisRuntimeManager",
    "configure_g2_mimic_constraints",
    "g2_physics_summary",
    "forward_kinematics",
    "genesis_backend_constant",
    "make_rigid_options",
    "object_finger_contact_forces_n",
    "override_simulation_backend",
    "validate_g2_contact_substeps",
    "validate_rigid_physics",
]
