"""Independent rigid-body reference backend (Pinocchio) for dynamics validation.

Holds :class:`PinocchioReference` and the gravity default vector. This module
imports ``pinocchio`` lazily inside the constructor so that merely importing the
report/analysis layer never hard-requires Pinocchio; ``load_reference_backend``
controls the ``--require-reference`` CLI option.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# Align with the common Genesis scene gravity magnitude (m/s^2).
DEFAULT_GRAVITY_VECTOR = (0.0, 0.0, -9.81)


class PinocchioReference:
    """Optional independent rigid-body dynamics reference backend."""

    def __init__(
        self,
        urdf_path: str | Any,
        *,
        gravity_vector: Sequence[float] | None = None,
    ):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "Pinocchio reference backend is unavailable. Install optional dependency: pip install '.[dynamics]'"
            ) from exc

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        g = np.asarray(gravity_vector if gravity_vector is not None else DEFAULT_GRAVITY_VECTOR, dtype=np.float64)
        self.model.gravity.linear = g

    @property
    def available(self) -> bool:
        return True

    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float64)
        mat = self.pin.crba(self.model, self.data, q_np)
        return np.asarray((mat + mat.T) * 0.5, dtype=np.float64)

    def gravity(self, q: Sequence[float]) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float64)
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q_np), dtype=np.float64)

    def rnea(self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        return np.asarray(
            self.pin.rnea(
                self.model,
                self.data,
                np.asarray(q, dtype=np.float64),
                np.asarray(qd, dtype=np.float64),
                np.asarray(qdd, dtype=np.float64),
            ),
            dtype=np.float64,
        )


def load_reference_backend(
    urdf_path: str | Any,
    *,
    required: bool = False,
    gravity_vector: Sequence[float] | None = None,
) -> PinocchioReference | None:
    try:
        return PinocchioReference(urdf_path, gravity_vector=gravity_vector)
    except ImportError:
        if required:
            raise
        return None


def armature_aligned_mass(mass_genesis: np.ndarray, armature: Sequence[float] | np.ndarray | None) -> np.ndarray:
    """Return ``mass_genesis`` with the reflected rotor armature removed.

    Genesis adds a reflected rotor inertia (``armature``) inboard of each joint
    on top of the link CRBA. To compare apples-to-apples with an independent
    URDF CRBA we subtract ``diag(armature)``. URDFs without an ``<armature>``
    tag should read back zero armature, making this a no-op (the call sites
    additionally assert/record the readback so a hidden implicit default is
    visible).
    """
    m = np.asarray(mass_genesis, dtype=np.float64).copy()
    if armature is None:
        return m
    arm = np.asarray(armature, dtype=np.float64).reshape(-1)
    k = min(len(arm), m.shape[0])
    if k > 0:
        m[:k, :k] -= np.diag(arm[:k])
    return m
