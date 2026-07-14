"""Shared precise NumPy types for strict domain modules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

__all__ = ["FloatArray"]
