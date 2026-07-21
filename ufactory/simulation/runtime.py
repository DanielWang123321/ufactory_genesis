"""Single-owner Genesis process lifecycle."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
import threading
from types import TracebackType
from typing import TYPE_CHECKING

from ufactory.config.models import SimulationConfig

if TYPE_CHECKING:
    from ufactory.config.models import ResolvedRuntimeConfig

BACKEND_INIT_HINT = (
    "Genesis backend init failed. If this machine has no Genesis-supported GPU "
    "(including older cards that still enumerate under CUDA), use --backend cpu "
    "or set simulation.backend: cpu in a --config overlay, and install the CPU "
    "PyTorch wheel from https://pytorch.org before pip install -e '.[sim]'."
)


class GenesisRuntimeError(RuntimeError):
    pass


def override_simulation_backend(config: ResolvedRuntimeConfig, backend: str) -> ResolvedRuntimeConfig:
    """Return config with simulation.backend replaced (cli/overlay override)."""
    if backend not in {"cpu", "gpu"}:
        raise ValueError("simulation backend must be cpu or gpu")
    if config.simulation.backend == backend:
        return config
    return replace(config, simulation=replace(config.simulation, backend=backend))


def genesis_backend_constant(gs_module: object, backend: str) -> object:
    """Map config backend string to genesis.cpu / genesis.gpu."""
    if backend not in {"cpu", "gpu"}:
        raise ValueError("simulation backend must be cpu or gpu")
    return gs_module.gpu if backend == "gpu" else gs_module.cpu


class GenesisRuntimeManager(AbstractContextManager["GenesisRuntimeManager"]):
    """Initialize Genesis once, expose ownership, and destroy it on exit."""

    _lock = threading.RLock()
    _active: GenesisRuntimeManager | None = None

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.initialized = False

    @classmethod
    def active(cls) -> GenesisRuntimeManager:
        with cls._lock:
            if cls._active is None or not cls._active.initialized:
                raise GenesisRuntimeError(
                    "Genesis is not initialized. Enter GenesisRuntimeManager before building a scene."
                )
            return cls._active

    def __enter__(self) -> GenesisRuntimeManager:
        with self._lock:
            if self.__class__._active is not None:
                raise GenesisRuntimeError("another GenesisRuntimeManager already owns this process")
            from ufactory.simulation.compat import require_genesis_runtime

            gs = require_genesis_runtime()
            backend = genesis_backend_constant(gs, self.config.backend)
            try:
                gs.init(
                    backend=backend,
                    precision=self.config.precision,
                    logging_level="warning",
                    seed=self.config.seed,
                )
            except Exception as exc:  # noqa: BLE001 — surface actionable install/backend guidance
                raise GenesisRuntimeError(BACKEND_INIT_HINT) from exc
            self.initialized = True
            self.__class__._active = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        del exc_type, exc_value, traceback
        with self._lock:
            if self.initialized:
                import genesis as gs

                destroy = getattr(gs, "destroy", None)
                if callable(destroy):
                    destroy()
                self.initialized = False
            if self.__class__._active is self:
                self.__class__._active = None
        return None

    def assert_compatible(self, config: SimulationConfig) -> None:
        if config != self.config:
            raise GenesisRuntimeError("scene simulation config differs from active Genesis runtime")
