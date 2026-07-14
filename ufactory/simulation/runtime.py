"""Single-owner Genesis process lifecycle."""

from __future__ import annotations

from contextlib import AbstractContextManager
import threading
from types import TracebackType

from ufactory.config.models import SimulationConfig
from ufactory.simulation.compat import require_genesis_runtime


class GenesisRuntimeError(RuntimeError):
    pass


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
            gs = require_genesis_runtime()

            backend = gs.gpu if self.config.backend == "gpu" else gs.cpu
            gs.init(
                backend=backend,
                precision=self.config.precision,
                logging_level="warning",
                seed=self.config.seed,
            )
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
