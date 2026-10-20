"""Task paths and rendering settings for the pick_place RL example."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ufactory.training.runtime import (
    configure_runtime as configure_training_runtime,
    patch_pyglet_screen as patch_pyglet_screen,
    require_checkpoint as require_training_checkpoint,
    require_rsl_runtime as require_training_runtime,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECIPE = Path(__file__).with_name("recipe.yaml")
OUTPUT_ROOT = ROOT / "outputs" / "rl" / "pick_place"
CACHE_ROOT = ROOT / ".cache" / "rl" / "pick_place"


def configure_runtime(cache_root: Path | None = None, *, viewer: bool = False) -> None:
    configure_training_runtime(
        Path(cache_root or os.environ.get("PICK_PLACE_CACHE_DIR", CACHE_ROOT)),
        project_root=ROOT,
        viewer=viewer,
    )


def require_rsl_runtime() -> None:
    require_training_runtime("pick-place PPO")


def require_backend(gs: Any, requested: str) -> None:
    if requested == "gpu" and not str(gs.device).startswith("cuda"):
        gs.destroy()
        raise RuntimeError("GPU was requested but Genesis fell back to CPU; use --backend cpu explicitly")


def require_checkpoint(path: Path) -> Path:
    return require_training_checkpoint(
        path,
        train_command="python -m examples.rl.pick_place.train -e pick_place_e2e_s1",
        output_root="outputs/rl/pick_place",
    )
