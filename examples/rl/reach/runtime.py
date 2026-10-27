"""Task paths and rendering settings for the reach RL example."""

from __future__ import annotations

import os
from pathlib import Path

from ufactory.training.artifacts import ArtifactError, load_training_config
from ufactory.training.runtime import (
    configure_runtime as configure_training_runtime,
    patch_pyglet_screen as patch_pyglet_screen,
    require_checkpoint as require_training_checkpoint,
    require_rsl_runtime as require_training_runtime,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECIPE = Path(__file__).with_name("recipe.yaml")
OUTPUT_ROOT = ROOT / "outputs" / "rl" / "reach"
CACHE_ROOT = ROOT / ".cache" / "rl" / "reach"


def configure_runtime(cache_root: Path | None = None, *, viewer: bool = False) -> None:
    configure_training_runtime(
        Path(cache_root or os.environ.get("REACH_CACHE_DIR", CACHE_ROOT)),
        project_root=ROOT,
        viewer=viewer,
        viewer_platform="glx",
    )


def require_rsl_runtime() -> None:
    require_training_runtime("reach")


def require_checkpoint(path: Path) -> Path:
    return require_training_checkpoint(
        path,
        train_command="python -m examples.rl.reach.train -e reach",
        output_root="outputs/rl/reach",
    )


def resolve_reach_robot(robot: str | None, config_path: Path | None = None) -> str:
    if robot is not None:
        return robot
    if config_path is not None and config_path.is_file():
        try:
            return str(load_training_config(config_path)["robot_key"])
        except ArtifactError:
            pass
    return "xarm6"
