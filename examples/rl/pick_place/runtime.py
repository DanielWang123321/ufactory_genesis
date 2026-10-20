"""Shared paths, caches and dependency checks for the pick-place RL example."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECIPE = Path(__file__).with_name("recipe.yaml")
OUTPUT_ROOT = ROOT / "outputs" / "rl" / "pick_place"
CACHE_ROOT = ROOT / ".cache" / "rl" / "pick_place"


def patch_pyglet_screen() -> None:
    """Pyglet's RandR fallback reads ``is_primary`` on the plain Xlib screen."""
    if sys.platform != "linux":
        return
    from pyglet.display.xlib import XlibScreen

    if not hasattr(XlibScreen, "is_primary"):
        XlibScreen.is_primary = False


def configure_runtime(cache_root: Path | None = None, *, viewer: bool = False) -> None:
    """Keep generated caches in the checkout; configure GLX before its imports."""
    cache_root = Path(cache_root or os.environ.get("PICK_PLACE_CACHE_DIR", CACHE_ROOT)).expanduser().resolve()
    if not cache_root.is_relative_to(ROOT):
        raise ValueError("pick_place caches must stay inside the project directory")
    for variable, name in (
        ("XDG_CACHE_HOME", "xdg"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("GS_CACHE_FILE_PATH", "genesis"),
        ("QD_OFFLINE_CACHE_FILE_PATH", "quadrants"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("__GL_SHADER_DISK_CACHE_PATH", "gl"),
        ("TORCH_HOME", "torch"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        path = cache_root / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if viewer and os.environ.get("DISPLAY"):
        os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
        import pyglet

        pyglet.options["shadow_window"] = False
        patch_pyglet_screen()
    elif viewer:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def require_rsl_runtime() -> None:
    from packaging.version import Version

    try:
        try:
            metadata.version("rsl-rl")
        except metadata.PackageNotFoundError:
            if Version(metadata.version("rsl-rl-lib")) >= Version("5.4.2"):
                return
    except metadata.PackageNotFoundError:
        pass
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib>=5.4.2' for the pick-place PPO policy.")


def require_backend(gs: Any, requested: str) -> None:
    if requested == "gpu" and not str(gs.device).startswith("cuda"):
        gs.destroy()
        raise RuntimeError("GPU was requested but Genesis fell back to CPU; use --backend cpu explicitly")


def require_checkpoint(path: Path) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Train first with "
            "`python -m examples.rl.pick_place.train -e pick_place_e2e_s1`; "
            "checkpoints are written under outputs/rl/pick_place/<exp-name>/."
        )
    config = checkpoint.parent / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"missing training config beside checkpoint: {config}")
    return checkpoint
