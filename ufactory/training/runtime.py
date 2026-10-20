"""Runtime setup and artifact directories shared by RL examples."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import shutil
import sys


def patch_pyglet_screen() -> None:
    """Pyglet's RandR fallback reads ``is_primary`` on the plain Xlib screen."""
    if sys.platform != "linux":
        return
    from pyglet.display.xlib import XlibScreen

    if not hasattr(XlibScreen, "is_primary"):
        XlibScreen.is_primary = False


def configure_runtime(
    cache_root: Path, *, project_root: Path, viewer: bool = False, viewer_platform: str | None = None
) -> None:
    """Keep generated caches in the checkout; configure GLX before its imports."""
    cache_root = Path(cache_root).expanduser().resolve()
    if not cache_root.is_relative_to(project_root.resolve()):
        raise ValueError("training caches must stay inside the project directory")
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
    if viewer and (viewer_platform == "glx" or os.environ.get("DISPLAY")):
        if viewer_platform is not None:
            os.environ["PYOPENGL_PLATFORM"] = viewer_platform
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
        import pyglet

        pyglet.options["shadow_window"] = False
        patch_pyglet_screen()
    elif viewer:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def require_rsl_runtime(policy: str) -> None:
    from packaging.version import Version

    try:
        try:
            metadata.version("rsl-rl")
        except metadata.PackageNotFoundError:
            if Version(metadata.version("rsl-rl-lib")) >= Version("5.4.2"):
                return
    except metadata.PackageNotFoundError:
        pass
    raise ImportError(f"Please uninstall 'rsl_rl' and install 'rsl-rl-lib>=5.4.2' for the {policy} policy.")


def require_checkpoint(path: Path, *, train_command: str, output_root: str) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Train first with "
            f"`{train_command}`; "
            f"checkpoints are written under {output_root}/<exp-name>/."
        )
    config = checkpoint.parent / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"missing training config beside checkpoint: {config}")
    return checkpoint


def prepare_log_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"training directory already exists: {path}; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)
