"""Import HuggingFace ``lerobot`` without shadowing by this directory name."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
_LOCAL_MARKER = "xArm6 + G2 LeRobot ACT dataset pipeline"


def _path_shadows_hf_lerobot(entry: str) -> bool:
    base = Path(entry).resolve()
    init_py = base / "lerobot" / "__init__.py"
    if not init_py.is_file():
        return False
    try:
        head = init_py.read_text(encoding="utf-8")[:200]
    except OSError:
        return False
    return _LOCAL_MARKER in head


def _purge_cached_lerobot() -> None:
    for key in list(sys.modules):
        if key == "lerobot" or key.startswith("lerobot."):
            mod = sys.modules.get(key)
            mod_file = getattr(mod, "__file__", "") or ""
            if mod_file and _LOCAL_DIR.resolve() in Path(mod_file).resolve().parents:
                del sys.modules[key]


def import_hf_lerobot():
    """Return the installed HuggingFace lerobot package module."""
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if not _path_shadows_hf_lerobot(p)]
    _purge_cached_lerobot()
    try:
        return importlib.import_module("lerobot")
    finally:
        sys.path[:] = saved_path


def get_lerobot_dataset_class():
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if not _path_shadows_hf_lerobot(p)]
    _purge_cached_lerobot()
    try:
        mod = importlib.import_module("lerobot.datasets.lerobot_dataset")
        return mod.LeRobotDataset
    finally:
        sys.path[:] = saved_path
