"""Thin UF850 wrapper for the unified v0.2.5 grasp-place command."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from ufactory.cli.grasp_place import main


if __name__ == "__main__":
    raise SystemExit(main(["--robot", "uf850", *sys.argv[1:]]))
