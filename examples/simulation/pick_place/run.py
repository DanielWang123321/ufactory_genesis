"""Run the unified multi-robot pick-place command."""

from __future__ import annotations

import sys

from ufactory.cli.pick_place import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
