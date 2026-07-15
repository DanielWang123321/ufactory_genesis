"""Run the configuration-driven multi-robot packaging showcase."""

from __future__ import annotations

import sys

from ufactory.cli.packaging import main as packaging_main


def main() -> None:
    raise SystemExit(packaging_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
