#!/usr/bin/env python3
"""CLI: build a sanitized release-evidence summary from project-check JSON reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from ufactory.quality.evidence_summary import write_evidence_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sanitized release-evidence summary from local project-check JSON reports. "
            "Full reports under reports/ stay gitignored; the summary omits command lines, "
            "inventory paths, IPs, and serials."
        )
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="One or more project-check JSON reports (e.g. reports/project-check/v0.2.7_sim.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Sanitized summary JSON path (safe to attach to a GitHub Release)",
    )
    args = parser.parse_args(argv)
    document = write_evidence_summary(list(args.reports), args.output)
    print(f"wrote {args.output} (all_passed={document['all_passed']}, reports={len(document['reports'])})")
    return 0 if document["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
