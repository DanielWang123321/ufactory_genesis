"""Sanitize local project-check reports for public Release attachments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SERIALISH_RE = re.compile(r"\b(?:SN|serial)[=:\s]+[A-Za-z0-9_-]+\b", re.IGNORECASE)


def scrub_text(text: str) -> str:
    text = _IP_RE.sub("<redacted-ip>", text)
    return _SERIALISH_RE.sub("<redacted-id>", text)


def summarize_check(check: dict[str, Any]) -> dict[str, Any]:
    reason = check.get("reason")
    if isinstance(reason, str):
        reason = scrub_text(reason)
    return {
        "name": check.get("name"),
        "status": check.get("status"),
        "duration_s": check.get("duration_s"),
        "reason": reason,
    }


def summarize_report(report: dict[str, Any], *, source: str) -> dict[str, Any]:
    env = report.get("environment") if isinstance(report.get("environment"), dict) else {}
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    return {
        "source_file": Path(source).name,
        "schema_version": report.get("schema_version"),
        "mode": report.get("mode"),
        "passed": report.get("passed"),
        "generated_at_utc": report.get("generated_at_utc"),
        "git": {
            "commit": (report.get("git") or {}).get("commit"),
            "dirty": (report.get("git") or {}).get("dirty"),
        },
        "environment": {
            "python": env.get("python"),
            "platform": env.get("platform"),
            "genesis": env.get("genesis"),
            "torch": env.get("torch"),
            "xarm_sdk_installed": bool(env.get("xarm_sdk")),
            "pinocchio_installed": bool(env.get("pinocchio")),
        },
        "checks": [summarize_check(item) for item in checks if isinstance(item, dict)],
        "check_counts": {
            "total": len(checks),
            "pass": sum(1 for item in checks if isinstance(item, dict) and item.get("status") == "PASS"),
            "fail": sum(1 for item in checks if isinstance(item, dict) and item.get("status") == "FAIL"),
        },
    }


def build_evidence_document(report_paths: list[Path]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"report must be a JSON object: {path}")
        summaries.append(summarize_report(payload, source=str(path)))
    return {
        "schema_version": 1,
        "kind": "ufactory_genesis_release_evidence_summary",
        "all_passed": all(item.get("passed") is True for item in summaries),
        "reports": summaries,
    }


def write_evidence_summary(report_paths: list[Path], output: Path) -> dict[str, Any]:
    document = build_evidence_document(report_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document
