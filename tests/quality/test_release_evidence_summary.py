"""Unit tests for sanitized release-evidence summaries."""

from __future__ import annotations

import json
from pathlib import Path

from ufactory.quality.evidence_summary import summarize_report, write_evidence_summary


def test_summarize_report_redacts_ip_and_serial() -> None:
    report = {
        "schema_version": 1,
        "mode": "hardware",
        "passed": True,
        "generated_at_utc": "2026-07-17T00:00:00+00:00",
        "git": {"commit": "deadbeef", "dirty": False},
        "environment": {
            "python": "3.13.0",
            "platform": "Linux",
            "genesis": "1.3.0",
            "torch": "2.10.0",
            "xarm_sdk": "1.0.0",
            "pinocchio": None,
        },
        "checks": [
            {
                "name": "hardware-xarm6",
                "status": "PASS",
                "duration_s": 12.0,
                "reason": "ok at 192.168.1.65 serial=ABC123",
                "data": {"command": ["should", "not", "appear"]},
            }
        ],
    }
    summary = summarize_report(report, source="reports/project-check/v0.2.7_hardware.json")
    assert summary["source_file"] == "v0.2.7_hardware.json"
    assert summary["passed"] is True
    assert summary["environment"]["xarm_sdk_installed"] is True
    assert summary["environment"]["pinocchio_installed"] is False
    assert "command" not in summary["checks"][0]
    assert summary["checks"][0]["reason"] == "ok at <redacted-ip> <redacted-id>"


def test_write_evidence_summary_roundtrip(tmp_path: Path) -> None:
    report_path = tmp_path / "v0.2.7_sim.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "sim",
                "passed": True,
                "generated_at_utc": "2026-07-17T00:00:00+00:00",
                "git": {"commit": "abc", "dirty": False},
                "environment": {"python": "3.13.0", "platform": "Linux", "genesis": "1.3.0"},
                "checks": [{"name": "pytest-gpu", "status": "PASS", "duration_s": 1.0, "reason": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    document = write_evidence_summary([report_path], output)
    assert document["all_passed"] is True
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["kind"] == "ufactory_genesis_release_evidence_summary"
    assert loaded["reports"][0]["mode"] == "sim"
