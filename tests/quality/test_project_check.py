"""Unit tests for project-check helpers (no GPU / inventory)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ufactory.quality.project_check import (
    _fast_report_complete,
    _path_allowed_for_evidence_carry,
    build_pip_audit_command,
    compare_lock_to_installed,
    filter_uv_export_lines,
    package_version,
)


def test_package_version_reads_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.2.8"\n',
        encoding="utf-8",
    )
    assert package_version(tmp_path) == "0.2.8"


def test_package_version_rejects_missing(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="project.version"):
        package_version(tmp_path)


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("README.md", True),
        ("CHANGELOG.md", True),
        ("docs/guide.md", True),
        ("examples/README.md", True),
        ("ufactory/quality/project_check.py", False),
        ("tests/foo.py", False),
    ],
)
def test_evidence_carry_path_whitelist(path: str, allowed: bool) -> None:
    assert _path_allowed_for_evidence_carry(path) is allowed


def test_fast_report_complete_requires_all_checks() -> None:
    checks = [
        {"name": name, "status": "PASS"}
        for name in (
            "config-assets",
            "ruff-check",
            "ruff-format",
            "mypy-domain",
            "compileall",
            "pytest-fast",
            "pytest-safety-coverage",
        )
    ]
    assert _fast_report_complete({"mode": "fast", "passed": True, "checks": checks}) is True
    assert _fast_report_complete({"mode": "fast", "passed": True, "checks": checks[:-1]}) is False
    assert _fast_report_complete({"mode": "sim", "passed": True, "checks": checks}) is False


def test_filter_uv_export_lines_drops_comments_and_editables() -> None:
    text = "# comment\ntorch==2.10.0\n-e .\npillow==11.3.0\n"
    assert filter_uv_export_lines(text) == ["torch==2.10.0", "pillow==11.3.0"]


def test_build_pip_audit_command_skips_venv_resolution(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text("torch==2.10.0\n", encoding="utf-8")
    command = build_pip_audit_command(req, python="/usr/bin/python")
    assert command[:3] == ["/usr/bin/python", "-m", "pip_audit"]
    assert "--no-deps" in command
    assert "--disable-pip" in command
    assert "--strict" in command
    assert command[command.index("-r") + 1] == str(req)
    assert "--ignore-vuln" in command


def test_compare_lock_to_installed_reports_drift_and_missing() -> None:
    import importlib.metadata as md

    lines = [
        "demo-matched==1.0.0",
        "demo-drift==2.0.0",
        "demo-absent==3.0.0",
        "skip-me>=1.0",
    ]

    def fake_version(name: str) -> str:
        if name == "demo-matched":
            return "1.0.0"
        if name == "demo-drift":
            return "2.1.0"
        raise md.PackageNotFoundError(name)

    with patch("ufactory.quality.project_check.importlib.metadata.version", side_effect=fake_version):
        result = compare_lock_to_installed(lines)

    assert result["matched"] == 1
    assert result["mismatches"] == [{"name": "demo-drift", "locked": "2.0.0", "installed": "2.1.0"}]
    assert result["missing"] == [{"name": "demo-absent", "locked": "3.0.0"}]
    assert result["skipped"] >= 1


def test_compare_lock_to_installed_respects_false_markers() -> None:
    lines = ["only-windows==1.0.0 ; sys_platform == 'win32'"]
    with patch(
        "ufactory.quality.project_check.importlib.metadata.version",
        side_effect=AssertionError("should not query inactive markers"),
    ):
        result = compare_lock_to_installed(lines)
    assert result["matched"] == 0
    assert result["mismatches"] == []
    assert result["missing"] == []
