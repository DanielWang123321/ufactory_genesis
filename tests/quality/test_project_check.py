"""Unit tests for project-check helpers (no GPU / inventory)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.quality.project_check import (
    _fast_report_complete,
    _path_allowed_for_evidence_carry,
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
