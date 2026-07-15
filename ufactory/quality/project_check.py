"""Local quality checks for v0.2.7 (no remote CI required)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import yaml

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.safety.gate import sha256_file


MODES = ("fast", "sim", "sdk-sim", "hardware", "release")
ROBOT_KEYS = ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850", "lite6")


@dataclass
class CheckResult:
    name: str
    status: str
    duration_s: float
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    reason: str = ""
    output_tail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(args: Sequence[str], root: Path) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


class ProjectCheck:
    def __init__(self, root: Path, mode: str) -> None:
        self.root = root
        self.mode = mode
        self.results: list[CheckResult] = []
        try:
            self.commit = _git(("rev-parse", "HEAD"), root)
            self.dirty = bool(_git(("status", "--porcelain"), root))
        except (OSError, subprocess.CalledProcessError):
            self.commit = "unknown"
            self.dirty = True

    def command(self, name: str, command: Sequence[str], *, env: dict[str, str] | None = None) -> CheckResult:
        started = time.perf_counter()
        try:
            run = subprocess.run(
                list(command),
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            output = (run.stdout + "\n" + run.stderr).strip()
            result = CheckResult(
                name=name,
                status="PASS" if run.returncode == 0 else "FAIL",
                duration_s=time.perf_counter() - started,
                command=list(command),
                exit_code=run.returncode,
                reason="" if run.returncode == 0 else f"command exited with {run.returncode}",
                output_tail=output[-12000:],
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = CheckResult(
                name=name,
                status="FAIL",
                duration_s=time.perf_counter() - started,
                command=list(command),
                reason=str(exc),
            )
        self.results.append(result)
        return result

    def record(self, name: str, function: Any) -> CheckResult:
        started = time.perf_counter()
        try:
            data = function() or {}
            result = CheckResult(name, "PASS", time.perf_counter() - started, data=data)
        except Exception as exc:
            result = CheckResult(name, "FAIL", time.perf_counter() - started, reason=str(exc))
        self.results.append(result)
        return result

    def incomplete(self, name: str, reason: str) -> None:
        self.results.append(CheckResult(name, "INCOMPLETE", 0.0, reason=reason))

    def config_and_assets(self) -> dict[str, Any]:
        store = RepositoryAssetStore.discover(self.root)
        manifest = store.validate_manifest()
        hashes = manifest.get("sha256", {})
        if not isinstance(hashes, dict):
            raise ValueError("asset manifest sha256 must be a mapping")
        for relative, expected in hashes.items():
            actual = sha256_file(store.require(relative))
            if actual != expected:
                raise ValueError(f"asset hash mismatch: {relative}")
        configs = {key: load_runtime_config(key, asset_store=store).sha256 for key in ROBOT_KEYS}
        if len(set(configs.values())) != len(configs):
            raise ValueError("robot default configurations unexpectedly share a hash")
        return {"manifest_version": manifest["manifest_version"], "config_sha256": configs}

    def fast(self) -> None:
        python = sys.executable
        self.record("config-assets", self.config_and_assets)
        self.command("ruff-check", (python, "-m", "ruff", "check", "."))
        self.command("ruff-format", (python, "-m", "ruff", "format", "--check", "."))
        self.command(
            "mypy-domain",
            (
                python,
                "-m",
                "mypy",
                "--strict",
                "--follow-imports=skip",
                "ufactory/config",
                "ufactory/safety/models.py",
                "ufactory/safety/interfaces.py",
                "ufactory/safety/statistics.py",
                "ufactory/safety/sdk_sim.py",
                "ufactory/safety/gate.py",
                "ufactory/safety/approved.py",
                "ufactory/safety/clock.py",
                "ufactory/trajectory/segments.py",
                "ufactory/trajectory/planner.py",
                "ufactory/trajectory/validation.py",
                "ufactory/trajectory/compile.py",
                "ufactory/trajectory/execution.py",
                "ufactory/trajectory/preflight.py",
            ),
        )
        self.command("compileall", (python, "-m", "compileall", "-q", "ufactory", "examples", "scripts"))
        self.command(
            "pytest-fast",
            (
                python,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not slow and not gpu and not integration and not display and not hardware",
                "--cov=ufactory",
                "--cov-branch",
                "--cov-fail-under=75",
                "--cov-report=term-missing",
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "QD_KERNEL_COVERAGE": "0"},
        )
        self.command(
            "pytest-safety-coverage",
            (
                python,
                "-m",
                "pytest",
                "-q",
                "tests/safety",
                "--cov=ufactory.safety",
                "--cov-branch",
                "--cov-fail-under=90",
                "--cov-report=term-missing",
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "QD_KERNEL_COVERAGE": "0"},
        )

    def sim(self) -> None:
        self.fast()
        self.command(
            "pytest-sim",
            (sys.executable, "-m", "pytest", "-q", "-m", "gpu", "--maxfail=1"),
        )

    def _inventory(self, path: Path | None, mode: str, *, confirm_real: bool) -> None:
        if path is None or not path.is_file():
            self.incomplete(f"{mode}-inventory", "an operator-maintained --inventory YAML is required")
            return
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            self.results.append(CheckResult(f"{mode}-inventory", "FAIL", 0.0, reason="invalid inventory schema"))
            return
        robots = data.get("robots")
        if not isinstance(robots, list):
            self.results.append(CheckResult(f"{mode}-inventory", "FAIL", 0.0, reason="robots must be a list"))
            return
        by_key = {str(item.get("robot_key")): item for item in robots if isinstance(item, dict)}
        missing = sorted(set(ROBOT_KEYS) - set(by_key))
        if missing:
            self.results.append(CheckResult(f"{mode}-inventory", "FAIL", 0.0, reason=f"missing robots: {missing}"))
            return
        if mode == "hardware" and not confirm_real:
            self.results.append(CheckResult("hardware-confirmation", "FAIL", 0.0, reason="--confirm-real is required"))
            return
        # Inventory commands are intentionally explicit, never inferred.  Each
        # row must point to a local evidence-producing command approved by the
        # operator; absent commands cannot be marked PASS.
        for key in ROBOT_KEYS:
            serial = str(by_key[key].get("serial_number", "")).strip()
            if len(serial) < 8 or serial == "REQUIRED":
                self.incomplete(f"{mode}-{key}-identity", "inventory requires the complete robot serial number")
                continue
            commands = by_key[key].get(f"{mode}_commands")
            if not isinstance(commands, list) or not commands:
                self.incomplete(f"{mode}-{key}", "inventory has no approved command matrix")
                continue
            for index, command in enumerate(commands):
                if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                    self.results.append(
                        CheckResult(f"{mode}-{key}-{index}", "FAIL", 0.0, reason="command must be argv list")
                    )
                    continue
                result = self.command(f"{mode}-{key}-{index}", command)
                result.data.update(robot_key=key, serial_number=serial)

    def snapshot_fast(self) -> None:
        # Use a detached worktree so snapshot tests that call ``git ls-files`` /
        # ``git show`` still have repository metadata (``git archive`` does not).
        with tempfile.TemporaryDirectory(prefix="ufactory-release-") as temp_dir:
            snapshot = Path(temp_dir) / "source"
            add = subprocess.run(
                ["git", "worktree", "add", "--detach", str(snapshot), "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            if add.returncode != 0:
                self.results.append(
                    CheckResult("snapshot-fast", "FAIL", 0.0, reason=add.stderr.strip() or add.stdout.strip())
                )
                return
            report_path = Path(temp_dir) / "snapshot-report.json"
            env = {**os.environ, "PYTHONPATH": str(snapshot), "CUDA_VISIBLE_DEVICES": ""}
            original_root = self.root
            try:
                self.root = snapshot
                result = self.command(
                    "snapshot-fast",
                    (sys.executable, "-m", "ufactory.quality.project_check", "fast", "--report", str(report_path)),
                    env=env,
                )
            finally:
                self.root = original_root
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(snapshot)],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                )
            if result.status == "PASS" and report_path.is_file():
                result.data["snapshot_report"] = json.loads(report_path.read_text(encoding="utf-8"))

    def release(self, version: str | None) -> None:
        if version != "0.2.7":
            self.results.append(CheckResult("release-version", "FAIL", 0.0, reason="--version must be 0.2.7"))
        else:
            self.results.append(CheckResult("release-version", "PASS", 0.0))
        if self.dirty:
            self.results.append(CheckResult("clean-worktree", "FAIL", 0.0, reason="release requires a clean worktree"))
            return
        self.results.append(CheckResult("clean-worktree", "PASS", 0.0))
        self.snapshot_fast()
        evidence_dir = self.root / "reports" / "project-check"
        for required in ("sim", "sdk-sim", "hardware"):
            matches = []
            for path in evidence_dir.glob(f"*_{required}.json") if evidence_dir.is_dir() else ():
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if report.get("git", {}).get("commit") == self.commit and report.get("passed") is True:
                    matches.append(str(path))
            if matches:
                self.results.append(CheckResult(f"evidence-{required}", "PASS", 0.0, data={"reports": matches}))
            else:
                self.incomplete(f"evidence-{required}", f"no PASS evidence for commit {self.commit}")
        # Audit the frozen project dependency set from uv.lock, not the whole
        # conda environment (which includes unrelated editable installs).
        with tempfile.TemporaryDirectory(prefix="ufactory-pip-audit-") as temp_dir:
            export_path = Path(temp_dir) / "uv-export.txt"
            filtered_path = Path(temp_dir) / "requirements.txt"
            export = subprocess.run(
                [
                    "uv",
                    "export",
                    "--frozen",
                    "--no-annotate",
                    "--no-hashes",
                    "--extra",
                    "sim",
                    "--extra",
                    "real",
                    "--extra",
                    "rl",
                    "--extra",
                    "dynamics",
                    "--extra",
                    "showcase",
                    "--extra",
                    "trajectory",
                    "--extra",
                    "dev",
                    "-o",
                    str(export_path),
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            if export.returncode != 0:
                self.results.append(
                    CheckResult(
                        "pip-audit",
                        "FAIL",
                        0.0,
                        reason=(export.stderr or export.stdout or "uv export failed").strip(),
                    )
                )
            else:
                lines = [
                    line
                    for line in export_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#") and not line.startswith("-e ")
                ]
                filtered_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                # Validated pins may carry advisories without a compatible fix in this stack.
                ignored = (
                    "PYSEC-2026-139",  # torch 2.10.0
                    "CVE-2025-3000",  # torch 2.10.0
                    "PYSEC-2026-165",
                    "PYSEC-2026-2249",
                    "PYSEC-2026-2250",
                    "PYSEC-2026-2251",
                    "PYSEC-2026-2252",
                    "PYSEC-2026-2253",
                    "PYSEC-2026-2254",
                    "PYSEC-2026-2255",
                    "PYSEC-2026-2256",
                    "PYSEC-2026-2257",
                    "PYSEC-2026-2874",  # pillow (Genesis stack)
                    "PYSEC-2026-196",  # pip
                    "PYSEC-2026-3447",  # setuptools
                    "PYSEC-2026-2689",  # onnx
                )
                command: list[str] = [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--strict",
                    "--progress-spinner",
                    "off",
                    "--timeout",
                    "30",
                    "-r",
                    str(filtered_path),
                ]
                for vuln_id in ignored:
                    command.extend(["--ignore-vuln", vuln_id])
                self.command("pip-audit", command)

    def report(self) -> dict[str, Any]:
        passed = bool(self.results) and all(result.status == "PASS" for result in self.results)
        return {
            "schema_version": 1,
            "mode": self.mode,
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": self.commit, "dirty": self.dirty},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "genesis": _version("genesis-world"),
                "torch": _version("torch"),
                "xarm_sdk": _version("xarm-python-sdk"),
                "pinocchio": _version("pin"),
                "coal": _version("coal"),
            },
            "checks": [asdict(result) for result in self.results],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="project-check")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--confirm-real", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    root = RepositoryAssetStore.discover().root
    check = ProjectCheck(root, args.mode)
    if args.mode == "fast":
        check.fast()
    elif args.mode == "sim":
        check.sim()
    elif args.mode == "sdk-sim":
        check._inventory(args.inventory, "sdk-sim", confirm_real=False)
    elif args.mode == "hardware":
        check._inventory(args.inventory, "hardware", confirm_real=args.confirm_real)
    else:
        check.release(args.version)
    report = check.report()
    path = args.report
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / "reports" / "project-check" / f"{stamp}_{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"project-check {args.mode}: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"report: {path}")
    for result in check.results:
        print(f"  {result.status:10s} {result.name}{': ' + result.reason if result.reason else ''}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
