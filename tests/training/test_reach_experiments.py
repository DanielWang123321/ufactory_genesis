"""Reach workflow failure handling, artifact preservation and subprocess cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from examples.rl.reach import run_experiments as experiments


def test_defaults_and_three_seed_extension():
    args = experiments.parse_args([])
    assert args.seeds == [7]
    assert args.num_envs == 16384
    assert args.max_iterations == 300
    args = experiments.parse_args(["--seeds", "1", "7", "17", "-B", "64", "--max-iterations", "2"])
    assert args.seeds == [1, 7, 17]
    assert args.num_envs == 64 and args.max_iterations == 2


@pytest.mark.parametrize("argv", [["--seeds", "7", "7"], ["-B", "0"], ["--max-iterations", "0"], ["--stage", "select"]])
def test_invalid_experiments_fail_before_creating_artifacts(argv):
    with pytest.raises(SystemExit) as exc:
        experiments.parse_args(argv)
    assert exc.value.code == 2


def test_existing_output_is_preserved(tmp_path):
    artifact = tmp_path / "existing.txt"
    artifact.write_text("keep")
    with pytest.raises(FileExistsError):
        experiments.main(["--output", str(tmp_path)])
    assert artifact.read_text() == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing.txt"]


def test_failed_acceptance_returns_nonzero_and_records_both_holdouts(tmp_path, monkeypatch):
    directory = tmp_path / "seed-7"
    directory.mkdir()
    (directory / "model_299.pt").touch()
    (directory / "checkpoint-selection.json").write_text(json.dumps({"selected_checkpoint": "model_299.pt"}))
    calls = []

    def evaluate(output, name, arguments, *, allowed):
        calls.append(name)
        seed = int(arguments[arguments.index("--seed") + 1])
        report = Path(arguments[arguments.index("--summary-json") + 1])
        report.write_text(json.dumps({"passed": seed == 17001, "summary": {"success_rate": 0.9, "mean_m": 0.011}}))
        return 1 if seed == 17000 else 0

    monkeypatch.setattr(experiments, "run", evaluate)
    monkeypatch.setattr(experiments, "configure_runtime", lambda: None)
    assert experiments.main(["--output", str(tmp_path), "--stage", "accept"]) == 1
    assert calls == ["seed-7-acceptance-17000", "seed-7-acceptance-17001"]
    status = json.loads((tmp_path / "accept-status.json").read_text())
    assert status["stop_reason"] == "acceptance_failed"
    assert status["seeds"][0]["passed"] is False
    assert len(status["seeds"][0]["holdouts"]) == 2
    before = (directory / "acceptance-result.json").read_bytes()
    with pytest.raises(FileExistsError):
        experiments.main(["--output", str(tmp_path), "--stage", "accept"])
    assert (directory / "acceptance-result.json").read_bytes() == before


def test_subprocess_error_has_nonzero_exit_and_preserves_log(tmp_path):
    with pytest.raises(RuntimeError, match="exited 3"):
        experiments.run(tmp_path, "failure", ["-c", "print('diagnostic', flush=True); raise SystemExit(3)"])
    assert "diagnostic" in (tmp_path / "failure.log").read_text()
    status = json.loads((tmp_path / "failure-process.json").read_text())
    assert status["exit_code"] == 3 and status["status"] == "exited"
    with pytest.raises(FileExistsError):
        experiments.run(tmp_path, "failure", ["-c", "raise SystemExit(0)"])


@pytest.mark.skipif(sys.platform != "linux", reason="process-group cleanup uses POSIX signals and /proc")
def test_interrupt_reaps_child_and_terminates_grandchild(tmp_path):
    marker = tmp_path / "pids.json"
    child = (
        "import json,os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        f"open({str(marker)!r},'w').write(json.dumps([os.getpid(),p.pid])); time.sleep(120)"
    )
    controller = (
        "from pathlib import Path; from examples.rl.reach.run_experiments import run; "
        f"run(Path({str(tmp_path)!r}), 'interrupt', ['-c', {child!r}])"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", controller], cwd=experiments.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "subprocess tree did not start"
        pids = json.loads(marker.read_text())
        os.kill(process.pid, signal.SIGINT)
        assert process.wait(timeout=15) != 0
        status = json.loads((tmp_path / "interrupt-process.json").read_text())
        assert status["status"] == "interrupted" and status["exit_code"] is not None
        for pid in pids:
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z", f"process {pid} still running"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
