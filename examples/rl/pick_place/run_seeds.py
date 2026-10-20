"""Train three independent PPO seeds, evaluate 64 episodes each, and record B=1 GLB replays."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .runtime import OUTPUT_ROOT, ROOT


def run(command: list[str], log: Path) -> None:
    print(f"Running {' '.join(command)}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("-B", "--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.num_envs < 1 or not 1 <= args.max_iterations <= 2000:
        parser.error("require positive --num-envs and --max-iterations in [1, 2000]")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in (1, 2, 3) for seed in args.seeds):
        parser.error("--seeds must be distinct values from 1, 2, 3")
    for seed in args.seeds:
        directory = OUTPUT_ROOT / f"pick_place_e2e_s{seed}"
        if directory.exists():
            parser.error(f"experiment already exists: {directory}; retain it and choose a fresh project snapshot")
    logs = OUTPUT_ROOT / "e2e_validation"
    logs.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "seeds": {},
        "passed": False,
    }
    for seed in args.seeds:
        name = f"pick_place_e2e_s{seed}"
        directory = OUTPUT_ROOT / name
        prefix = [sys.executable, "-B", "-m"]
        status = {"passed": False, "experiment": name}
        try:
            run(
                prefix
                + [
                    "examples.rl.pick_place.train",
                    "-e",
                    name,
                    "--seed",
                    str(seed),
                    "-B",
                    str(args.num_envs),
                    "--max-iterations",
                    str(args.max_iterations),
                    "--eval-interval",
                    "50",
                    "--eval-episodes",
                    "64",
                    "--backend",
                    args.backend,
                ],
                logs / f"s{seed}_train.log",
            )
            checkpoint = directory / "model_best.pt"
            evaluation_command = prefix + [
                "examples.rl.pick_place.evaluate",
                "--checkpoint",
                str(checkpoint),
                "-B",
                "32",
                "--episodes",
                "64",
                "--seed",
                str(18000 + seed),
                "--min-success-rate",
                "0.98",
                "--max-press-mm",
                "1.0",
                "--quiet-episodes",
                "--out",
                str(directory / "test.json"),
                "--backend",
                args.backend,
            ]
            try:
                run(evaluation_command, logs / f"s{seed}_evaluate.log")
            except subprocess.CalledProcessError as error:
                # A failed acceptance test still needs its B=1 video for review.
                if error.returncode != 1 or not (directory / "test.json").is_file():
                    raise
            status["evaluation"] = json.loads((directory / "test.json").read_text())
            run(
                prefix
                + [
                    "examples.rl.pick_place.visualize",
                    "--checkpoint",
                    str(checkpoint),
                    "-B",
                    "1",
                    "--episodes",
                    "1",
                    "--seed",
                    str(18000 + seed),
                    "--headless",
                    "--record",
                    str(directory / "replay.mp4"),
                    "--out",
                    str(directory / "glb.json"),
                    "--backend",
                    args.backend,
                ],
                logs / f"s{seed}_video.log",
            )
            status["glb"] = json.loads((directory / "glb.json").read_text())
            status["passed"] = bool(status["evaluation"]["passed"] and status["glb"]["passed"])
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            status["error"] = str(error)
            test_report = directory / "test.json"
            if test_report.is_file():
                status["evaluation"] = json.loads(test_report.read_text())
        results["seeds"][str(seed)] = status
        results["passed"] = len(results["seeds"]) == len(args.seeds) and all(
            s["passed"] for s in results["seeds"].values()
        )
        (logs / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if not results["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
