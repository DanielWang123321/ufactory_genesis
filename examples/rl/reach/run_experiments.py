"""Serial reach smoke, training, checkpoint selection and independent acceptance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ufactory.config import load_runtime_config
from ufactory.training import build_reach_task_configs, load_training_recipe

from .runtime import DEFAULT_RECIPE, OUTPUT_ROOT, ROOT, configure_runtime


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_process(process: subprocess.Popen) -> None:
    """Reap the child and terminate any descendants in its process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(output: Path, name: str, arguments: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
    command = [sys.executable, *arguments]
    log = output / (name + ".log")
    record_path = output / (name + "-process.json")
    performance_path = output / (name + "-performance.jsonl")
    for path in (log, record_path, performance_path):
        if path.exists():
            raise FileExistsError(f"experiment artifact already exists: {path}")
    record = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    process = None
    with log.open("x", encoding="utf-8") as stream, performance_path.open("x", encoding="utf-8") as performance:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = process.pid
            write_json(record_path, record)
            print(f"{name} pid={process.pid}; log={log}", flush=True)
            next_sample = 0.0
            while process.poll() is None:
                if time.monotonic() >= next_sample:
                    try:
                        sampled = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        gpu = sampled.stdout.strip() or sampled.stderr.strip()
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        gpu = str(exc)
                    performance.write(
                        json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu}) + "\n"
                    )
                    performance.flush()
                    next_sample = time.monotonic() + 5
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            record["status"] = "interrupted"
            if process is not None:
                stop_process(process)
            raise
        finally:
            record.update(
                exit_code=None if process is None else process.returncode,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            if record["status"] == "running":
                record["status"] = "exited"
            write_json(record_path, record)
    if process.returncode not in allowed:
        raise RuntimeError(f"{name} exited {process.returncode}; see {log}")
    return process.returncode


def record_source_hashes(output: Path, args: argparse.Namespace) -> None:
    runtime = load_runtime_config("xarm6", task="reach", config_path=args.runtime_config)
    _, _, robot = build_reach_task_configs("xarm6", recipe_path=args.recipe, runtime_config_path=args.runtime_config)
    sources = list(Path(__file__).parent.glob("*.py")) + [args.recipe, Path(robot["urdf_path"]), ROOT / "uv.lock"]
    sources += list((ROOT / "ufactory/training").glob("*.py"))
    sources += list((ROOT / "ufactory/training/logic").glob("*.py"))
    sources += [ROOT / source for source in runtime.sources]
    if args.runtime_config is not None:
        sources.append(args.runtime_config)
    hashes = {
        os.path.relpath(path.resolve(), ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(sources))
    }
    write_json(
        output / "source-sha256.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "recipe": str(args.recipe),
            "runtime_config": None if args.runtime_config is None else str(args.runtime_config),
            "runtime_config_sha256": runtime.sha256,
            "hashes": hashes,
        },
    )


def common_options(args: argparse.Namespace) -> list[str]:
    options = ["--recipe", str(args.recipe)]
    if args.runtime_config is not None:
        options += ["--runtime-config", str(args.runtime_config)]
    return options


def smoke(output: Path, args: argparse.Namespace) -> None:
    common = common_options(args)
    run(
        output,
        "smoke-train",
        [
            "-m",
            "examples.rl.reach.train",
            *common,
            "--log-dir",
            str(output / "smoke"),
            "-B",
            "64",
            "--max-iterations",
            "2",
            "--seed",
            str(args.seeds[0]),
        ],
    )
    run(
        output,
        "smoke-resume",
        [
            "-m",
            "examples.rl.reach.train",
            *common,
            "--log-dir",
            str(output / "smoke-resume"),
            "-B",
            "64",
            "--max-iterations",
            "1",
            "--seed",
            str(args.seeds[0]),
            "--resume",
            str(output / "smoke/model_1.pt"),
        ],
    )
    run(
        output,
        "smoke-evaluate",
        [
            "-m",
            "examples.rl.reach.evaluate",
            *common,
            "--checkpoint",
            str(output / "smoke-resume/model_2.pt"),
            "-B",
            "64",
            "--episodes",
            "8",
            "--seed",
            "17002",
            "--quiet-episodes",
            "--summary-json",
            str(output / "smoke-evaluation.json"),
        ],
    )


def select_checkpoint(output: Path, directory: Path, seed: int, args: argparse.Namespace) -> Path:
    selection = directory / "checkpoint-selection.json"
    if selection.exists():
        raise FileExistsError(f"checkpoint selection already exists: {selection}")
    checkpoints = sorted(directory.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints to select in {directory}")
    candidates = []
    for checkpoint in checkpoints:
        report = directory / (checkpoint.stem + "-development.json")
        run(
            output,
            f"seed-{seed}-{checkpoint.stem}-development",
            [
                "-m",
                "examples.rl.reach.evaluate",
                *common_options(args),
                "--checkpoint",
                str(checkpoint),
                "-B",
                "128",
                "--episodes",
                "128",
                "--seed",
                "17002",
                "--quiet-episodes",
                "--summary-json",
                str(report),
            ],
        )
        candidates.append(
            {"checkpoint": checkpoint.name, "report": report.name, "summary": json.loads(report.read_text())["summary"]}
        )
    candidates.sort(
        key=lambda entry: (
            -entry["summary"]["success_rate"],
            entry["summary"]["mean_m"],
            entry["summary"]["mean_success_elapsed_s"]
            if entry["summary"]["mean_success_elapsed_s"] is not None
            else float("inf"),
        )
    )
    selected = directory / candidates[0]["checkpoint"]
    write_json(
        selection,
        {
            "selected_checkpoint": selected.name,
            "candidates": candidates,
            "development_seed": 17002,
            "development_episodes": 128,
            "ranking": ["success_rate_desc", "mean_m_asc", "mean_success_elapsed_s_asc"],
        },
    )
    print(f"Selected checkpoint: {selected}", flush=True)
    return selected


def accept_checkpoint(output: Path, directory: Path, selected: Path, seed: int, args: argparse.Namespace) -> dict:
    result_path = directory / "acceptance-result.json"
    if result_path.exists():
        raise FileExistsError(f"acceptance result already exists: {result_path}")
    holdouts = []
    accepted = True
    for evaluation_seed in (17000, 17001):
        report = directory / f"acceptance-{evaluation_seed}.json"
        code = run(
            output,
            f"seed-{seed}-acceptance-{evaluation_seed}",
            [
                "-m",
                "examples.rl.reach.evaluate",
                *common_options(args),
                "--checkpoint",
                str(selected),
                "-B",
                "128",
                "--episodes",
                "512",
                "--seed",
                str(evaluation_seed),
                "--min-success-rate",
                "0.95",
                "--max-mean-distance-m",
                "0.010",
                "--quiet-episodes",
                "--summary-json",
                str(report),
            ],
            allowed=(0, 1),
        )
        payload = json.loads(report.read_text())
        accepted &= code == 0 and payload["passed"]
        holdouts.append({"seed": evaluation_seed, "summary": payload["summary"], "report": report.name})
    result = {"training_seed": seed, "selected_checkpoint": selected.name, "passed": accepted, "holdouts": holdouts}
    write_json(result_path, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--output-dir", type=Path)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-B", "--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument(
        "--stage",
        choices=("all", "train", "select", "accept"),
        default="all",
        help="all: full serial workflow; select/accept: use an existing output directory",
    )
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain distinct non-negative integers")
    if args.stage in ("select", "accept") and args.output is None:
        parser.error("--stage select/accept requires --output")
    recipe = load_training_recipe(args.recipe)
    args.num_envs = args.num_envs if args.num_envs is not None else int(recipe["environment"]["num_envs"])
    args.max_iterations = (
        args.max_iterations if args.max_iterations is not None else int(recipe["train"]["runner"]["max_iterations"])
    )
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--num-envs and --max-iterations must be positive")
    args.output = (args.output or OUTPUT_ROOT / ("reach-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    args.recipe = args.recipe.expanduser().resolve()
    if args.runtime_config is not None:
        args.runtime_config = args.runtime_config.expanduser().resolve()
    build_reach_task_configs("xarm6", recipe_path=args.recipe, runtime_config_path=args.runtime_config)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output
    if args.stage in ("all", "train"):
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {output}")
    status_path = output / ("experiment-status.json" if args.stage in ("all", "train") else f"{args.stage}-status.json")
    if status_path.exists():
        raise FileExistsError(f"stage already recorded: {status_path}; use a new output directory")
    configure_runtime()
    status = {
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": "running",
        "stage": args.stage,
        "training_seeds": args.seeds,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seeds": [],
    }
    write_json(status_path, status)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.stage in ("all", "train"):
            record_source_hashes(output, args)
            smoke(output, args)
        for seed in args.seeds:
            directory = output / f"seed-{seed}"
            if args.stage in ("all", "train"):
                run(
                    output,
                    f"seed-{seed}-train",
                    [
                        "-m",
                        "examples.rl.reach.train",
                        *common_options(args),
                        "--log-dir",
                        str(directory),
                        "-B",
                        str(args.num_envs),
                        "--max-iterations",
                        str(args.max_iterations),
                        "--seed",
                        str(seed),
                    ],
                )
            if args.stage in ("all", "select"):
                selected = select_checkpoint(output, directory, seed, args)
            if args.stage == "accept":
                selection = json.loads((directory / "checkpoint-selection.json").read_text())
                selected = directory / selection["selected_checkpoint"]
                if selected.parent != directory or not selected.is_file():
                    raise ValueError("invalid selected checkpoint")
            if args.stage in ("all", "accept"):
                result = accept_checkpoint(output, directory, selected, seed, args)
                status["seeds"].append(result)
                write_json(status_path, status)
                if not result["passed"]:
                    status["stop_reason"] = "acceptance_failed"
                    return 1
        status["stop_reason"] = "all_seeds_passed" if args.stage in ("all", "accept") else "completed"
        return 0
    except KeyboardInterrupt:
        status["stop_reason"] = "KeyboardInterrupt"
        return 130
    except Exception as exc:
        status.update(stop_reason=type(exc).__name__, error=str(exc))
        print(f"Experiment failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)


if __name__ == "__main__":
    sys.exit(main())
