"""Unattended pick-place campaign: search clones until the grasp gate, then PPO.

The clone gate is a from-home ``contact_v1`` smoke: grasp and lift must match
the episode count, and clipping / IK faults must stay at zero. Placement is
recorded but not gated; placement refinement is PPO's job (place-phase resets
plus grasp-degradation monitoring). PPO starts only after that gate. A seed
whose learned-from-home grasp collapses is abandoned; leftover wall-clock goes
back to clone search.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping

from examples.rl.pick_place.bc_gate import GateVerdict, judge_eval_summary_path
from examples.rl.pick_place.monitor_utils import grasp_collapse_reason, metrics_by_iteration


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "rl" / "pick_place"
SCENARIO_BANK = EXAMPLE_DIR / "scenarios" / "fixed_seed17000_n512.json"
RECIPE_PATH = EXAMPLE_DIR / "recipe.yaml"
STATUS_NAME = "overnight-status.json"
PID_NAME = "overnight.pid"
CAMPAIGN_PREFIX = "v0213e"

TRAIN_ENVS = 128
SMOKE_ENVS = 8
SMOKE_EPISODES = 8
BANK_ENVS = 64
BANK_EPISODES = 64
NOISE_EPISODES = 512
NOISE_STD = 0.02
NOISE_BANK = "20260817"
PPO_LR = 1e-5
PPO_PLACE_RESET = 0.40
PPO_SEEDS = (1, 7, 17)
RELEASE_CANDIDATE_SEED = 7
PPO_ITERS = 300
PPO_PROBE_ITERS = 10
PPO_CHUNK_ITERS = 50
GRASP_COLLAPSE_FLOOR = 0.01
NUMBA_CACHE_DIR = Path.home() / ".cache" / "numba"

# Wall-clock reserves. Genesis rebuilds kernels in every new process.
BC_BASE_SECONDS = 15 * 60
BC_DAGGER_SECONDS = 4 * 60
BC_LONG_ROLLOUT_SECONDS = 4 * 60
SMOKE_SECONDS = 10 * 60
PPO_PROBE_SECONDS = 25 * 60
PPO_CHUNK_SECONDS = 55 * 60
BANK_SECONDS = 20 * 60
GRID_B1_SECONDS = 12 * 60
NOISE_SECONDS = 50 * 60
DOCS_SECONDS = 2 * 60


@dataclass(frozen=True)
class BcRecipe:
    """One behaviour-cloning collection and fit configuration."""

    tag: str
    seed: int = 1
    rollout_steps: int = 600
    dagger_rounds: int = 0
    transport_weight: float = 4.0
    close_lift_weight: float = 1.0
    release_weight: float = 1.0
    near_table_weight: float = 1.0
    far_open_penalty: float = 20.0
    place_reset: float = 0.0
    grasp_reset: float = 0.0
    carry_reset: float = 0.0
    action_y_weight: float = 1.0
    action_gripper_weight: float = 1.0
    epochs: int = 8
    learning_rate: float = 1e-3
    phase_balanced: bool = True
    warm_start: str | None = None

    def fingerprint(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key != "tag"}
        return json.dumps(payload, sort_keys=True)


@dataclass(frozen=True)
class BcAttempt:
    """One clone fit plus its from-home smoke verdict."""

    recipe: BcRecipe
    log_dir: str
    checkpoint: str | None
    checkpoint_sha256: str | None
    returncode: int
    verdict: GateVerdict | None
    failure_mode: str

    @property
    def place(self) -> int:
        return 0 if self.verdict is None else self.verdict.place

    @property
    def grasp(self) -> int:
        return 0 if self.verdict is None else self.verdict.grasp

    @property
    def p99_xy(self) -> float:
        if self.verdict is None or self.verdict.p99_final_xy_error_m is None:
            return 1e9
        return float(self.verdict.p99_final_xy_error_m)

    @property
    def p99_release(self) -> float:
        if self.verdict is None or self.verdict.p99_release_xy_dist_m is None:
            return 1e9
        return float(self.verdict.p99_release_xy_dist_m)


def campaign_env() -> dict[str, str]:
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def remaining_seconds(deadline: float) -> float:
    return deadline - time.time()


def has_time(deadline: float, needed: float) -> bool:
    return remaining_seconds(deadline) >= needed + DOCS_SECONDS


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def find_latest_checkpoint(log_dir: Path) -> Path | None:
    points = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    return points[-1] if points else None


def estimate_bc_seconds(recipe: BcRecipe) -> float:
    extra = BC_DAGGER_SECONDS * int(recipe.dagger_rounds)
    extra += BC_LONG_ROLLOUT_SECONDS if int(recipe.rollout_steps) > 600 else 0
    return BC_BASE_SECONDS + extra + SMOKE_SECONDS


def rank_key(attempt: BcAttempt) -> tuple[int, int, float, float]:
    """Higher is better: places, grasps, then smaller XY and release distances."""

    return (attempt.place, attempt.grasp, -attempt.p99_xy, -attempt.p99_release)


def best_attempt(attempts: Iterable[BcAttempt]) -> BcAttempt | None:
    materialised = [attempt for attempt in attempts if attempt.checkpoint]
    if not materialised:
        return None
    return max(materialised, key=rank_key)


def ladder_recipes() -> list[BcRecipe]:
    faropen = BcRecipe(tag="faropen", far_open_penalty=20.0, transport_weight=4.0)
    dagger = replace(faropen, tag="dagger", dagger_rounds=1)
    dagger_place = replace(dagger, tag="dagger-place", place_reset=0.15)
    dagger_setdown = replace(dagger_place, tag="dagger-setdown", near_table_weight=2.0)
    return [faropen, dagger, dagger_place, dagger_setdown]


def warmtune_recipe(source: BcAttempt) -> BcRecipe:
    recipe = source.recipe
    return replace(
        recipe,
        tag=f"warmtune-{recipe.tag}"[:80],
        warm_start=source.checkpoint,
        learning_rate=3e-4,
        epochs=4,
        dagger_rounds=0,
    )


def _unique(recipes: Iterable[BcRecipe], tried: set[str]) -> list[BcRecipe]:
    unique: list[BcRecipe] = []
    seen = set(tried)
    for recipe in recipes:
        fingerprint = recipe.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(recipe)
    return unique


def followup_recipes(last: BcAttempt, tried: set[str], best: BcAttempt | None) -> list[BcRecipe]:
    """Propose the next clones from the last failure mode."""

    recipe = last.recipe
    mode = last.failure_mode
    candidates: list[BcRecipe] = []

    def add(**changes: Any) -> None:
        tag_bits = [mode.replace("_", "-"), f"s{changes.get('seed', recipe.seed)}"]
        tagged = replace(recipe, **changes, tag="-".join(str(bit) for bit in tag_bits))
        digest = hashlib.sha1(tagged.fingerprint().encode("utf-8")).hexdigest()[:8]
        candidates.append(replace(tagged, tag=f"{tagged.tag}-{digest}"))

    if mode == "premature_release":
        for penalty in (40.0, 80.0, 160.0):
            if penalty > recipe.far_open_penalty:
                add(far_open_penalty=penalty, release_weight=min(recipe.release_weight, 0.25))
        add(
            close_lift_weight=max(recipe.close_lift_weight, 2.0),
            release_weight=0.25,
            far_open_penalty=max(recipe.far_open_penalty, 40.0),
        )
    elif mode == "overshoot_no_release":
        if recipe.dagger_rounds < 2:
            add(dagger_rounds=recipe.dagger_rounds + 1)
        if recipe.near_table_weight < 2.0:
            add(near_table_weight=2.0)
        if recipe.place_reset < 0.15:
            add(place_reset=0.15)
        if recipe.rollout_steps < 1200:
            add(rollout_steps=1200)
        if recipe.transport_weight > 4.0:
            add(transport_weight=4.0)
        add(near_table_weight=2.0, place_reset=0.15, dagger_rounds=max(1, recipe.dagger_rounds))
    elif mode == "no_grasp":
        add(transport_weight=2.0, action_y_weight=1.0, grasp_reset=0.15, dagger_rounds=0)
        add(transport_weight=1.0, close_lift_weight=2.0, grasp_reset=0.15, far_open_penalty=20.0)
        add(transport_weight=2.0, rollout_steps=600, dagger_rounds=0, action_y_weight=1.0)
    elif mode == "missed_target":
        add(dagger_rounds=min(2, recipe.dagger_rounds + 1), near_table_weight=2.0)
        add(place_reset=0.15, far_open_penalty=max(20.0, recipe.far_open_penalty))
    elif mode == "near_miss":
        add(epochs=16, rollout_steps=1200, dagger_rounds=max(1, recipe.dagger_rounds))
    elif mode == "crash":
        add(rollout_steps=600, dagger_rounds=0, epochs=8)
    elif mode == "pass":
        return grid_recipes(tried)
    for seed in PPO_SEEDS:
        if seed != recipe.seed:
            add(seed=seed)
    followups = _unique(candidates, tried)
    if best is not None:
        tuned = warmtune_recipe(best)
        followups.extend(_unique([tuned], tried | {item.fingerprint() for item in followups}))
    return followups


def grid_recipes(tried: set[str], limit: int = 8) -> list[BcRecipe]:
    """Fill leftover time with a deterministic hyperparameter grid."""

    recipes: list[BcRecipe] = []
    for seed in PPO_SEEDS:
        for dagger in (0, 1, 2):
            for far_open in (10.0, 20.0, 40.0, 80.0):
                for transport in (2.0, 4.0, 8.0):
                    for place in (0.0, 0.15):
                        for steps in (600, 1200):
                            for near in (1.0, 2.0):
                                for release in (1.0, 0.25):
                                    recipe = BcRecipe(
                                        tag="grid",
                                        seed=seed,
                                        dagger_rounds=dagger,
                                        far_open_penalty=far_open,
                                        transport_weight=transport,
                                        place_reset=place,
                                        rollout_steps=steps,
                                        near_table_weight=near,
                                        release_weight=release,
                                    )
                                    digest = hashlib.sha1(recipe.fingerprint().encode("utf-8")).hexdigest()[:8]
                                    recipe = replace(recipe, tag=f"grid-{digest}")
                                    if recipe.fingerprint() in tried:
                                        continue
                                    recipes.append(recipe)
                                    if len(recipes) >= limit:
                                        return recipes
    return recipes


def next_recipes(
    *,
    attempts: list[BcAttempt],
    tried: set[str],
    ladder_done: bool,
) -> list[BcRecipe]:
    if not ladder_done:
        remaining = [recipe for recipe in ladder_recipes() if recipe.fingerprint() not in tried]
        if remaining:
            return remaining[:1]
        best = best_attempt(attempts)
        if best is not None:
            tuned = _unique([warmtune_recipe(best)], tried)
            if tuned:
                return tuned
    if attempts:
        followups = followup_recipes(attempts[-1], tried, best_attempt(attempts))
        if followups:
            return followups[:3]
    return grid_recipes(tried)


def bc_command(recipe: BcRecipe, log_dir: Path, python: str) -> list[str]:
    command = [
        python,
        "-m",
        "examples.rl.pick_place.pretrain_bc",
        "-B",
        str(TRAIN_ENVS),
        "--rollout-steps",
        str(int(recipe.rollout_steps)),
        "--dagger-rounds",
        str(int(recipe.dagger_rounds)),
        "--seed",
        str(int(recipe.seed)),
        "--epochs",
        str(int(recipe.epochs)),
        "--learning-rate",
        str(float(recipe.learning_rate)),
        "--transport-phase-weight",
        str(float(recipe.transport_weight)),
        "--close-lift-phase-weight",
        str(float(recipe.close_lift_weight)),
        "--release-phase-weight",
        str(float(recipe.release_weight)),
        "--near-table-phase-weight",
        str(float(recipe.near_table_weight)),
        "--action-y-weight",
        str(float(recipe.action_y_weight)),
        "--action-gripper-weight",
        str(float(recipe.action_gripper_weight)),
        "--far-open-penalty",
        str(float(recipe.far_open_penalty)),
        "--grasp-phase-reset-frac",
        str(float(recipe.grasp_reset)),
        "--carry-phase-reset-frac",
        str(float(recipe.carry_reset)),
        "--place-phase-reset-frac",
        str(float(recipe.place_reset)),
        "--log-dir",
        str(log_dir),
        "-e",
        log_dir.name,
        "--recipe",
        str(RECIPE_PATH),
    ]
    if recipe.phase_balanced:
        command.append("--phase-balanced-weighting")
    if recipe.warm_start:
        command.extend(["--warm-start", str(recipe.warm_start)])
    return command


def eval_command(
    *,
    python: str,
    checkpoint: Path,
    summary_json: Path,
    num_envs: int,
    episodes: int,
    seed: int | None = None,
    scenario_bank: Path | None = None,
    require_target: str | None = None,
    action_noise_std: float = 0.0,
    action_noise_bank: str | None = None,
) -> list[str]:
    command = [
        python,
        "-m",
        "examples.rl.pick_place.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--headless",
        "-B",
        str(int(num_envs)),
        "--episodes",
        str(int(episodes)),
        "--acceptance-profile",
        "contact_v1",
        "--stage",
        "0",
        "--summary-json",
        str(summary_json),
        "--grasp-phase-reset-frac",
        "0",
        "--carry-phase-reset-frac",
        "0",
        "--place-phase-reset-frac",
        "0",
    ]
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    if scenario_bank is not None:
        command.extend(["--scenario-bank", str(scenario_bank)])
    if require_target is not None:
        command.extend(["--require-target", require_target])
    if action_noise_std > 0.0:
        command.extend(["--action-noise-std", str(float(action_noise_std))])
        if action_noise_bank is not None:
            command.extend(["--action-noise-bank", str(action_noise_bank)])
    return command


def train_command(
    *,
    python: str,
    exp_name: str,
    log_dir: Path,
    seed: int,
    max_iterations: int,
    warm_start: Path | None,
    resume: Path | None,
) -> list[str]:
    command = [
        python,
        "-m",
        "examples.rl.pick_place.train",
        "-B",
        str(TRAIN_ENVS),
        "--max-iterations",
        str(int(max_iterations)),
        "--seed",
        str(int(seed)),
        "--learning-rate",
        str(PPO_LR),
        "--log-dir",
        str(log_dir),
        "-e",
        exp_name,
        "--recipe",
        str(RECIPE_PATH),
    ]
    if warm_start is not None:
        command.extend(["--warm-start", str(warm_start)])
    if resume is not None:
        command.extend(["--resume", str(resume)])
    # Placement starts from table-ready holds (v0.2.12-validated magnitude): the
    # clone already carries, so PPO spends its budget learning the set-down.
    command.extend(["--place-phase-reset-frac", str(PPO_PLACE_RESET)])
    return command


def nvidia_snapshot() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"
    return (completed.stdout or completed.stderr or "").strip()


def run_logged(command: list[str], log_path: Path, *, timeout: float | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {iso_now()} {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=campaign_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return int(completed.returncode)


def terminate_process_group(proc: subprocess.Popen[Any], *, grace_seconds: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        proc.terminate()
    deadline = time.time() + grace_seconds
    while proc.poll() is None and time.time() < deadline:
        time.sleep(1.0)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_learned_grasp(metrics_path: Path) -> float | None:
    rows = read_metrics_rows(metrics_path)
    grouped = metrics_by_iteration(rows) if rows and rows[0].keys() >= {"training_iteration"} else {}
    if not grouped:
        return None
    latest = max(grouped)
    values = [
        float(row["learned_grasp_success_rate"]) for row in grouped[latest] if row.get("learned_grasp_success_rate")
    ]
    if not values:
        return None
    return sum(values) / len(values)


def collapse_reason(metrics_path: Path, *, grace_iterations: int) -> str | None:
    rows = read_metrics_rows(metrics_path)
    if not rows:
        return None
    try:
        grouped = metrics_by_iteration(rows)
    except ValueError:
        return None
    return grasp_collapse_reason(
        grouped,
        floor=GRASP_COLLAPSE_FLOOR,
        grace_iterations=grace_iterations,
        failure_iterations=3 if grace_iterations <= 10 else 5,
    )


class OvernightCampaign:
    """Drive clone search and optional PPO until the wall-clock deadline."""

    def __init__(self, *, python: str, deadline: float, output_root: Path) -> None:
        self.python = python
        self.deadline = deadline
        self.output_root = output_root
        self.status_path = output_root / STATUS_NAME
        self.log_path = output_root / "overnight.log"
        self.attempts: list[BcAttempt] = []
        self.evals: list[dict[str, Any]] = []
        self.ppo_runs: list[dict[str, Any]] = []
        self.phase = "starting"
        self.note = ""
        self.passed_clone: BcAttempt | None = None
        self.started_at = iso_now()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{iso_now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def status_payload(self) -> dict[str, Any]:
        best = best_attempt(self.attempts)
        remaining = max(0.0, remaining_seconds(self.deadline))
        return {
            "phase": self.phase,
            "note": self.note,
            "started_at": self.started_at,
            "updated_at": iso_now(),
            "deadline_unix": self.deadline,
            "remaining_seconds": remaining,
            "remaining_hours": remaining / 3600.0,
            "python": self.python,
            "gpu": nvidia_snapshot(),
            "bc_gate": "contact_v1 from-home 8/8 grasp+lift, zero clip/IK; place recorded",
            "passed_clone": None if self.passed_clone is None else _attempt_dict(self.passed_clone),
            "best_clone": None if best is None else _attempt_dict(best),
            "attempts": [_attempt_dict(attempt) for attempt in self.attempts],
            "ppo_runs": self.ppo_runs,
            "evals": self.evals,
        }

    def flush_status(self) -> None:
        write_json(self.status_path, self.status_payload())

    def unique_dir(self, name: str) -> Path:
        path = self.output_root / name
        if not path.exists():
            return path
        stamp = datetime.now().strftime("%H%M%S")
        return self.output_root / f"{name}-{stamp}"

    def run_eval(
        self,
        checkpoint: Path,
        *,
        label: str,
        num_envs: int,
        episodes: int,
        seed: int | None = None,
        scenario_bank: Path | None = None,
        require_target: str | None = None,
        action_noise_std: float = 0.0,
        action_noise_bank: str | None = None,
        expected_episodes: int | None = None,
    ) -> GateVerdict | None:
        summary = checkpoint.parent / f"{checkpoint.stem}-{label}-summary.json"
        log_path = checkpoint.parent / f"{checkpoint.stem}-{label}.log"
        command = eval_command(
            python=self.python,
            checkpoint=checkpoint,
            summary_json=summary,
            num_envs=num_envs,
            episodes=episodes,
            seed=seed,
            scenario_bank=scenario_bank,
            require_target=require_target,
            action_noise_std=action_noise_std,
            action_noise_bank=action_noise_bank,
        )
        self.log(f"eval {label}: {' '.join(command)}")
        code = run_logged(command, log_path)
        if not summary.is_file():
            self.log(f"eval {label} produced no summary (exit {code})")
            self.evals.append({"label": label, "checkpoint": str(checkpoint), "returncode": code, "summary": None})
            self.flush_status()
            return None
        verdict = judge_eval_summary_path(
            summary,
            expected_episodes=episodes if expected_episodes is None else expected_episodes,
        )
        self.evals.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "returncode": code,
                "summary": str(summary),
                "verdict": verdict.to_json_dict(),
            }
        )
        self.flush_status()
        return verdict

    def run_bc(self, recipe: BcRecipe) -> BcAttempt:
        log_dir = self.unique_dir(f"{CAMPAIGN_PREFIX}-bc-{recipe.tag}")
        train_log = log_dir.parent / f"{log_dir.name}-train.log"
        self.phase = f"bc:{recipe.tag}"
        self.note = recipe.fingerprint()
        self.flush_status()
        self.log(f"BC {recipe.tag} -> {log_dir}")
        code = run_logged(bc_command(recipe, log_dir, self.python), train_log)
        checkpoint = log_dir / "model_0.pt"
        sha = sha256_file(checkpoint) if checkpoint.is_file() else None
        verdict: GateVerdict | None = None
        failure_mode = "crash"
        if checkpoint.is_file():
            verdict = self.run_eval(
                checkpoint,
                label="smoke",
                num_envs=SMOKE_ENVS,
                episodes=SMOKE_EPISODES,
                expected_episodes=SMOKE_EPISODES,
            )
            if verdict is not None:
                failure_mode = verdict.failure_mode
        elif code == 0:
            failure_mode = "crash"
        attempt = BcAttempt(
            recipe=recipe,
            log_dir=str(log_dir),
            checkpoint=str(checkpoint) if checkpoint.is_file() else None,
            checkpoint_sha256=sha,
            returncode=code,
            verdict=verdict,
            failure_mode=failure_mode,
        )
        verdict_path = log_dir / "bc-gate-verdict.json"
        write_json(verdict_path, _attempt_dict(attempt))
        self.attempts.append(attempt)
        self.flush_status()
        return attempt

    def search_clones(self) -> BcAttempt | None:
        tried = {attempt.recipe.fingerprint() for attempt in self.attempts}
        ladder_tags = {recipe.tag for recipe in ladder_recipes()}
        while has_time(self.deadline, BC_BASE_SECONDS + SMOKE_SECONDS):
            already_passed = self.passed_clone is not None
            ladder_present = ladder_tags <= {attempt.recipe.tag for attempt in self.attempts}
            warmtuned = any(attempt.recipe.warm_start for attempt in self.attempts)
            ladder_done = already_passed or (ladder_present and warmtuned)
            recipes = next_recipes(attempts=self.attempts, tried=tried, ladder_done=ladder_done)
            if not recipes:
                self.note = "clone search space exhausted"
                self.flush_status()
                break
            recipe = recipes[0]
            if not has_time(self.deadline, estimate_bc_seconds(recipe)):
                self.note = "not enough time for the next clone"
                self.flush_status()
                break
            tried.add(recipe.fingerprint())
            attempt = self.run_bc(recipe)
            if attempt.verdict is not None and attempt.verdict.passed:
                self.passed_clone = attempt
                return attempt
        return self.passed_clone

    def run_ppo_process(
        self, command: list[str], log_path: Path, metrics_path: Path, *, grace: int
    ) -> tuple[int, str | None]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# {iso_now()} {' '.join(command)}\n")
            handle.flush()
            proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=campaign_env(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            reason = None
            try:
                while proc.poll() is None:
                    if not has_time(self.deadline, 30):
                        reason = "deadline"
                        terminate_process_group(proc)
                        break
                    collapsed = collapse_reason(metrics_path, grace_iterations=grace)
                    if collapsed:
                        reason = collapsed
                        terminate_process_group(proc)
                        break
                    time.sleep(30)
            except Exception:
                terminate_process_group(proc)
                raise
            code = proc.wait()
            return int(code), reason

    def run_ppo_seed(self, seed: int, warm_start: Path) -> dict[str, Any]:
        probe_dir = self.unique_dir(f"{CAMPAIGN_PREFIX}-ppo-probe-s{seed}")
        probe_log = probe_dir.parent / f"{probe_dir.name}.log"
        self.phase = f"ppo-probe:{seed}"
        self.flush_status()
        probe_cmd = train_command(
            python=self.python,
            exp_name=probe_dir.name,
            log_dir=probe_dir,
            seed=seed,
            max_iterations=PPO_PROBE_ITERS,
            warm_start=warm_start,
            resume=None,
        )
        self.log(f"PPO probe seed {seed}")
        code, reason = self.run_ppo_process(probe_cmd, probe_log, probe_dir / "metrics.csv", grace=5)
        probe_ckpt = find_latest_checkpoint(probe_dir)
        record: dict[str, Any] = {
            "seed": seed,
            "probe_dir": str(probe_dir),
            "probe_returncode": code,
            "probe_stop": reason,
            "probe_checkpoint": None if probe_ckpt is None else str(probe_ckpt),
            "learned_grasp": latest_learned_grasp(probe_dir / "metrics.csv"),
        }
        if probe_ckpt is None:
            record["status"] = "probe_failed"
            self.ppo_runs.append(record)
            self.flush_status()
            return record
        smoke = self.run_eval(
            probe_ckpt,
            label="probe-smoke",
            num_envs=SMOKE_ENVS,
            episodes=SMOKE_EPISODES,
        )
        record["probe_smoke"] = None if smoke is None else smoke.to_json_dict()
        grasp_ok = (
            reason is None
            and smoke is not None
            and smoke.grasp > 0
            and (record["learned_grasp"] is None or float(record["learned_grasp"]) > GRASP_COLLAPSE_FLOOR)
        )
        if not grasp_ok:
            record["status"] = "probe_grasp_collapsed"
            self.ppo_runs.append(record)
            self.flush_status()
            return record

        current_ckpt = probe_ckpt
        current_iter = PPO_PROBE_ITERS
        while current_iter < PPO_ITERS and has_time(self.deadline, PPO_CHUNK_SECONDS + SMOKE_SECONDS):
            target = min(PPO_ITERS, current_iter + PPO_CHUNK_ITERS)
            chunk_dir = self.unique_dir(f"{CAMPAIGN_PREFIX}-ppo-s{seed}-i{target}")
            chunk_log = chunk_dir.parent / f"{chunk_dir.name}.log"
            self.phase = f"ppo:{seed}:{target}"
            self.flush_status()
            chunk_cmd = train_command(
                python=self.python,
                exp_name=chunk_dir.name,
                log_dir=chunk_dir,
                seed=seed,
                max_iterations=target,
                warm_start=None,
                resume=current_ckpt,
            )
            code, reason = self.run_ppo_process(
                chunk_cmd,
                chunk_log,
                chunk_dir / "metrics.csv",
                grace=10,
            )
            latest = find_latest_checkpoint(chunk_dir) or current_ckpt
            smoke = self.run_eval(
                latest,
                label=f"iter{target}-smoke",
                num_envs=SMOKE_ENVS,
                episodes=SMOKE_EPISODES,
            )
            record.setdefault("chunks", []).append(
                {
                    "dir": str(chunk_dir),
                    "target": target,
                    "returncode": code,
                    "stop": reason,
                    "checkpoint": str(latest),
                    "learned_grasp": latest_learned_grasp(chunk_dir / "metrics.csv"),
                    "smoke": None if smoke is None else smoke.to_json_dict(),
                }
            )
            if reason or smoke is None or smoke.grasp == 0:
                record["status"] = "grasp_collapsed"
                record["final_checkpoint"] = str(latest)
                self.ppo_runs.append(record)
                self.flush_status()
                return record
            current_ckpt = latest
            current_iter = target

        record["status"] = "completed" if current_iter >= PPO_ITERS else "time_stop"
        record["final_checkpoint"] = str(current_ckpt)
        self.ppo_runs.append(record)
        self.flush_status()
        return record

    def run_release_evals(self, checkpoint: Path) -> None:
        """Evaluate one selected checkpoint across the complete fixed-layout matrix."""

        if has_time(self.deadline, SMOKE_SECONDS):
            self.run_eval(checkpoint, label="final-smoke", num_envs=SMOKE_ENVS, episodes=SMOKE_EPISODES)
        if has_time(self.deadline, BANK_SECONDS):
            self.run_eval(
                checkpoint,
                label="bank64",
                num_envs=BANK_ENVS,
                episodes=BANK_EPISODES,
                scenario_bank=SCENARIO_BANK,
                require_target="standard",
            )
        for seed in PPO_SEEDS:
            for batch in (1, 8, 64):
                needed = GRID_B1_SECONDS if batch == 1 else (SMOKE_SECONDS if batch == 8 else BANK_SECONDS)
                if not has_time(self.deadline, needed):
                    return
                self.run_eval(
                    checkpoint,
                    label=f"grid-s{seed}-b{batch}",
                    num_envs=batch,
                    episodes=batch,
                    seed=seed,
                )
        if has_time(self.deadline, NOISE_SECONDS):
            self.run_eval(
                checkpoint,
                label="noise512",
                num_envs=BANK_ENVS,
                episodes=NOISE_EPISODES,
                scenario_bank=SCENARIO_BANK,
                require_target="robustness",
                action_noise_std=NOISE_STD,
                action_noise_bank=NOISE_BANK,
                expected_episodes=NOISE_EPISODES,
            )

    def run(self) -> int:
        self.phase = "bc-search"
        self.flush_status()
        update_campaign_docs(self.status_payload(), started=True)
        passed = self.search_clones()
        if passed is None or passed.checkpoint is None:
            self.phase = "bc-failed"
            self.note = "no clone passed the grasp+lift gate; PPO not started"
            self.flush_status()
            update_campaign_docs(self.status_payload(), started=False)
            return 1
        original_clone = passed
        actor_path = Path(passed.checkpoint)
        release_checkpoint: Path | None = None
        for seed in PPO_SEEDS:
            if not has_time(self.deadline, PPO_PROBE_SECONDS):
                break
            if self.passed_clone is None or self.passed_clone.checkpoint is None:
                break
            result = self.run_ppo_seed(seed, actor_path)
            if result.get("status") == "probe_grasp_collapsed" and seed == PPO_SEEDS[0]:
                self.phase = "ppo-probe-failed"
                self.note = "seed 1 probe lost learned grasp; returning to clone search"
                self.flush_status()
                extra = self.search_clones()
                if extra is not None and extra.checkpoint is not None and extra is not original_clone:
                    actor_path = Path(extra.checkpoint)
                    original_clone = extra
                    result = self.run_ppo_seed(seed, actor_path)
                else:
                    break
            if result.get("status") == "probe_grasp_collapsed":
                if seed == PPO_SEEDS[0]:
                    break
                continue
            final = result.get("final_checkpoint") or result.get("probe_checkpoint")
            if final and seed == RELEASE_CANDIDATE_SEED:
                release_checkpoint = Path(final)
        if release_checkpoint is not None:
            self.run_release_evals(release_checkpoint)
        self.phase = "done"
        self.note = "campaign finished"
        self.flush_status()
        update_campaign_docs(self.status_payload(), started=False)
        return 0 if self.passed_clone is not None else 1


def _attempt_dict(attempt: BcAttempt) -> dict[str, Any]:
    return {
        "tag": attempt.recipe.tag,
        "recipe": asdict(attempt.recipe),
        "log_dir": attempt.log_dir,
        "checkpoint": attempt.checkpoint,
        "checkpoint_sha256": attempt.checkpoint_sha256,
        "returncode": attempt.returncode,
        "failure_mode": attempt.failure_mode,
        "verdict": None if attempt.verdict is None else attempt.verdict.to_json_dict(),
    }


def _best_summary_lines(status: Mapping[str, Any]) -> list[str]:
    best = status.get("best_clone") or {}
    passed = status.get("passed_clone")
    sha = str(best.get("checkpoint_sha256") or "")
    sha_short = f"{sha[:8]}…{sha[-5:]}" if len(sha) >= 13 else (sha or "none")
    verdict = best.get("verdict") or {}
    grasp = verdict.get("grasp", 0)
    lift = verdict.get("lift", 0)
    place = verdict.get("place", 0)
    xy = verdict.get("p99_final_xy_error_m")
    xy_mm = "n/a" if xy is None else f"{float(xy) * 1000:.0f} mm"
    tag = best.get("tag", "none")
    acceptance_result = "抓取和抬升达到要求" if passed else "抓取或抬升未达到要求"
    ppo = status.get("ppo_runs") or []
    ppo_note = "未开 PPO" if not ppo else "、".join(f"种子{item.get('seed')} {item.get('status')}" for item in ppo)
    return [
        f"自适应行为克隆验收 `{tag}`：抓取 {grasp}/8，抬升 {lift}/8，放置 {place}/8（仅记录，不影响是否继续训练），误差 {xy_mm}，{sha_short}，{acceptance_result}",
        f"阶段 {status.get('phase')}。{ppo_note}。公开模型由独立发布检查决定",
    ]


def _replace_section(text: str, heading: str, body: str) -> str:
    start = text.find(heading)
    if start < 0:
        return text.rstrip() + "\n\n" + heading + "\n\n" + body + "\n"
    rest = text[start + len(heading) :]
    next_head = rest.find("\n## ")
    prefix = text[: start + len(heading)]
    suffix = "" if next_head < 0 else rest[next_head:]
    return prefix + "\n\n" + body.strip() + "\n" + suffix


def update_campaign_docs(status: Mapping[str, Any], *, started: bool) -> None:
    """Rewrite the local campaign notes; public checkpoint rules stay unchanged."""

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    lines = _best_summary_lines(status)
    if started:
        title = "v0.2.13 启动 15 小时行为克隆验收，达到要求后启动 PPO"
        bullets = [
            "验收标准仍是 contact_v1，从家基础测试抓取与抬升须 8/8，放置只记录；未达到要求则不启动 PPO，达到要求后以 1e-5 学习率试训 10 轮，再完成三个种子各 300 轮",
            "状态文件 `outputs/rl/pick_place/overnight-status.json`；不提交、不降低验收标准、不加载 Genesis 1.3.1 权重",
        ]
    else:
        title = "v0.2.13 15 小时行为克隆验收结束"
        bullets = lines

    worklog = REPO_ROOT / "worklog.md"
    if worklog.is_file():
        current = worklog.read_text(encoding="utf-8")
        marker = f"## {stamp} — {title}"
        if marker not in current:
            entry = marker + "\n\n" + "\n".join(f"- {item}" for item in bullets) + "\n"
            needle = "\n## "
            idx = current.find(needle)
            if idx >= 0:
                current = current[: idx + 1] + entry + "\n" + current[idx + 1 :]
            else:
                current = current.rstrip() + "\n\n" + entry + "\n"
            worklog.write_text(current, encoding="utf-8")
    if started:
        return

    cn_body = (
        "当前脚本专家完整与质量成功 8/8，固定库 64/64。配方将专家参数写入\n"
        "`scripted_action_hint_config`：锁存路点、夹爪 22 mm、远离桌子快走、只在近桌刹车。\n"
        "15 小时无人值守行为克隆验收结果：" + "。".join(lines) + "。\n"
        "本工具只处理固定起点；公开模型由独立发布检查决定。"
    )
    en_body = (
        "The current scripted expert passed 8/8 full and quality episodes and 64/64 on the\n"
        "fixed bank. A 15-hour unattended clone gate ran from home under `contact_v1`.\n"
        + " ".join(lines)
        + "\nThis tool handles fixed starts only; the public model is decided by the separate release checks."
    )
    cn_path = EXAMPLE_DIR / "README_cn.md"
    en_path = EXAMPLE_DIR / "README.md"
    if cn_path.is_file():
        cn_path.write_text(
            _replace_section(cn_path.read_text(encoding="utf-8"), "## 当前 v0.2.13 训练", cn_body),
            encoding="utf-8",
        )
    if en_path.is_file():
        en_path.write_text(
            _replace_section(en_path.read_text(encoding="utf-8"), "## Current v0.2.13 campaign", en_body),
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search pick-place clones until 8/8, then PPO")
    parser.add_argument("--deadline-hours", type=float, default=15.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Rewrite overnight-status.json from current fields without launching Genesis",
    )
    args = parser.parse_args(argv)
    if args.deadline_hours <= 0.0:
        parser.error("--deadline-hours must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    pid_path = output_root / PID_NAME
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.time() + float(args.deadline_hours) * 3600.0
    campaign = OvernightCampaign(python=str(args.python), deadline=deadline, output_root=output_root)
    campaign.log(f"overnight start python={args.python} hours={args.deadline_hours} gpu={nvidia_snapshot()}")
    try:
        if args.status_only:
            campaign.flush_status()
            return 0
        return campaign.run()
    except Exception:
        campaign.phase = "crashed"
        campaign.note = traceback.format_exc()
        campaign.flush_status()
        update_campaign_docs(campaign.status_payload(), started=False)
        raise
    finally:
        if pid_path.is_file():
            pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
