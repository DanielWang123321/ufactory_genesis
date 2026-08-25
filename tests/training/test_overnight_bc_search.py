"""CPU tests for the unattended clone-search recipe ladder."""

from __future__ import annotations

from pathlib import Path
import time

from examples.rl.pick_place.bc_gate import GateVerdict
from examples.rl.pick_place.overnight import (
    BcAttempt,
    BcRecipe,
    OvernightCampaign,
    RELEASE_CANDIDATE_SEED,
    best_attempt,
    bc_command,
    followup_recipes,
    ladder_recipes,
    next_recipes,
    warmtune_recipe,
)


def _verdict(*, grasp: int, place: int, xy: float, release: float, mode: str) -> GateVerdict:
    # Gate semantics since the F3 rearrangement: grasp+lift decide, place is recorded.
    return GateVerdict(
        passed=grasp == 8,
        reasons=(f"place {place}/8",),
        failure_mode=mode,
        acceptance_profile="contact_v1",
        episodes=8,
        grasp=grasp,
        lift=grasp,
        place=place,
        p99_final_xy_error_m=xy,
        p99_release_xy_dist_m=release,
        max_action_clip_fraction=0.0,
        max_ik_failure_fraction=0.0,
        max_ik_jump_reject_fraction=0.0,
    )


def _attempt(recipe: BcRecipe, verdict: GateVerdict) -> BcAttempt:
    return BcAttempt(
        recipe=recipe,
        log_dir=f"outputs/rl/pick_place/v0213e-bc-{recipe.tag}",
        checkpoint=f"outputs/rl/pick_place/v0213e-bc-{recipe.tag}/model_0.pt",
        checkpoint_sha256="abc",
        returncode=0,
        verdict=verdict,
        failure_mode=verdict.failure_mode,
    )


def test_ladder_starts_with_from_home_transport_and_far_open() -> None:
    first = ladder_recipes()[0]
    assert first.tag == "faropen"
    assert first.transport_weight == 4.0
    assert first.far_open_penalty == 20.0
    assert first.place_reset == 0.0
    assert first.grasp_reset == 0.0
    assert first.dagger_rounds == 0
    command = bc_command(first, Path("/tmp/v0213e-bc-faropen"), "python")
    assert "--near-table-phase-weight" in command
    assert command[command.index("--near-table-phase-weight") + 1] == "1.0"
    assert command[command.index("--place-phase-reset-frac") + 1] == "0.0"
    assert command[command.index("--grasp-phase-reset-frac") + 1] == "0.0"
    assert "--phase-balanced-weighting" in command
    assert "-B" in command and command[command.index("-B") + 1] == "128"


def test_next_recipes_walk_ladder_then_warmtune() -> None:
    recipes = ladder_recipes()
    first = next_recipes(attempts=[], tried=set(), ladder_done=False)
    assert first[0].tag == "faropen"
    tried = {recipes[0].fingerprint()}
    second = next_recipes(attempts=[], tried=tried, ladder_done=False)
    assert second[0].tag == "dagger"
    assert second[0].dagger_rounds == 1
    tried.add(recipes[1].fingerprint())
    third = next_recipes(attempts=[], tried=tried, ladder_done=False)
    assert third[0].tag == "dagger-place"
    assert third[0].place_reset == 0.15
    tried.add(recipes[2].fingerprint())
    fourth = next_recipes(attempts=[], tried=tried, ladder_done=False)
    assert fourth[0].tag == "dagger-setdown"
    assert fourth[0].near_table_weight == 2.0


def test_premature_release_raises_far_open_without_repeating() -> None:
    recipe = ladder_recipes()[0]
    attempt = _attempt(
        recipe,
        _verdict(grasp=8, place=0, xy=0.198, release=0.285, mode="premature_release"),
    )
    followups = followup_recipes(attempt, {recipe.fingerprint()}, attempt)
    assert followups
    assert all(item.fingerprint() != recipe.fingerprint() for item in followups)
    assert any(item.far_open_penalty > recipe.far_open_penalty for item in followups)


def test_best_attempt_prefers_more_places_then_smaller_error() -> None:
    weak = _attempt(
        BcRecipe(tag="a"),
        _verdict(grasp=8, place=0, xy=0.20, release=0.0, mode="overshoot_no_release"),
    )
    better = _attempt(
        BcRecipe(tag="b", far_open_penalty=40.0),
        _verdict(grasp=8, place=1, xy=0.05, release=0.04, mode="near_miss"),
    )
    wider = _attempt(
        BcRecipe(tag="c"),
        _verdict(grasp=8, place=1, xy=0.09, release=0.04, mode="near_miss"),
    )
    assert best_attempt([weak, better, wider]) is better


def test_warmtune_uses_best_checkpoint_and_lower_lr() -> None:
    attempt = _attempt(
        BcRecipe(tag="faropen", far_open_penalty=20.0),
        _verdict(grasp=8, place=0, xy=0.20, release=0.0, mode="overshoot_no_release"),
    )
    tuned = warmtune_recipe(attempt)
    assert tuned.warm_start == attempt.checkpoint
    assert tuned.learning_rate == 3e-4
    assert tuned.epochs == 4
    assert tuned.dagger_rounds == 0


def test_release_evals_use_one_selected_checkpoint_for_the_full_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign = OvernightCampaign(
        python="python",
        deadline=time.time() + 24 * 60 * 60,
        output_root=tmp_path,
    )
    calls: list[dict] = []

    def record_eval(checkpoint: Path, **kwargs):
        calls.append({"checkpoint": checkpoint, **kwargs})
        return None

    monkeypatch.setattr(campaign, "run_eval", record_eval)
    checkpoint = tmp_path / "model_299.pt"
    campaign.run_release_evals(checkpoint)

    assert RELEASE_CANDIDATE_SEED == 7
    grid = [call for call in calls if call["label"].startswith("grid-")]
    assert [(call["seed"], call["num_envs"], call["episodes"]) for call in grid] == [
        (seed, batch, batch) for seed in (1, 7, 17) for batch in (1, 8, 64)
    ]
    assert {call["checkpoint"] for call in calls} == {checkpoint}
    assert sum(call["label"] == "bank64" for call in calls) == 1
    assert sum(call["label"] == "noise512" for call in calls) == 1


def test_overnight_helper_never_modifies_maintainer_version_plan() -> None:
    source = Path("examples/rl/pick_place/overnight.py").read_text(encoding="utf-8")
    assert "versions.md" not in source
