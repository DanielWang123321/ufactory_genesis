# 2. Pick & Place: xArm6 + Gripper G2

This second RL example uses Gripper G2 to grasp, lift and transfer a cube, then
set it down and release it at a tabletop target. It runs in simulation with one
fixed `+Y` layout. Random-layout generalization, CPU/Windows training and
real-robot policy deployment are outside its scope.

[中文说明](README_cn.md) · [RL examples](../README.md) · [Previous: 1. Reach](../reach/README.md)

## Reference environment and artifact status

Use Linux, an NVIDIA GPU, Python 3.12 or 3.13, Genesis World 1.4.4, Quadrants 1.3.3,
PyTorch 2.10, and RSL-RL 5.4.2. The repository lock file is the reference:

```bash
uv sync --extra sim --extra rl
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
```

The environment always resolves 32 rigid substeps from `simulation.substeps`; the
removed task-local fields are invalid. This 0.625 ms rigid-step limit is the
project's conservative production margin, not a claim that Genesis 1.4.4 always
needs 32 substeps. The primary numerical fix is the compliant solver tuple applied
to all five G2 mimic equalities; it also passed the 128-environment perturbed-contact
check at 8 substeps.

The release-intent hysteresis (`release_command_margin_m`, 18 mm in the recipe)
requires the commanded gripper gap to climb well above the close gap before a
release latches. Checkpoints saved before this change evaluate with
`--release-command-margin-m 0.0005`.

The current physics profile is `g2_calibrated_v1`: `kp=40`, `kv=5`, `ki=40`,
0.01 kg·m² gripper joint armature and a 1.25 ms contact time constant. RL keeps
`noslip_iterations: 0` (deterministic pick-place uses 5) because the no-slip pass
makes the 20480-environment batch about 7× slower; held cubes therefore creep
about 2.4 mm/s in the pad frame during training and evaluation. The policy
keeps its existing position action, observation dimensions and reward functions;
`simulation.g2` supplies speed and force percentages. See the [G2 guide](../../gripper_g2/README.md).

Bundled seed-7 `model_299_g2stable.pt` (SHA-256 `f2142a63…f783f2`) remains
unchanged under historical `g2_stable_v1_3_3` physics. Evaluating it requires
`--allow-physics-transfer`; original config and checkpoint hashes are still checked.
Older-physics checkpoints cannot fully resume training. Published historical success
rates do not describe the new dynamics.

## Layered guidance disclosure

The observation includes four scripted guide-action columns (44 -> 48, recipe flag
`include_scripted_action_hint`). The guide is the same hashed scripted expert that
labels behaviour-cloning demonstrations, so observations and labels can never
disagree. This is a disclosed layered/simulator-guided policy in the v0.2.12
precedent: the published artifact must never be presented as a learned policy that
generalizes without the guide. A 2026-08-21 diagnosis (`rl/reports/`) showed the
44-dimensional observation cannot distinguish "holding the cube at the pickup" from
"holding the cube at the target", which made the clone open its gripper at the
pickup; the guide column resolves that ambiguity.

## Fresh training path

First prove that the scripted expert reaches the immutable `contact_v1` target:

```bash
python -m examples.rl.pick_place.evaluate --expert --headless -B 8 --episodes 8 \
  --acceptance-profile contact_v1 --require-target robustness
```

Collect new demonstrations and behavior-clone them without a warm start. The
standard collection tier is 256 environments x 8000 steps (about 2.05 million
samples); smaller tiers fit the action mean better but collapse closed-loop:

```bash
python -m examples.rl.pick_place.pretrain_bc -B 256 --rollout-steps 8000 \
  --far-open-penalty 20.0 --seed 1 -e v0213g-fixed-bc-seed1
```

Run three independent 300-iteration PPO jobs from that new BC actor:

```bash
for seed in 1 7 17; do
  python -m examples.rl.pick_place.train -B 256 --max-iterations 300 \
    --seed "$seed" \
    --warm-start outputs/rl/pick_place/v0213g-fixed-bc-seed1/model_0.pt \
    -e "v0213g-fixed-ppo-seed${seed}"
done
```

Outputs stay under ignored `outputs/rl/pick_place/`. Do not use `pretrained/` as a
training directory and do not warm-start from a checkpoint trained under an older
physics baseline.

## Evaluate the expert or an explicit current-profile candidate

A complete bundle consists of `model_N.pt`, `config.yaml`, and
`model_N.checkpoint_manifest.json`. Hashes, task/robot identity, and the complete
runtime contract are checked before PyTorch's safe weights-only load.

```bash
# Current physical-contact baseline
python -m examples.rl.pick_place.evaluate --expert \
  --headless -B 8 --episodes 8 --acceptance-profile contact_v1

# Local complete bundle
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/<run>/model_N.pt \
  --headless -B 8 --episodes 8 --acceptance-profile contact_v1
```

The archived bank remains bound to runtime config `654b538e…7ed` (SHA-256
`2238202c…eb0`). Generate a separate bank for new dynamics:

```bash
python -m examples.rl.pick_place.make_scenario_bank --mode fixed --count 512 --seed 17000 \
  --output reports/g2/fixed_seed17000_n512.json
```

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/<run>/model_N.pt \
  --scenario-bank reports/g2/fixed_seed17000_n512.json \
  --headless -B 64 --episodes 64 --acceptance-profile contact_v1 \
  --require-target standard
```

The formal action-noise trial uses standard deviation `0.02` and the new bank seed
`20260817`:

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/<run>/model_N.pt \
  --scenario-bank reports/g2/fixed_seed17000_n512.json \
  --headless -B 64 --episodes 512 --acceptance-profile contact_v1 \
  --action-noise-std 0.02 --action-noise-bank 20260817 \
  --require-target robustness
```

## Release gates

A candidate must be a nonzero PPO checkpoint and satisfy all of the following with
zero action clipping, IK failures, and IK jump rejects:

- seeds `1/7/17` × batch sizes `1/8/64`: 9/9 runs and 219/219 episodes;
- an independent fixed-bank run: 64/64 full and quality success;
- 512 fixed-layout action-noise episodes: at least 99% full and quality success;
- P99 final XY error ≤10 mm, pre-lift drag ≤5 mm, and post-release drift ≤3 mm.

`Full success` includes grasp, lift, valid set-down/release, final pose and velocity,
hold time, and every selected quality limit. A failed screening run is not promoted
to a published artifact, even if training metrics or partial task stages look good.

## Historical release result (`g2_stable_v1_3_3`)

The bundled `model_299_g2stable.pt` was trained end to end under the
`g2_stable_v1_3_3` profile on Genesis World 1.3.3. The 48-column observation
fixed the phase-label ambiguity found on 2026-08-21. The accepted behavior clone
passed 8/8 from home, then three independent PPO jobs ran for 300 iterations with
seeds 1, 7, and 17. The selected seed-7 final checkpoint uses
`place_phase_reset_frac: 0.40` and learning rate `1e-5`.

The same selected checkpoint passed all nine evaluation seed/batch combinations
(219/219 episodes), the independent fixed bank (64/64 full and quality success),
and the fixed 512-episode sigma-0.02 action-noise bank (512/512). Its noise P99
final XY error, pre-lift drag, and post-release drift were 0.93, 0.35, and
0.07 mm; action clipping, IK failures, IK jump rejects, and post-release
recontacts were zero. See `pretrained/evaluation_summary.json` for the
machine-readable evidence.
