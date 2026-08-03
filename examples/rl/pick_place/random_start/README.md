# Random-object-start xArm6 + Gripper G2 policy

This v0.2.12 simulation-only workflow randomizes the cube start over the configured
XY envelope while keeping the target fixed. It is deliberately separate from the
[fixed-layout RL baseline](../README.md).

[中文说明](README_cn.md)

## What the bundled policy is

The release artifact is a transparent hierarchical policy, not a claim that PPO
learned random-layout generalization:

- on the canonical layout, `GuidedPickPlaceMLPModel` executes the original
  `model_199.pt` actor with bit-for-bit identical tensors and actions;
- on a non-canonical object layout, six immutable layout offsets select the four
  bounded actions produced by `scripted_pick_place_expert_v2`;
- the guide uses simulator state and therefore requires this complete environment.
  The `.pt` file is not a standalone learned random-layout controller.

The 54-value policy contract is `44 legacy + 6 layout + 4 guide action`. Stochastic
random-layout PPO is rejected because replacing sampled actions would invalidate
their policy log probabilities. The inherited actor is fully frozen and guarded
during the one-iteration artifact-initialization run.

## Install and scope

Use Linux, an NVIDIA GPU, Python 3.12 or 3.13, Genesis World 1.3.1, RSL-RL 5.4.2,
and PyTorch 2.10. The reference environment is the repository lock file:

```bash
uv sync --extra sim --extra rl
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
```

This workflow supports only xArm6 + Gripper G2 in simulation. The target is fixed;
target randomization, action-noise robustness, CPU/Windows execution, and real-robot
policy deployment are not release claims.

## Evaluate the bundled checkpoint

Always pass both the random-start recipe and checkpoint. The evaluator verifies the
configuration, checkpoint, runtime, and scenario-bank hashes before safe weights-only
loading.

Quick headless check:

```bash
python -m examples.rl.pick_place.evaluate \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --checkpoint examples/rl/pick_place/random_start/pretrained/model_0.pt \
  --scenario-bank examples/rl/pick_place/random_start/scenarios/object_uniform_seed31212_n512.json \
  --episodes 8 -B 8 --stage 4 --headless --acceptance-profile contact_v1
```

Formal standard gate (512 uniform object starts):

```bash
python -m examples.rl.pick_place.evaluate \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --checkpoint examples/rl/pick_place/random_start/pretrained/model_0.pt \
  --scenario-bank examples/rl/pick_place/random_start/scenarios/object_uniform_seed31212_n512.json \
  --episodes 512 -B 512 --stage 4 --headless --require-target standard \
  --summary-json outputs/rl/pick_place/random-uniform-final.json
```

Formal boundary robustness gate:

```bash
python -m examples.rl.pick_place.evaluate \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --checkpoint examples/rl/pick_place/random_start/pretrained/model_0.pt \
  --scenario-bank examples/rl/pick_place/random_start/scenarios/object_edge_seed31213_n256.json \
  --episodes 256 -B 256 --stage 4 --headless --require-target robustness \
  --summary-json outputs/rl/pick_place/random-edge-final.json
```

Reduce `-B` on smaller GPUs; every bank entry is still assigned exactly once.

## Published results

The checkpoint passed the final, post-selection banks:

| Bank | Full / quality | P99 final XY | P99 pre-lift drag | P99 post-release drift |
|---|---:|---:|---:|---:|
| uniform, seed 31212 | 512/512 | 1.72 mm | 4.46 mm | 0.50 mm |
| boundary, seed 31213 | 256/256 | 1.80 mm | 4.56 mm | 0.49 mm |

Both runs had zero action clipping, IK failures, IK jump rejects, hard landings, and
post-release recontacts. Exact machine-readable reports live in `pretrained/`.

## Reproduce the artifact initialization

This command imports and freezes the fixed actor. Its stage-0 rollout initializes the
critic and writes a complete hashed bundle; it does not train the random-layout guide:

```bash
python -m examples.rl.pick_place.train \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --warm-start examples/rl/pick_place/pretrained/model_199.pt \
  --curriculum-max-stage 0 --max-iterations 1 -B 256 \
  -e random-start-guided-artifact
```

The BC, DAgger, protected residual, and curriculum paths remain available for research,
but their v0.2.12 experiments did not meet the release gates and are not represented as
the bundled solution.
