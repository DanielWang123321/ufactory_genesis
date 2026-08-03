# Fixed-layout xArm6 + Gripper G2 RL

This is the initial public RL example in v0.2.11. It learns one fixed `+Y`
pick-place layout with xArm6 and Gripper G2. It does **not** claim random-layout
generalization, 99% disturbance robustness, CPU/Windows training, or real-robot
policy deployment.

[中文说明](README_cn.md)

## Install and hardware boundary

Use Linux, an NVIDIA GPU supported by Genesis, Python 3.12 or 3.13, Genesis World
1.3.1, RSL-RL 5.4.2, and PyTorch 2.10. The lock file is the reference environment:

```bash
uv sync --extra sim --extra rl
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
```

An editable pip install is also supported, but its lower bounds may resolve newer
packages than the reference lock:

```bash
pip install -e ".[sim,rl]"
```

The published policy has a small network, but Genesis scene compilation, parallel
environments, PPO rollouts, and BC datasets dominate memory. Start with `-B 1` for
visualization, `-B 8` for evaluation, and `-B 256` or below for training if your GPU
has limited memory. The canonical recipe's `20480` environments target a large
training GPU and should not be treated as a universal default.

## Evaluate the bundled checkpoint

The evaluator defaults to `pretrained/model_199.pt` and loads `config.yaml` from
the checkpoint's own directory. It validates the config hash, checkpoint hash,
task contract, robot contract, and runtime config before using PyTorch's safe
weights-only loader. Evaluation does not write into `pretrained/`; reports are
created only when an output flag is supplied.

Quick visualization (`performance_mode` defaults off, avoiding the old static-array
viewer compile path):

```bash
python -m examples.rl.pick_place.evaluate --episodes 1
```

Small headless check:

```bash
python -m examples.rl.pick_place.evaluate --headless -B 8 --episodes 8 \
  --acceptance-profile contact_v1
```

Published deterministic 64-episode gate:

```bash
python -m examples.rl.pick_place.evaluate --headless -B 64 --episodes 64 \
  --scenario-bank examples/rl/pick_place/scenarios/fixed_seed17000_n512.json \
  --acceptance-profile contact_v1 \
  --summary-json outputs/rl/pick_place/eval/deterministic_64.json
```

Formal fixed-bank disturbance measurement:

```bash
python -m examples.rl.pick_place.evaluate --headless -B 64 --episodes 512 \
  --scenario-bank examples/rl/pick_place/scenarios/fixed_seed17000_n512.json \
  --acceptance-profile contact_v1 \
  --action-noise-std 0.02 --action-noise-bank 20260731 \
  --summary-json outputs/rl/pick_place/eval/noise_512.json
```

For an explicit checkpoint, `--checkpoint path/to/model_N.pt` defaults to
`path/to/config.yaml` and `path/to/model_N.checkpoint_manifest.json`. A complete
bundle is required; a bare `.pt` file is intentionally rejected.

## Results and metric meaning

Under the v0.2.12 Genesis 1.3.1 physics baseline, the unchanged checkpoint passed
the nine no-disturbance combinations of seeds `1/7/17` and batch sizes `1/8/64`
(9/9, 219/219 episodes). Its independent fixed-bank 64-episode run was 64/64.

With Gaussian action disturbance standard deviation `0.02` and the fixed seed
`20260731`, the Genesis 1.3.1 rebaseline achieved **512/512** full and quality
success. P99 final XY error, pre-lift drag, and post-release drift were 8.35, 1.05,
and 0.73 mm, with zero action clipping or IK faults. The diagnostic still recorded
post-release finger contact in 120/512 episodes. Recontact is not one of the aggregate
robustness-gate predicates, so this result does not claim contact-free disturbance
handling or real-world robustness. Exact metadata is in `pretrained/evaluation_summary.json`.

`Full success` means grasp, lift, valid set-down/release, final pose, velocity,
hold time, and all selected acceptance-profile quality limits passed. It is stricter
than grasp or lift success alone.

## Canonical training path

First verify that the scripted expert can reach the current acceptance target:

```bash
python -m examples.rl.pick_place.evaluate --expert --headless -B 8 --episodes 8 \
  --acceptance-profile contact_v1
```

Then behavior-clone the expert and start PPO from the BC policy:

```bash
python -m examples.rl.pick_place.pretrain_bc -B 256 --rollout-steps 600 \
  -e fixed-layout-bc

python -m examples.rl.pick_place.train -B 256 --max-iterations 300 \
  --warm-start outputs/rl/pick_place/fixed-layout-bc/model_0.pt \
  -e fixed-layout-ppo
```

This compact expert → BC → PPO path is runnable, but it is **not guaranteed to
reproduce** `model_199.pt`, which was selected from a longer private multi-stage
experiment history.

Fine-tune the bundled actor and critic in a new output directory:

```bash
python -m examples.rl.pick_place.train -B 256 --max-iterations 20 \
  --actor-critic-warm-start examples/rl/pick_place/pretrained/model_199.pt \
  -e model-199-finetune
```

One-iteration PPO smoke test:

```bash
python -m examples.rl.pick_place.train -B 1 --max-iterations 1 -e smoke
```

All default training and BC products go under ignored
`outputs/rl/pick_place/<experiment>/`. Never use `pretrained/` as `--log-dir`.

To regenerate the bundled fixed scenario bank (not needed for normal evaluate/train):

```bash
python -m examples.rl.pick_place.make_scenario_bank --count 512 --seed 17000 \
  --output examples/rl/pick_place/scenarios/fixed_seed17000_n512.json
```
