# 1. Reach: xArm 6 target reaching

The xArm 6 runs without a gripper in simulation; no Reach weights are bundled. Success requires a distance strictly below **5 mm**; the default training seed is 7. Run the commands below from the repository root.

[中文说明](README_cn.md) · [RL examples](../README.md) · [Next: 2. Pick & Place](../pick_place/README.md)

The environment has 24 observations, six actions, a 20 ms control period and a 5 s episode limit. Targets span x 0.15–0.55 m, y −0.30–0.30 m and z 0.05–0.50 m.

## Train from scratch

```bash
python -m examples.rl.reach.train \
  --log-dir outputs/rl/reach/reach-new/seed-7 \
  --seed 7 -B 16384 --max-iterations 300
```

This trains for 300 iterations, saving every 25 iterations and finally as `model_299.pt`. Choose another name if the output directory exists.

## Resume training

```bash
python -m examples.rl.reach.train \
  --resume outputs/rl/reach/reach-new/seed-7/model_299.pt \
  --log-dir outputs/rl/reach/reach-new/seed-7-resume \
  --seed 7 -B 16384 --max-iterations 100
```

`--max-iterations 100` adds 100 iterations. `-B` must match the original training run.

## Select and validate a checkpoint

```bash
# Select using 128 targets with seed 17002
python -m examples.rl.reach.run_experiments \
  --stage select --output outputs/rl/reach/reach-new --seeds 7

# Validate using 512 targets each with seeds 17000 and 17001
python -m examples.rl.reach.run_experiments \
  --stage accept --output outputs/rl/reach/reach-new --seeds 7
```

Read `selected_checkpoint` in `outputs/rl/reach/reach-new/seed-7/checkpoint-selection.json`. Each validation set requires ≥95% success and ≤10 mm mean terminal distance.

## Real-time GLB inference

After training and checkpoint selection, read the actual model path from the selection report:

```bash
RUN_DIR=outputs/rl/reach/reach-new/seed-7
CHECKPOINT="$RUN_DIR/$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$RUN_DIR/checkpoint-selection.json")"

# Single arm
python -m examples.rl.reach.visualize --checkpoint "$CHECKPOINT"

# Four arms
python -m examples.rl.reach.visualize --checkpoint "$CHECKPOINT" -b=4
```

Each successful arm holds for 1 s. Close the window or press Ctrl+C to exit. Keep the model, `config.yaml` and its `*.checkpoint_manifest.json` in the same directory.

## Evaluate a fixed target batch

```bash
python -m examples.rl.reach.evaluate \
  --checkpoint "$CHECKPOINT" \
  -B 128 --episodes 512 --seed 17000 \
  --min-success-rate 0.95 --max-mean-distance-m 0.010 \
  --quiet-episodes --summary-json outputs/rl/reach/evaluation-new-17000.json
```

For the second set, change `--seed` to `17001` and the report name to `evaluation-new-17001.json`.

## Run training, selection and validation

```bash
python -m examples.rl.reach.run_experiments \
  --output outputs/rl/reach/reach-auto \
  --seeds 7 -B 16384 --max-iterations 300
```

The output directory must not exist. For three training seeds, replace `--seeds 7` with `--seeds 1 7 17`.
