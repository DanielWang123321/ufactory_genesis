# Bare-arm xArm 6 reach

Simulation only. The arm has no gripper and moves its end effector to a new random point each episode. The repository does not ship a trained model, and this example has no real-robot executor.

[中文说明](README_cn.md)

Targets are drawn uniformly from x 0.15–0.55 m, y -0.30–0.30 m, and z 0.05–0.50 m. The episode ends below 3 cm, or at 5 seconds. The complete 6-joint action range `[-1, 1]` maps linearly to `±0.01 rad` per step. Position commands accumulate within URDF joint limits.

The 24 observations contain joint positions, velocities, end-effector position, relative target position, and position command minus actual joint position. Reward is `dt × [exp(-10d) − 0.001‖q̇‖²] + 2 × success`, with one success bonus per episode. PPO uses a Beta distribution, fixed `3e-4` learning rate, and actor/critic observation normalization. Environment version 2 requires retraining older 18-observation checkpoints.

The reference stack matches the pick-place example: Linux, an NVIDIA GPU, Genesis World 1.4.4, and RSL-RL 5.4.2.

```bash
pip install -e ".[sim,rl]"
export NUMBA_CACHE_DIR="$HOME/.cache/numba"

python -m examples.rl.reach.train -e reach
python -m examples.rl.reach.evaluate \
  --checkpoint outputs/rl/reach/reach/model_299.pt \
  --episodes 512 -B 128 --seed 17000 \
  --min-success-rate 0.95 --max-mean-distance-m 0.035 \
  --summary-json outputs/rl/reach/reach/evaluation-17000.json
python -m examples.rl.reach.evaluate \
  --checkpoint outputs/rl/reach/reach/model_299.pt --vis
```

With `--vis`, a successful reach holds the current pose for 1 second before reset and the next target. Headless evaluation does not hold.

Train first, then pass the `model_*.pt` that run wrote. The default run saves every 25 iterations and finishes at `model_299.pt`. Keep the checkpoint, its integrity manifest, and `config.yaml` together. No weights are shipped in the repository.

The default of 16384 parallel environments is the faster setting measured on this project's RTX 4060 Ti (8 GB) for this bare-arm scene. Pass `-B` to use fewer environments. Resume with the same `-B` as the original run. Artifacts go to `outputs/rl/reach/` and are not committed.

Each fixed target runs once until success or timeout. Targets use an independent CPU generator, so changing the batch size preserves the target bank. The default evaluation seed is 17002. JSON includes every successful and failed episode; failing either requested acceptance threshold exits with status 1. Select checkpoints on 128 targets with seed 17002, then validate separately on 512 targets each with seeds 17000 and 17001.

Runs retain source snapshots, configuration and source/URDF hashes, per-iteration statistics, stop reasons, and checkpoints. Statistics weight completed episodes correctly and include success rate, terminal distance, reward components, action saturation, and command tracking error. Resume restores joint positions, velocities, commands, and episode accounting.
