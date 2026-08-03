# Reinforcement-learning examples

The public v0.2.12 RL surface contains two deliberately bounded xArm6 + Gripper G2
pick-place workflows:

- [fixed-layout baseline](pick_place/README.md), retained with its deterministic
  512-episode bank and validated `model_199.pt` bundle;
- [random object start with a fixed target](pick_place/random_start/README.md), with
  a separate recipe, scenario banks, checkpoint bundle, and strict aggregate gates.
  Its bundled policy transparently combines the fixed RL actor with a simulator-state
  scripted guide; it is not a learned random-layout PPO claim.

Run every entry as a module from the repository root. The examples support Linux
with an NVIDIA GPU only. They do not provide a real-robot policy executor.

[中文说明](README_cn.md)
