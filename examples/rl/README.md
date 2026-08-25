# Reinforcement-learning examples

The v0.2.13 public RL surface contains one bounded xArm6 + Gripper G2 workflow:
[fixed-layout training and evaluation](pick_place/README.md). It targets Linux,
an NVIDIA GPU, Genesis World 1.3.3, Quadrants 1.3.0, PyTorch 2.10, and RSL-RL
5.4.2.

The bundled seed-7 `model_299_g2stable.pt` was retrained under the
`g2_stable_v1_3_3` contact-physics profile and passes every release check:
219/219 episodes across the nine evaluation seed/batch combinations, 64/64 in
the independent fixed bank, and 512/512 under 0.02 action noise. Details and
limitations are in the task guide.

Run every entry as a module from the repository root. The example is
simulation-only and provides no real-robot policy executor. Random object starts
are outside the v0.2.13 public scope and are deferred to a later version.

[中文说明](README_cn.md)
