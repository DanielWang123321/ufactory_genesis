# Reinforcement-learning examples

The v0.3.1 public RL surface has two simulation-only workflows and no real-robot
policy executor. It targets Linux, an NVIDIA GPU, Genesis World 1.4.4, Quadrants
1.3.3, PyTorch 2.10, and RSL-RL 5.4.2.

[Fixed-layout pick-place](pick_place/README.md) is the published xArm6 + Gripper G2
workflow. The bundled seed-7 `model_299_g2stable.pt` was retrained under the
`g2_stable_v1_3_3` contact-physics profile and passes every release check:
219/219 episodes across the nine evaluation seed/batch combinations, 64/64 in
the independent fixed bank, and 512/512 under 0.02 action noise. Details and
limitations are in the task guide. Random object starts stay outside the public scope.

[Bare-arm reach](reach/README.md) trains an xArm 6 to move its end effector to a
random point. No trained model is bundled.

Run every entry as a module from the repository root.

[中文说明](README_cn.md)
