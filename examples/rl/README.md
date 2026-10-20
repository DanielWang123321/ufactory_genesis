# Reinforcement-learning examples

Follow the examples in increasing order of difficulty. These workflows run in simulation; run their commands as modules from the repository root.

Policy weights and checkpoints are not bundled or committed. Train policies locally and keep run artifacts under `outputs/rl/`, which Git ignores.

1. **[Reach — target reaching](reach/README.md)**: move the xArm 6 end effector to a target without a gripper. Success requires a distance strictly below **5 mm**; the default training seed is 7. The guide covers training, checkpoint selection, acceptance and real-time GLB inference. No weights are bundled.
2. **[Pick & Place — tabletop](pick_place/README.md)**: use xArm6 + Gripper G2 to grasp, lift and transfer a cube, then set it down and release it at a tabletop target. The current example uses a fixed layout; the guide covers training and inference.
3. **Box packing — planned**: place an object inside a box, adding box geometry and contact constraints. The RL environment and training/inference entries remain to be implemented.

The existing [packaging showcase](../README.md#packaging-showcase-simulation) uses trajectory control.

[中文说明](README_cn.md)
