# 强化学习示例

v0.2.13 的公开强化学习内容只有一条边界明确的 xArm6 + Gripper G2 主线：
[固定布局训练与评估](pick_place/README_cn.md)。参考环境为 Linux、NVIDIA GPU、
Genesis World 1.3.3、Quadrants 1.3.0、PyTorch 2.10 和 RSL-RL 5.4.2。

内置的 seed 7 `model_299_g2stable.pt` 已在 `g2_stable_v1_3_3` 接触物理配置下
重新训练并通过全部发布检查：9 组评估种子/并行数量组合共 219/219 回合、独立
固定场景库 64/64、0.02 动作噪声下 512/512。详细数据与限制见任务指南。

请在仓库根目录通过模块方式运行所有入口。示例仅用于仿真，不提供强化学习真机
执行器。随机方块起点不属于 v0.2.13 的公开范围，延期到后续版本。

[English](README.md)
