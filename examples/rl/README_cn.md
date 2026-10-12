# 强化学习示例

v0.3.1 的公开强化学习有两条仿真流程，都不提供真机策略执行。参考环境为 Linux、NVIDIA GPU、
Genesis World 1.4.4、Quadrants 1.3.3、PyTorch 2.10 和 RSL-RL 5.4.2。

[固定布局抓放](pick_place/README_cn.md) 是 xArm6 + Gripper G2 的发布主线。内置的 seed 7
`model_299_g2stable.pt` 已在 `g2_stable_v1_3_3` 接触物理配置下重新训练并通过全部发布检查：
9 组评估种子/并行数量组合共 219/219 回合、独立固定场景库 64/64、0.02 动作噪声下 512/512。
详细数据与限制见任务指南。随机方块起点不属于当前公开范围。

[裸臂到达](reach/README_cn.md) 训练 xArm 6 把末端伸向随机点。仓库不附带训练好的模型。

请在仓库根目录通过模块方式运行所有入口。

[English](README.md)
