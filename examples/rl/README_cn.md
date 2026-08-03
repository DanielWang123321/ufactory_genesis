# 强化学习示例

v0.2.12 公开两条边界明确的 xArm6 + Gripper G2 抓放主线：

- [固定布局基线](pick_place/README_cn.md)：保留固定 512 回合场景库和经过完整性校验的
  `model_199.pt`；
- [方块起点随机、目标固定](pick_place/random_start/README_cn.md)：使用独立配方、场景库、
  检查点包和严格汇总门禁；随仓库策略透明组合固定 RL actor 与读取仿真状态的脚本
  guide，不宣称 PPO 已学习随机布局泛化。

请在仓库根目录通过模块方式运行所有入口。这些示例仅支持 Linux + NVIDIA GPU，且不提供
任何 RL 真机执行接口。

[English](README.md)
