# xArm6 + Gripper G2 方块随机起点策略

这是 v0.2.12 新增的纯仿真主线：方块起点在配置的 XY 包络内随机，目标保持固定。
它与[固定布局 RL 基线](../README_cn.md)分开发布。

[English](README.md)

## 随仓库策略的真实性质

发布工件是公开透明的分层策略，并不宣称 PPO 已学会随机布局泛化：

- 规范固定布局下，`GuidedPickPlaceMLPModel` 执行原 `model_199.pt` actor，参数和
  动作均逐位一致；
- 方块起点不是规范布局时，六个不可变布局偏移会选择
  `scripted_pick_place_expert_v2` 产生的四维有界动作；
- guide 读取仿真器状态，因此必须依赖完整环境，单独的 `.pt` 不是已学习的随机布局
  控制器。

策略观测为 `44 个旧观测 + 6 个布局偏移 + 4 个 guide 动作 = 54`。随机布局上的
随机 PPO 会被主动拒绝，因为替换采样动作会破坏 PPO 的概率计算。工件初始化期间，
继承 actor 全部冻结，并在每次更新前后检查其参数没有变化。

## 安装与范围

参考环境为 Linux、Genesis 支持的 NVIDIA GPU、Python 3.12/3.13、Genesis World
1.3.1、RSL-RL 5.4.2 和 PyTorch 2.10：

```bash
uv sync --extra sim --extra rl
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
```

本主线只支持仿真中的 xArm6 + Gripper G2，目标保持固定。目标随机化、动作噪声
99% 稳健性、CPU/Windows 运行及 RL 真机部署均不在发布声明内。

## 评估随仓库检查点

必须同时指定 random-start 配方和检查点。评估器会先校验配置、检查点、运行时及
场景库哈希，再用安全权重模式加载。

小规模无窗口检查：

```bash
python -m examples.rl.pick_place.evaluate \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --checkpoint examples/rl/pick_place/random_start/pretrained/model_0.pt \
  --scenario-bank examples/rl/pick_place/random_start/scenarios/object_uniform_seed31212_n512.json \
  --episodes 8 -B 8 --stage 4 --headless --acceptance-profile contact_v1
```

正式 standard 门（512 个均匀方块起点）：

```bash
python -m examples.rl.pick_place.evaluate \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --checkpoint examples/rl/pick_place/random_start/pretrained/model_0.pt \
  --scenario-bank examples/rl/pick_place/random_start/scenarios/object_uniform_seed31212_n512.json \
  --episodes 512 -B 512 --stage 4 --headless --require-target standard \
  --summary-json outputs/rl/pick_place/random-uniform-final.json
```

正式边界 robustness 门：

```bash
python -m examples.rl.pick_place.evaluate \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --checkpoint examples/rl/pick_place/random_start/pretrained/model_0.pt \
  --scenario-bank examples/rl/pick_place/random_start/scenarios/object_edge_seed31213_n256.json \
  --episodes 256 -B 256 --stage 4 --headless --require-target robustness \
  --summary-json outputs/rl/pick_place/random-edge-final.json
```

显存较小时可降低 `-B`；场景库中的每个条目仍只分配一次。

## 发布结果

冻结模型后生成的最终场景库结果如下：

| 场景库 | Full / quality | 最终 XY P99 | 夹起前拖动 P99 | 松爪后漂移 P99 |
|---|---:|---:|---:|---:|
| 均匀，seed 31212 | 512/512 | 1.72 mm | 4.46 mm | 0.50 mm |
| 边界，seed 31213 | 256/256 | 1.80 mm | 4.56 mm | 0.49 mm |

两次运行的动作裁剪、IK 失败、IK 跳变拒绝、硬着陆和松爪后重接触均为零。精确的
机器可读报告位于 `pretrained/`。

## 复现工件初始化

下列命令导入并冻结固定布局 actor。stage-0 rollout 只初始化 critic 并写出完整哈希
工件，不会训练随机布局 guide：

```bash
python -m examples.rl.pick_place.train \
  --recipe examples/rl/pick_place/random_start/recipe.yaml \
  --warm-start examples/rl/pick_place/pretrained/model_199.pt \
  --curriculum-max-stage 0 --max-iterations 1 -B 256 \
  -e random-start-guided-artifact
```

BC、DAgger、受保护 residual 和 curriculum 路径仍可用于研究，但 v0.2.12 的相关
实验没有达到发布门，因此随仓库方案不会把它们描述成成功的随机布局学习结果。
