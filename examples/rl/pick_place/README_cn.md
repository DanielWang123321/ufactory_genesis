# xArm6 + Gripper G2 固定布局 RL

这是 v0.2.11 首次公开的 RL 示例：xArm6 搭配 Gripper G2，在固定 `+Y` 布局完成抓取、
抬升、搬运、落桌和松爪。它不宣称随机布局泛化、99% 扰动稳健性、Windows/CPU 训练，
也不提供 RL 真机策略执行。

[English](README.md)

## 安装与硬件边界

参考环境为 Linux、Genesis 支持的 NVIDIA GPU、Python 3.12/3.13、Genesis World 1.3.0、
RSL-RL 5.4.2 和 PyTorch 2.10。锁文件是标准复现环境：

```bash
uv sync --extra sim --extra rl
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
```

也可使用可编辑安装，但最低版本约束可能解析到比参考锁更新的依赖：

```bash
pip install -e ".[sim,rl]"
```

策略网络本身很小，显存主要消耗在 Genesis 场景、并行环境、PPO rollout 和 BC 数据集。
建议可视化从 `-B 1`、评估从 `-B 8` 开始；显存较小时训练先使用 `-B 256` 或更低。
规范配方中的 `20480` 面向大显存训练卡，不是所有显卡的通用默认值。

## 评估随仓库检查点

评估器默认定位 `pretrained/model_199.pt`，并从检查点同目录读取 `config.yaml`。在
PyTorch 以安全权重模式加载前，它会先校验配置哈希、检查点哈希、任务/机器人契约及
运行时配置。评估不会写入 `pretrained/`；只有显式传入报告参数时才产生输出文件。

快速可视化（默认关闭静态数组性能模式，避免旧的 viewer 长编译路径）：

```bash
python -m examples.rl.pick_place.evaluate --episodes 1
```

小规模无窗口检查：

```bash
python -m examples.rl.pick_place.evaluate --headless -B 8 --episodes 8 \
  --acceptance-profile contact_v1
```

公开的固定 64 回合门槛：

```bash
python -m examples.rl.pick_place.evaluate --headless -B 64 --episodes 64 \
  --scenario-bank examples/rl/pick_place/scenarios/fixed_seed17000_n512.json \
  --acceptance-profile contact_v1 \
  --summary-json outputs/rl/pick_place/eval/deterministic_64.json
```

正式固定噪声库测量：

```bash
python -m examples.rl.pick_place.evaluate --headless -B 64 --episodes 512 \
  --scenario-bank examples/rl/pick_place/scenarios/fixed_seed17000_n512.json \
  --acceptance-profile contact_v1 \
  --action-noise-std 0.02 --action-noise-bank 20260731 \
  --summary-json outputs/rl/pick_place/eval/noise_512.json
```

显式指定 `--checkpoint path/to/model_N.pt` 时，默认从同目录寻找 `config.yaml` 和
`model_N.checkpoint_manifest.json`。缺少其中任一文件的裸 `.pt` 会被拒绝。

## 已知结果与指标

随仓库策略在种子 `1/7/17` × 并行环境 `1/8/64` 的九组无扰动测试中为 9/9，固定
64 回合为 64/64。

动作高斯扰动标准差为 `0.02`、固定种子为 `20260731` 时，结果为
**442/512（86.3%）**；抓取和抬升均为 512/512，最终平面误差、首次夹起前拖动、
松爪后漂移的 P99 分别为 8.52、1.07、0.73 mm。原定 99% 稳健性目标仍未完成，
不作为 v0.2.11 初版发布阻塞项。精确工件元数据见 `pretrained/evaluation_summary.json`。

`Full success` 同时要求抓取、抬升、有效落桌/松爪、最终位置与速度、连续稳定时间，
以及所选验收配置的全部质量限制通过；它比单独的抓取或抬升成功严格得多。

## 规范训练主线

先用脚本专家确认当前环境中的目标可达：

```bash
python -m examples.rl.pick_place.evaluate --expert --headless -B 8 --episodes 8 \
  --acceptance-profile contact_v1
```

随后进行行为克隆，并从 BC 策略启动 PPO：

```bash
python -m examples.rl.pick_place.pretrain_bc -B 256 --rollout-steps 600 \
  -e fixed-layout-bc

python -m examples.rl.pick_place.train -B 256 --max-iterations 300 \
  --warm-start outputs/rl/pick_place/fixed-layout-bc/model_0.pt \
  -e fixed-layout-ppo
```

这条“专家可达性检查 → BC → PPO”的精简主线可以运行，但**不保证一次训练复现**
`model_199.pt`；该策略来自更长的私人多阶段实验历史。

从随仓库 actor/critic 建立新的微调分支：

```bash
python -m examples.rl.pick_place.train -B 256 --max-iterations 20 \
  --actor-critic-warm-start examples/rl/pick_place/pretrained/model_199.pt \
  -e model-199-finetune
```

单轮 PPO 冒烟：

```bash
python -m examples.rl.pick_place.train -B 1 --max-iterations 1 -e smoke
```

训练与 BC 默认输出到已忽略的 `outputs/rl/pick_place/<实验名>/`。不要把
`pretrained/` 用作 `--log-dir`。

如需重新生成随仓库的固定场景库（日常评估/训练不必跑）：

```bash
python -m examples.rl.pick_place.make_scenario_bank --count 512 --seed 17000 \
  --output examples/rl/pick_place/scenarios/fixed_seed17000_n512.json
```
