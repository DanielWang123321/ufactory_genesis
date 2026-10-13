# 1. Reach：xArm 6 目标到达

使用未安装夹爪的 xArm 6，仅在仿真中运行，仓库不附带模型权重。成功条件为末端距离严格小于 **5 mm**，默认训练种子 7。在项目根目录运行以下指令。

[English](README.md) · [RL 示例顺序](../README_cn.md) · [下一步：2. Pick & Place](../pick_place/README_cn.md)

24 维观测、6 维动作，控制周期 20 ms，每回合最长 5 s。目标范围为 x 0.15–0.55 m、y -0.30–0.30 m、z 0.05–0.50 m。

## 从零训练

```bash
python -m examples.rl.reach.train \
  --log-dir outputs/rl/reach/reach-new/seed-7 \
  --seed 7 -B 16384 --max-iterations 300
```

训练 300 轮，每 25 轮保存检查点，最后为 `model_299.pt`。输出目录已存在时换一个名称。

## 续训

```bash
python -m examples.rl.reach.train \
  --resume outputs/rl/reach/reach-new/seed-7/model_299.pt \
  --log-dir outputs/rl/reach/reach-new/seed-7-resume \
  --seed 7 -B 16384 --max-iterations 100
```

`--max-iterations 100` 表示追加 100 轮；`-B` 必须与原训练一致。

## 筛选与验收

```bash
# 用 seed 17002 的 128 个目标筛选检查点
python -m examples.rl.reach.run_experiments \
  --stage select --output outputs/rl/reach/reach-new --seeds 7

# 用 seed 17000、17001 各 512 个目标验收选中的检查点
python -m examples.rl.reach.run_experiments \
  --stage accept --output outputs/rl/reach/reach-new --seeds 7
```

选中的模型名在 `outputs/rl/reach/reach-new/seed-7/checkpoint-selection.json` 的 `selected_checkpoint` 字段。每组验收要求成功率 ≥95%、平均终止距离 ≤10 mm。

## GLB 实时推理

训练并完成筛选后，从筛选报告读取实际模型路径：

```bash
RUN_DIR=outputs/rl/reach/reach-new/seed-7
CHECKPOINT="$RUN_DIR/$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$RUN_DIR/checkpoint-selection.json")"

# 单臂
python -m examples.rl.reach.visualize --checkpoint "$CHECKPOINT"

# 四臂
python -m examples.rl.reach.visualize --checkpoint "$CHECKPOINT" -b=4
```

成功后保持 1 秒，关闭窗口或 Ctrl+C 退出。模型、`config.yaml` 和对应的 `*.checkpoint_manifest.json` 须放在同一目录。

## 单独批量评估

```bash
python -m examples.rl.reach.evaluate \
  --checkpoint "$CHECKPOINT" \
  -B 128 --episodes 512 --seed 17000 \
  --min-success-rate 0.95 --max-mean-distance-m 0.010 \
  --quiet-episodes --summary-json outputs/rl/reach/evaluation-new-17000.json
```

第二组将 `--seed` 改为 `17001`，报告文件名也改为 `evaluation-new-17001.json`。

## 一键训练、筛选与验收

```bash
python -m examples.rl.reach.run_experiments \
  --output outputs/rl/reach/reach-auto \
  --seeds 7 -B 16384 --max-iterations 300
```

输出目录须不存在。三种子训练将 `--seeds 7` 改为 `--seeds 1 7 17`。
