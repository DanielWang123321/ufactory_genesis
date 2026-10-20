# 2. 抓放：xArm6 + Gripper G2

在仿真中将 30 mm 方块从 `(300, 0) mm` 搬到 `(300, 300) mm`。采用端到端 PPO，从零训练，每回合均从末端 `(300, 0, 300) mm` 和 50 mm 开度开始。

[English](README.md) · [RL 示例](../README_cn.md) · [前一例：Reach](../reach/README_cn.md)

环境版本为 **3**：70 维观测、4 维动作 `Δx, Δy, Δz, Δgap`，固定夹爪朝下，控制周期 20 ms、32 个物理子步，回合最长 8 s。XYZ 每步最大增量 5 mm，开度每步最大增量 3 mm；命令持续积分，零动作保持命令。四个通道始终有效，仅保留工作空间、URDF 关节限位、IK 检查和 25 mm 设定点距离限制。夹爪使用标定版 [G2Controller](../../../ufactory/simulation/g2_controller.py)。

观测包含关节位置与速度、末端和物体状态、目标相对位置、两侧实际指垫中心相对位置、命令与实测开度、末端设定点误差、双指接触力、桌面支撑力、机器人碰桌载荷、上一步动作及任务历史；不包含阶段信息。旧版模型的观测和动作语义不兼容，加载时会要求重新训练，物理迁移选项也不能跳过此检查。

奖励使用折扣势能差及一次性抓取、抬升和低位释放奖励，覆盖接近、夹持、抬升、搬运、释放和稳定保持。抓取前开度势能在放置确认前随指垫与方块距离连续变化，引导先张开接近再闭爪并在下降阶段平滑释放预紧行程。放置势能依据按姿态计算的方块最低点，释放前以 2 mm 离桌高度为缓冲目标，确认低位释放后在 0–2 mm 区间内保持满值。确认目标低位释放（`place_rewarded`）后，接近、预抓取与夹持势能平滑交接至无接触开度进展及稳定保持步数势能，消除脱爪后反手重夹的收益。夹持状态下离桌 20 mm 内过快下降、触桌后继续下移、桌面过载、高处脱爪与机器人碰桌均受罚。开合或反复抬升不能重复获得一次性奖励。

场景几何与初始末端位姿集中定义在 `ufactory/training/pick_place_config.py`。运行 YAML 支持覆盖 `task.parameters` 下的 `control_dt_s`、`workspace_lower_m`、`workspace_upper_m`、`object_mass_kg`，以及夹爪阻尼/摩擦、求解器设置、G2 速度/力百分比。8 s 回合时限、模型、方块尺寸和布局固定。不支持的覆盖及无效配方字段会报错；既有 v3 检查点仍执行哈希、动作契约和安全校验。

训练日志、检查点清单与缓存共用 `ufactory/training/runner.py` 和 `runtime.py`。

## 严格成功条件

- 起抓前拖行不超过 10 mm、倾角不超过 15°；双指接触连续确认至少 3 步，接触进入阈值 0.2 N、退出阈值 0.05 N。
- 夹持方块最低点抬到至少 65 mm（平放时中心高 80 mm），落点水平误差不超过 20 mm，放置倾角不超过 15°。
- 双指脱离时方块实际离桌高度为 **0–3 mm**，开度至少 35 mm；两指连续 3 步低于 0.05 N 才确认脱离。高处抛落后落入目标区仍判失败。
- 触桌至双指脱离的末端额外下降不超过 **1 mm**。首次触桌保留前一帧末端高度；单指仍接触及短暂接触抖动继续统计。脱爪后重新接触会重新测量触桌与释放，并保留回合内最差下压量及最高释放高度。
- 释放后由桌面支撑、物体速度小于 30 mm/s，并连续稳定至少 **0.2 s**；机器人碰桌判失败。

接触力通过对象筛选和 `valid_mask` 计算，区分方块支撑与机器人碰桌载荷，见 [Genesis 接触接口](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_solver/collision_detection.html)。报告包含 `release_height_mm`, `release_confirmed`, `press_mm`, `max_table_force_n`, `max_place_table_force_n`, `max_held_table_force_n` 和 `max_robot_table_force_n`。释放高度需结合是否确认脱爪读取。

## 基线与训练

在项目根目录、`py313` 环境运行；指定 GPU 时若回退 CPU，会明确报错。

```bash
python -m examples.rl.pick_place.evaluate --zero-policy -B 16 --episodes 32 --quiet-episodes
python -m examples.rl.pick_place.evaluate --expert -B 16 --episodes 32 \
  --min-success-rate 1.0 --max-press-mm 1.0 --quiet-episodes

# 单个种子：完整任务 PPO，从零训练，无专家训练或阶段复位
python -m examples.rl.pick_place.train -e pick_place_e2e_s1 --seed 1 \
  -B 2048 --max-iterations 2000 --eval-interval 50 --eval-episodes 64

# 三个种子分别训练、独立评估 64 回合并录制 B=1 GLB 回放
python -m examples.rl.pick_place.run_seeds
```

默认种子为 1、2、3；学习率 `3e-4`、`gamma=0.995`、初始动作标准差 `0.5`、熵系数 `0.005`。沿用 `[256, 256]` 网络和每轮 32 步采样。连续两次周期评估达到验收门限后提前停止，最多 2000 轮；`--early-stop-evaluations 0` 可关闭提前停止，实际轮数写入 `training_summary.json`。规则专家是独立评估策略，不参与 PPO。实验目录已存在时停止，避免覆盖训练产物。

输出位于 `outputs/rl/pick_place/pick_place_e2e_s1/`（其他种子对应 `s2/s3`）。检查点按严格成功率、有效低位触桌率、最大下压量、平均落点误差依次排序；保存配置、源码哈希、检查点清单、训练日志和 JSON/CSV 评估报告。三种子验收各要求成功率 **≥98%**、最大下压 **≤1 mm**；汇总和进程日志在 `outputs/rl/pick_place/e2e_validation/`。

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt \
  -B 32 --episodes 64 --seed 18001 --min-success-rate 0.98 --max-press-mm 1.0 \
  --out outputs/rl/pick_place/pick_place_e2e_s1/test.json

python -m examples.rl.pick_place.visualize -e pick_place_e2e_s1 -B 1
python -m examples.rl.pick_place.visualize -e pick_place_e2e_s1 -B 1 --episodes 1 \
  --headless --record outputs/rl/pick_place/pick_place_e2e_s1/replay.mp4
```

录像默认采用 GLB PBR，按 50 FPS 保存，同时输出 JSON/CSV 指标。在统一的 `impratio = 100.0, noslip_iterations = 0` 物理配置下，三种子 GPU 训练（`pick_place_e2e_s1..s3`）、64 回合独立严格评估及 `B=1` GLB 回放验收均已通过（成功率 100%，最大下压 0.084 mm，平均落点误差 4.44 mm）。
