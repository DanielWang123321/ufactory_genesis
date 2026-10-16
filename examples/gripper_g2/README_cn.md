# Gripper G2 标定动力学仿真

[English](README.md)

独立夹爪及 xArm5/6/7、UF850 的 G2 仿真统一使用 `g2_calibrated_v1`。
独立夹爪场景和验收用例在 `acceptance.py`；G2 模型、控制器和 Genesis 接口在 `ufactory/simulation/`。本机验证版本为 Genesis World 1.4.4。
真机和 SDK 仿真仍使用毫米开度，运动镜像使用相同的关节几何。

从项目根目录运行。抓放和包装预检需要 Pinocchio/Coal（`.[real]`）。

```bash
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
export GS_CACHE_FILE_PATH="$PWD/.cache/genesis"
export QD_OFFLINE_CACHE_FILE_PATH="$PWD/.cache/quadrants"

# 夹持 30 mm 自由方块，随后撤去支撑。
python examples/gripper_g2/grasp_cube.py --headless --backend gpu \
  --position-mm 0 --speed-percent 50 --force-percent 50 --csv reports/g2/cube.csv

# 原有 33 个独立动力学工况，保存各工况 CSV 和验收结果。
python examples/gripper_g2/verify.py --backend gpu --suite all --out reports/g2/validation
```

位置为指垫开度 **0–84 mm**，速度为 **0–100%**，力为 **1–100%**。
速度为零暂停位置进给并保留夹持预载；自动抓放和包装要求速度大于零。
力百分比通过曲线限制电机输出，不代表牛顿值。超过 3500 rpm 标定范围时，力采用边界曲线，运动速度保留输入值。

默认参数：`kp=40、kv=5、ki=40`，夹爪关节附加惯量 0.01 kg·m²，
50 Hz、32 子步、接触时间常数 1.25 ms、noslip 迭代 5 次。只驱动 `drive_joint`。
不启用 noslip 时，pyramidal 摩擦锥与软接触会使夹持中的方块蠕滑，力 50% 时约 2.4 mm/s。
抓放 RL 保持 `noslip_iterations: 0`，因为 5 次迭代会使 20480 环境批量仿真慢约 7 倍。
控制器使用位置、速度反馈及积分抗饱和；接触力仅用于流程确认。
`G2Controller` 只有一份 Torch 实现，提供
`command(position_mm, speed_percent, force_percent)`、`update` 和 `reset`；
单个场景在 CPU 上以 float64、`num_envs=1` 运行，RL 批量支持按环境复位和独立力矩限制。

稳态力曲线内置于 `ufactory/simulation/g2_model.py`。
参考原始 SDK 位置为 230，推断开度 24.155872 mm，并非官方声明的传感器厚度。
11 份标准 G2 URDF 的六个联动关节统一为：开爪 `0.057210812728026743 rad`、
闭爪 `0.86509464599328212 rad`、速度上限 `2.608067454724 rad/s`。
采用公称 **0–84 mm** 行程；硬件个体的 1–2 mm 偏差留给设备标定，本次不新增标定功能。

`ufactory/grippers/g2.py` 统一机构几何与标量换算，Torch 路径使用相同公式和参数。
开度／角度换算函数保留签名，角度统一表示实际关节弧度。基础换算无需 Torch、Genesis 或 trimesh。
仿真直接加载项目 URDF；已删除 `derive_urdf()` 和导出的 `derive_g2_urdf`。

## 抓放和包装

```bash
python examples/pick_place/run.py --robot xarm6 --mode sim --executor servo_cartesian \
  --position-mm 0 --speed-percent 50 --force-percent 50 \
  --report reports/g2/pick.preflight.json --csv reports/g2/pick.csv --sim-report reports/g2/pick.json
python examples/packaging/run.py --robot xarm6 --mode sim --executor servo_j --cycles 2 \
  --speed-percent 50 --force-percent 50 \
  --report reports/g2/pack.preflight.json --csv reports/g2/pack.csv --sim-report reports/g2/pack.json
```

其他机型替换为 `xarm5`、`xarm7` 或 `uf850`，两种执行器均适用。
YAML 参数为 `simulation.g2.position_mm`、`speed_percent` 和 `force_percent`，命令行优先。
包装 `--speed` 缩放机械臂路径和观察时间，`--speed-percent` 控制夹爪。
多轮记录使用 `.cycle-N` 文件名。

夹取要求双侧法向力均大于 0.2 N，平均力连续两帧进入当前开度对应目标力的 ±5%。
开爪要求 84±1 mm、双侧法向力均不超过 0.2 N，连续两帧。
确认后下一控制周期抬升、退回；包装进入已有落箱观察阶段。G2 仿真删除 `grip-settle`。
两阶段超时均为 `max(2 秒, 参考行程 / 指令速度 + 1 秒)`。
超时停止后续机械臂动作，保存失败 JSON 和原始 CSV；未指定路径时保存在 `reports/g2/failures/`。

场景和执行程序在预检前完成构建。夹持条件下的预检按物体宽度校验标定机构几何，
执行时仍须实际双侧到力确认才允许机械臂继续。碰撞距离和搬运验收门限不变。

`--report` 保存预检；`--sim-report` 保存完成状态、超时阶段、实际阶段时长、
到力／开爪确认数据、下一动作间隔以及时序和搬运分项结果；`--csv` 保存每个控制周期。
超时或时序、搬运验收不合格时，命令返回失败；搬运检查包括 5 mm 滑移和（桌面抓放的）10 mm 放置门限。
确定性 G2 任务默认启用 `simulation.noslip_iterations = 5`；CPU 搬运回归要求滑移不超过 1 mm。

## 动态演示与 RL

```bash
python examples/visualization/view_gripper_g2.py --headless --speed-percent 50 --force-percent 50
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo \
  --headless --position-mm 0 --speed-percent 50 --force-percent 50

python -m examples.rl.pick_place.train --num-envs 32 --max-iterations 5 --log-dir reports/g2/rl-short
python -m examples.rl.pick_place.train --num-envs 32 --max-iterations 1 \
  --resume reports/g2/rl-short/model_4.pt --log-dir reports/g2/rl-resume
python -m examples.rl.pick_place.evaluate --allow-physics-transfer --headless -B 1 \
  --episodes 16 --summary-json reports/g2/legacy-evaluation.json
```

现有 RL 任务仍为 xArm6，动作、观测维度、奖励及策略流程保持原接口。
位置由策略指令决定，速度和力来自 YAML。训练记录包含有效参数及当前资源、配置哈希。
固定 URDF 后重新计算哈希，历史检查点和清单保持原文件，兼容性仍严格校验。
内置权重继续保留原 `g2_stable_v1_3_3` 版本和文件；跨版本评估必须显式加
`--allow-physics-transfer`，仍校验原始检查点及配置完整性。
旧动力学检查点不能完整续训，历史成功率不代表新动力学表现。

G2 测试分为标定、约束、批量、确认、回放、CPU 搬运及 GPU 验收，共 7 个文件。
GPU 保留 22 项：抓放 9 项、故障 4 项、装箱 5 项（7 轮）、RL 2 项（1／4 环境）、
独立夹爪与 xArm6 动态演示 2 项。串行运行 GPU 矩阵，再运行 CPU 搬运回归：

```bash
python -m pytest tests/simulation/test_g2_gpu.py -q
python -m pytest tests/simulation/test_g2_transport_cpu.py -q
project-check fast
```

`project-check fast` 和 `project-check deep` 沿用现有标记入口。
矩阵在 `reports/g2/` 保存原始记录，时序和搬运质量分开报告。
独立动力学保留原验收标准：中央速度误差 10%、端点误差 1 mm、
稳定力误差 `max(2 N, 5%)`、关节力矩不超限。
