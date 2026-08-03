# 按任务组织的示例

[English](README.md)

v0.2.11 按任务组织公开示例。共享实现位于 `ufactory`，入口模块负责解析用户参数并启动对应流程。

## 前置条件

1. 克隆本仓库，并在**仓库根目录**运行命令。
2. 使用 editable install（仅支持源码树；不支持 wheel/sdist）：

```bash
pip install -e ".[sim]"

# 可选：
#   pip install -e ".[real]"      # xArm SDK + Pinocchio/Coal（dry-run / sdk-sim / real 安全预检）
#   pip install -e ".[showcase]"  # 装箱纸箱贴图（scipy）
#   pip install -e ".[sim,rl]"    # Linux/NVIDIA 固定布局 RL 示例
```

3. 运行依赖 Genesis 的流程前，设置可写的 Numba 缓存目录：

```bash
# Linux / macOS
export NUMBA_CACHE_DIR=~/.cache/numba

# Windows PowerShell
# $env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
```

4. 支持的 `--robot`：`xarm5`、`xarm6`、`xarm7`、`uf850`、`lite6`。

5. **Windows / 无可用 GPU：** 先装 CPU 版 PyTorch，再 `pip install -e ".[sim]"`。可视化、抓放、装箱命令加 `--backend cpu`（有显卡但不被 Genesis 支持时，不要依赖 GPU 自动回退）。详见根目录 README「Windows 与无可用 GPU 仿真」。

本版公开目录：

| 目录 | 用途 |
|---|---|
| `visualization/` | GLB 机型与夹爪查看器 |
| `kinematics/` | FK/IK / 机型校验包装 |
| `pick_place/` | 多机型抓放入口与可选覆盖配置 |
| `packaging/` | 多机型装箱展示入口与可选覆盖配置 |
| `rl/` | Linux/NVIDIA xArm6 + Gripper G2 固定布局 RL 示例 |

RL 边界见 [rl/README_cn.md](rl/README_cn.md)；安装、评估命令、训练与已知限制见 [rl/pick_place/README_cn.md](rl/pick_place/README_cn.md)。

## 可视化

```bash
# 裸臂
python examples/visualization/view_robot.py --robot xarm6

# Gripper G2（静态组合）
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2

# 可动手指 + 开合演示
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2 / Lite6 夹爪
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --movable --gripper-demo
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper

# 独立夹爪查看器
python examples/visualization/view_gripper_g2.py
python examples/visualization/view_bio_gripper_g2.py
python examples/visualization/view_lite6_gripper.py
```

| 参数 | 说明 |
|------|------|
| `--robot` | 必选机型 key |
| `--gripper-g2` / `--bio-gripper-g2` / `--lite6-gripper` / `--lite6-vacuum-gripper` | 末端组合（最多选一个） |
| `--movable` | 按连杆可动夹爪网格（需配合受支持的夹爪标志） |
| `--gripper-demo` | 循环开合（需要 `--movable`） |
| `--pd` | 平滑关节运动演示（约 0.873 rad/s），仅视觉 |
| `--show-tcp` | 法兰 TCP 红色标记（默认隐藏） |
| `--diagnose` | 无窗口 STL/GLB 连杆位姿诊断 |
| `--headless` | 不打开查看器窗口 |
| `--backend` | `cpu` / `gpu`（默认 `gpu`；无支持显卡时用 `cpu`） |

## 运动学

```bash
# 离线机型/资产自检（不连控制器）
python examples/kinematics/verify_robot.py --robot xarm6

# Genesis URDF 与 xArm SDK 的 FK/IK 对比（需要网络与 IP）
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>
```

动力学验证继续使用控制台命令：

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real
dynamics-sim-collision-check --robot xarm6 --ip <ip>
```

## 抓放

`examples/pick_place/run.py` 委托给稳定的 `ufactory-pick-place` 控制台命令，两者接受相同参数。

| 参数 | 取值 / 说明 |
|------|-------------|
| `--robot` | 必选：`xarm5` / `xarm6` / `xarm7` / `uf850` / `lite6` |
| `--mode` | 必选：`sim`（Genesis）、`dry-run`（离线预检，不连控制器）、`sdk-sim`（控制器仿真）、`real` |
| `--executor` | 必选：`servo_j` 或 `servo_cartesian` |
| `--backend` | 可选：`cpu` / `gpu`；覆盖 `simulation.backend`（无 Genesis 支持显卡时用 `cpu`） |
| `--config` | 可选的严格局部覆盖 YAML |
| `--print-config` | 打印解析后的运行时 YAML 后退出 |
| `--ip` | 控制器 IP（或设置 `XARM_IP`），用于 `sdk-sim` / `real` |
| `--calibration` | 逐台精确运动学 YAML（真机必需） |
| `--confirm-real` | `--mode real` 的显式确认门 |
| `--visual` | `sim`：强制 Genesis viewer；`real`：运动学镜像（不支持 dry-run/sdk-sim） |
| `--report` | 可选报告输出路径 |

```bash
# 仅解析配置
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j --print-config

# 离线预测预检
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Genesis 仿真并打开查看器
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual

# 仅 CPU / 不支持的显卡
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --backend cpu

# 可选覆盖（仅写出的字段覆盖 assets/configs/runtime）
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j \
  --config examples/pick_place/runtime.example.yaml

# 真机运动（必须精确标定 + 显式确认）
XARM_IP=<ip> python examples/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

`runtime.example.yaml` 是严格的局部覆盖文件；未写出的机器人、几何、运动、安全和仿真值仍由 `assets/configs/runtime` 解析。

## 装箱

首次装箱前先生成纸箱贴图（需要 `.[showcase]`）：

```bash
python scripts/generate_showcase_textures.py
```

`examples/packaging/run.py` 委托给 `ufactory-packaging-showcase`。

| 参数 | 取值 / 说明 |
|------|-------------|
| `--robot` | 默认 `xarm6`；五机型均支持 sim / dry-run / sdk-sim |
| `--mode` | 默认 `sim`；与抓放相同的四种模式 |
| `--executor` | 默认 `servo_j`；也可 `servo_cartesian` |
| `--backend` | 可选：`cpu` / `gpu`；含义与抓放相同 |
| `--cycles N` | 精确仿真轮数（默认 1） |
| `--speed` | 仿真播放倍率（`>1` 更快） |
| `--table-height` | 仅覆盖仿真展示高度，不改变基座系几何 |
| `--config` / `--print-config` / `--ip` / `--calibration` / `--confirm-real` / `--visual` / `--report` | 含义与抓放相同 |

```bash
# 一轮仿真结束后退出（加 --visual 可保持最终画面）
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j

# 仅 CPU / 不支持的显卡
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --backend cpu

# 三轮回归
python examples/packaging/run.py \
  --robot lite6 --mode sim --executor servo_j --cycles 3

# 离线预检 / 控制器仿真
python examples/packaging/run.py \
  --robot xarm7 --mode dry-run --executor servo_j
python examples/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml

# 真机装箱仅启用 xArm6 + G2、Lite6 + Lite6 夹爪
XARM_IP=<ip> python examples/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# 可选覆盖
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j \
  --config examples/packaging/runtime.example.yaml
```

真机模式始终只执行一轮。xArm5、xArm7、UF850 的 `--mode real` 会在连接控制器前拒绝，直到启用真实夹爪路径。

## 从 v0.2.6 迁移

| 旧类别 | v0.2.8 路径 |
|---|---|
| `examples/view_robot_glb.py`、各机型查看器 | `examples/visualization/view_robot.py --robot <key>` |
| 根目录/各机型 FK、IK 包装 | `examples/kinematics/verify_{robot,fk,ik}.py` |
| 各机型抓放包装 | `examples/pick_place/run.py --robot <key>` |
| 根目录/xArm6 装箱包装 | `examples/packaging/run.py --robot <key>` |

旧路径、bootstrap 文件和下划线前缀的示例内部模块均已直接删除，不提供兼容壳。v0.2.11 RL 入口仅支持 `examples.rl.pick_place` 模块方式。
