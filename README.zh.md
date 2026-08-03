# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.2.12-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.3.1-lightgrey" alt="Genesis">
  <a href="https://github.com/DanielWang123321/ufactory_genesis/actions/workflows/ci.yml"><img src="https://github.com/DanielWang123321/ufactory_genesis/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

UFACTORY 机器人模型与 Genesis 仿真工具集 — 高保真 GLB 可视化、运动学校准，以及轨迹抓放 / 装箱示例。

[English](README.md) | [贡献指南](CONTRIBUTING.md) | [变更日志](CHANGELOG.md) | [路线图](ROADMAP.md) | [安全策略](SECURITY.md)

## 项目状态

- **0.2.x 为 Alpha / 预览版。** 公开 API 可能在次版本之间破坏；二次开发请钉住 git tag。详见 [ROADMAP.md](ROADMAP.md)。
- 当前托管在**个人 GitHub 账号**（`DanielWang123321/ufactory_genesis`）。**不是** UFACTORY 官方组织仓库，也**不是**官方产品支持渠道。
- **计划在 0.3.x 迁入 UFACTORY 官方 GitHub 组织**，并作为首个面向用户的支持面。在此之前，将 0.2 文档与 API 视为过渡。
- xArm / UF850 / UFACTORY 等为权利方商标；本仓库不宣称获得官方 SDK 背书。

## 验证边界

| 层级 | 能证明什么 | 在哪里跑 |
|------|------------|----------|
| **公开 CI**（等价 `project-check fast`） | 静态检查、部分类型检查、CPU 单元测试 | GitHub Actions（无 GPU、不安装 Genesis） |
| **本地仿真**（`project-check sim`） | 进程内 GPU / Genesis 回归 | 维护者机器，需 `[sim]` |
| **维护者真机**（`sdk-sim` / `hardware`） | 柜控 SDK 仿真与所列真机 | 维护者实验室；可选脱敏摘要挂在 GitHub Release |

**参考（锁定）基线**为 Python 3.13、Genesis World 1.3.1、PyTorch 2.10.0+cu128；RL 使用 RSL-RL 5.4.2。更高 Genesis 版本仅在运行时兼容钩子仍匹配时可运行；在本地仿真与真机检查通过前，**不**视为维护者已核实的物理或硬件基线。公开 CI **不能**替代上述检查。

## 目录

- [项目状态](#项目状态)
- [验证边界](#验证边界)
- [快速开始](#快速开始)
- [Windows 与无可用 GPU 仿真](#windows-与无可用-gpu-仿真)
- [支持机型](#支持机型)
- [GLB 视觉预览](#glb-视觉预览)
- [轨迹抓放](#轨迹抓放)
- [RL 抓放示例](#rl-抓放示例)
- [展示场景](#展示场景yaml-驱动装箱)
- [API 快速参考](#api-快速参考)
- [真机运动学校准](#真机运动学校准按-sn-判断)
- [xArm 6 — 参考机型](#xarm-6)
- [项目结构](#项目结构)
- [参与贡献](#参与贡献)
- [开源协议](#开源协议)
- [引用](#引用)

## 快速开始

v0.2.12 仅支持克隆源码仓库后 editable install；不支持 wheel/sdist、远程资产下载或脱离 Git 源码树安装。仓库资产缺失时会以 `AssetLayoutError` 明确失败。最低要求 Genesis World 1.3.1。视觉 GLB 已用 Draco 压缩（典型 `assets/` 检出约数十 MB 量级）。

```bash
# 在已克隆仓库根目录
pip install -e ".[sim]"

# 可选依赖：
#   pip install -e ".[real]"      # xArm SDK + Pinocchio/Coal 安全后端
#   pip install -e ".[sim,rl]"    # RL 抓放示例 + ufactory.training
#   pip install -e ".[showcase]"  # 物理装箱展示 scipy 依赖

export NUMBA_CACHE_DIR=~/.cache/numba

# 预览 xArm 6 GLB 模型
python examples/visualization/view_robot.py --robot xarm6

# 本地 CPU 质量检查（与公开 CI 同级；不替代仿真/真机证据）
project-check fast
```

2024 年起新发货 xArm 均为 **XI1305** 硬件版本。短名 `xarm5` / `xarm6` / `xarm7` 会解析为 `xarm5_1305` / `xarm6_1305` / `xarm7_1305`；显式 `*_1305` 键名仍兼容。旧型号码（11、12、1300–1304）不在本仓库内置，请通过 `--urdf` 或 `prepare_robot_model_for_verification(robot_model=...)` 传入自有 URDF。

## Windows 与无可用 GPU 仿真

抓放 / 装箱的 `--mode sim`（以及 GLB 预览）可在 Windows、以及没有 Genesis 支持显卡的机器上运行。强化学习不在本路径范围内。

**先安装 CPU 版 PyTorch**，再安装本包（Genesis World ≥ 1.3.1）：

```bash
# Linux / Windows — CPU torch（最新 CPU 源见 https://pytorch.org）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[sim]"
# 装箱贴图（可选）: pip install -e ".[showcase]"
```

默认 `simulation.backend` 为 `gpu`（维护机优先 GPU）。若机器有显卡但**不被 Genesis 支持**（例如仍会被 CUDA 枚举到的旧卡），**不要依赖** GPU→CPU 自动回退，请强制 CPU：

```bash
python examples/visualization/view_robot.py --robot xarm6 --backend cpu
python examples/pick_place/run.py --robot lite6 --mode sim --executor servo_cartesian --backend cpu
python examples/packaging/run.py --robot xarm6 --mode sim --executor servo_j --backend cpu
```

也可在 `--config` 覆盖里写 `simulation.backend: cpu`。可选设置 `QD_ENABLE_CUDA=0` 跳过 CUDA 探测。保持默认 `GS_ENABLE_NDARRAY=1`。Windows 上不要设置 `PYOPENGL_PLATFORM=osmesa`。

**Windows（PowerShell）Numba 缓存：**

```powershell
$env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
# 不支持的显卡上可选：
# $env:QD_ENABLE_CUDA="0"
```

`.[real]` / Pinocchio dry-run 不在本条 Windows/无可用 GPU 目标内。

## 支持机型

| profile key | 别名 | 机型 | 自由度 | Gripper G2 | Bio Gripper G2 | Lite6 Gripper | Lite6 Vacuum |
|-------------|------|------|--------|:----------:|:--------------:|:-------------:|:------------:|
| `xarm5_1305` | `xarm5` | xArm 5 | 5 | ✓ | ✓ | — | — |
| `xarm6_1305` | `xarm6` | xArm 6 | 6 | ✓ | ✓ | — | — |
| `xarm7_1305` | `xarm7` | xArm 7 | 7 | ✓ | ✓ | — | — |
| `uf850` | — | UF850 | 6 | ✓ | ✓ | — | — |
| `lite6` | — | Lite6 | 6 | — | — | ✓ | ✓ |

✓ = 提供 combo URDF（静态 GLB visual）；Gripper G2、Bio Gripper G2、Lite6 Gripper 另支持 `--movable` 开合动画。

**Gripper G2** 与 **Bio Gripper G2** 为 xArm/UF850 共用配件；**Lite6 Gripper**（平行夹爪）与 **Lite6 Vacuum Gripper**（真空吸盘）仅适用于 Lite6。加载末端时各配件互斥，一次只能选一种。

## GLB 视觉预览

统一入口 `examples/visualization/view_robot.py`，支持各机型与末端：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# 仅机械臂
python examples/visualization/view_robot.py --robot <profile_key>

# Gripper G2（静态 / 可动开合）
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2（静态）
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2

# Lite6 平行夹爪（静态 / 可动开合）
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --movable --gripper-demo

# Lite6 真空吸盘（静态）
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper
```

独立可动夹爪查看器位于同一目录：`view_gripper_g2.py`、`view_bio_gripper_g2.py` 和 `view_lite6_gripper.py`。

| 参数 | 产品 | 说明 |
|------|------|------|
| `--gripper-g2` | Gripper G2 | 加载 combo URDF |
| `--movable` | Gripper G2 / Lite6 Gripper / Bio Gripper G2 | 分 link GLB（开合动画必需） |
| `--gripper-demo` | Gripper G2 / Bio Gripper G2 / Lite6 Gripper | 夹爪开合循环演示 |
| `--bio-gripper-g2` | Bio Gripper G2 | 静态 GLB 叠加 |
| `--lite6-gripper` | Lite6 Gripper | Lite6 平行夹爪 combo URDF |
| `--lite6-vacuum-gripper` | Lite6 Vacuum Gripper | Lite6 真空吸盘静态 GLB |
| `--pd` | 机械臂 | 关节演示（50°/s 平滑插值，非高增益 PD） |
| `--show-tcp` | 机械臂 | 显示 EE 法兰红色 TCP 调试标记（默认隐藏） |

## 轨迹抓放

v0.2.12 使用一个配置驱动入口覆盖五机型。`dry-run` 不连接控制器，会完成校准 FK、全时间线运动学统计以及 Pinocchio/Coal 全采样碰撞检查。

| 机型 | 命令 | 说明 |
|------|------|------|
| xArm5 | `ufactory-pick-place --robot xarm5 --mode dry-run --executor servo_j` | 两种执行器 |
| xArm6 | `ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j` | 两种执行器 |
| xArm7 | `ufactory-pick-place --robot xarm7 --mode dry-run --executor servo_j` | 两种执行器 |
| UF850 | `ufactory-pick-place --robot uf850 --mode dry-run --executor servo_j` | 两种执行器 |
| Lite6 | `ufactory-pick-place --robot lite6 --mode dry-run --executor servo_j` | 真机夹爪为二值能力 |

```bash
# 仅解析最终配置，不初始化 Genesis、不连接机械臂。
ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j --print-config

# 离线预测预检。
ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j

# 真机必须使用逐台严格校准并显式确认。
# 任务轨迹从 default_qpos 规划；若当前姿态偏离起点，--confirm-real 会先用
# MODE_POSITION 的 set_servo_angle 预定位，再切换 servo 流式执行。
XARM_IP=192.168.1.xx ufactory-pick-place --robot xarm6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real

# 真机 + 运动学镜像窗口（开环 teleport，异步更新，不进 servo 关键路径）。
XARM_IP=192.168.1.xx ufactory-pick-place --robot xarm6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real --visual
```

`--visual`：`--mode sim` 强制打开 Genesis viewer；`--mode real` 打开 kinematic mirror（非接触物理）。通用抓放镜像使用有上限的非阻塞更新；装箱命令在独立进程中跑满速 Genesis/GLB viewer，按正常 60 Hz 重绘消费全部 50 Hz 镜像状态，不与 servo 发送端共享同一 Python 调度器。窗口保持打开直至关闭或 Ctrl+C。`dry-run` / `sdk-sim` 不支持该标志。

v0.2.7 已从公开包删除在线真机策略部署、随机动作、策略会话和 SDK 策略动作适配器（见 [SECURITY.md](SECURITY.md)）。v0.2.12 的 RL 示例仍仅用于仿真，不增加任何真机策略执行接口。

## RL 抓放示例

原有 xArm6 + Gripper G2 固定 `+Y` 主线继续提供规范配方、固定 512 回合场景库和约 2.7 MB 的 `model_199.pt`。v0.2.12 在 `random_start/` 下新增“方块起点随机、目标固定”的独立主线，分别提供配方、场景库、评估门禁和指引。其随仓库检查点明确采用分层结构：规范布局保留固定 RL actor，非规范布局使用读取仿真状态的脚本 guide，不会描述成 PPO 已学习的随机布局泛化。两条主线都只用于仿真。

```bash
pip install -e ".[sim,rl]"
python -m examples.rl.pick_place.evaluate --episodes 1
```

这些示例仅支持 Linux + NVIDIA GPU。固定布局与随机起点的正式评估、训练、指标和限制请从[中文 RL 索引](examples/rl/README_cn.md)或[英文 RL 索引](examples/rl/README.md)进入。

## 展示场景（YAML 驱动装箱）

`ufactory-packaging-showcase` 的 `sim`、`dry-run`、`sdk-sim` 覆盖 xArm5、xArm6、xArm7、UF850、Lite6，并同时支持 `servo_j` 与 `servo_cartesian`。物理场景、轨迹、时序、成功阈值、接触白名单和夹爪几何均来自版本化 YAML；通用配置为 `assets/configs/runtime/tasks/packaging_showcase.yaml`，Lite6 叠加 `assets/configs/runtime/tasks/robots/lite6_packaging_showcase.yaml`。

| 机型/末端 | sim / dry-run / SDK 仿真 | 完整真机装箱 |
|-----------|:------------------------:|:------------:|
| xArm6 + Gripper G2 | ✓ | ✓ |
| Lite6 + Lite6 夹子 | ✓ | ✓ |
| xArm5 / xArm7 / UF850 + 配置中的 G2 模型 | ✓ | 未启用真实夹爪前禁用 |

仿真只依靠接触与摩擦，不会绑定方块。G2 保留已验证的 `22→29 mm / 0.2 s` 缓释和单控制周期全开目标；Lite6 禁用预释放缓释，在 `0.5 s` 释放/稳定段开始时一次下达二值全开目标。Lite6 使用更近的方块/home（`[0.200, 0, 0.015]` / `[0.200, 0, 0.200]`），箱心采用最近的完整预检安全位置 `(0.200, 0.220)`；固定 `300 × 200 mm` 箱体置于 `(0.200, 0.150)` 时会侵入 Lite6 link4 的拾取包络。首次运行需生成纸箱贴图：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python scripts/generate_showcase_textures.py

# 任选五机型；默认执行一轮并保持最终画面。
ufactory-packaging-showcase --robot xarm6 --mode sim --executor servo_j
ufactory-packaging-showcase --robot lite6 --mode sim --executor servo_cartesian

# 发布回归：指定机型/执行器连续三轮。
ufactory-packaging-showcase --robot lite6 --mode sim --executor servo_j --cycles 3

# 离线碰撞预检与控制器仿真。
ufactory-packaging-showcase --robot xarm7 --mode dry-run --executor servo_j
ufactory-packaging-showcase --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml

# 真机仅启用 xArm6 + G2、Lite6 + Lite6 夹子。
XARM_IP=<ip> ufactory-packaging-showcase --robot lite6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real

# 按任务组织的示例入口：委托给同一个 CLI。
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j
```

`servo_j` 启动时会先构建 Genesis 逆运动学场景，再对完整轨迹做安全预检。终端以 `[ik-compile]` 和 `[preflight]` 标出阶段、采样数及耗时；`preflight=PASS` 之前真机只做身份校验，运动保持未授权。Genesis 的 neutral self-collision 过滤提示和 Quadrants 的 `ast.keyword(..., ctx=...)` Python 3.15 弃用提示属于上游信息，在当前验证过的 Python 3.13 / Genesis 1.3.1 组合中不是失败。碰撞预检仍检查每个轨迹采样和每个几何对，但先使用配置的 5 mm 安全裕量筛选候选，仅对候选计算精确距离；若后端不支持该能力则自动回退到全距离检查。

| 参数 | 说明 |
|------|------|
| `--robot` | `xarm5`、`xarm6`、`xarm7`、`uf850` 或 `lite6` |
| `--table-height` | 仅覆盖仿真展示高度，不改变基座系装箱几何 |
| `--speed` | 仿真播放倍率（>1 更快） |
| `--cycles N` | 精确执行 N 次仿真任务（默认 1 次） |
| `--loop` / `--no-loop` | 显式无限循环 / 兼容的单次别名 |

多轮仿真会在每轮恢复方块位置、单位四元数及零速度；抓起、落箱或回零失败会停止后续轮次。真机必须使用逐台精确标定、显式确认且始终只执行一轮；`servo_cartesian` 会在同一进程取得匹配的 SDK 仿真证据。xArm5、xArm7、UF850 的 `--mode real` 会在连接控制器前拒绝，不会跳过夹爪后伪报装箱成功。

释放后主装箱场景不会重置方块、清零速度、绑定刚体或缩放回弹。`ufactory.manipulation.packaging.measure_natural_drop()` 使用配置中的方块、箱底和求解器参数进行隔离诊断，记录碰撞前后速度、最高回弹和静止时间，但不要求出现明显弹跳。

### 可选依赖矩阵

| 工作流 | 安装方式 | Pinocchio / Coal |
|---|---|---|
| 可视化与接触仿真 | `.[sim]` | 不安装、不导入 |
| 固定布局 RL 示例与 `ufactory.training` | `.[sim,rl]` | 不安装、不导入 |
| 真机预测安全 | `.[real]` | 必需；缺失时在运动前明确失败 |
| 独立动力学参考 | `.[dynamics]` | 必需；缺失时给出可执行的错误信息 |

公开 RL 目录严格限定为 `examples/rl/pick_place`；私人实验归档不进入公开示例。

## API 快速参考

项目采用按功能域划分的子包。根命名空间刻意保持精简：

```python
import ufactory
```

| 根 API | 说明 |
|--------|------|
| `ufactory.ROBOT_PROFILES` | 支持机型注册表 |
| `ufactory.get_robot_profile(key)` | 按 profile key 或短名解析机型 |
| `ufactory.robot_cli_choices()` | 排序后的 `--robot` 选项 |
| `ufactory.robot_urdf(key, name=None)` | 默认或指定 URDF 的绝对路径 |
| `ufactory.robot_visual_glb_urdf(key, with_*=..., movable=...)` | GLB 视觉 URDF 绝对路径 |
| `ufactory.robot_assets(name)` | 机器人资产目录 |
| `ufactory.kinematics_user_dir(robot_name)` | 逐台标定 YAML 目录 |
| `ufactory.RepositoryAssetStore` | 校验源码资产布局与清单 |

高级 API 从所属功能模块导入：

| 功能域 | 规范导入路径 |
|--------|--------------|
| 机器人注册与源码资产 | `ufactory.robots.registry`, `ufactory.robots.paths` |
| 版本化运行配置 | `ufactory.config` |
| 运动学校准与 FK/IK 验证 | `ufactory.kinematics.calibration`, `ufactory.kinematics.validation` |
| 动力学报告与验证服务 | `ufactory.dynamics` |
| 真机 SDK/session/hold 观测 | `ufactory.hardware.xarm`, `ufactory.hardware.session`, `ufactory.hardware.observe` |
| 夹爪命令转换与控制器 | `ufactory.grippers.g2`, `ufactory.grippers.bio_g2` |
| 经批准的规划/预检/执行 | `ufactory.trajectory`, `ufactory.safety` |
| GLB 可视化 | `ufactory.visualization.glb` |
| 装箱几何/场景/诊断 | `ufactory.manipulation.packaging` |
| RL 配置与安全产物 | `ufactory.training` |

```python
from ufactory.config import load_runtime_config
from ufactory.safety import SafetyGate, ApprovedProgram
from ufactory.trajectory import preflight_program, execute_real
```

v0.2.0 已移除旧根模块入口，例如 `ufactory.paths`、`ufactory.robot_params`、
`ufactory.kinematics_validation`、`ufactory.real_robot_session` 和
`ufactory.dynamics_validation`。

## 真机运动学校准（按 SN 判断）

控制柜内**逐台运动学补偿**是否可用，可由 SN 第 3–6 位（四位型号码）判断：

| 机型 | SN 型号码 | 是否有补偿 |
|------|-----------|------------|
| xArm 5/6/7 | `< 1304` | **一定没有** — 使用标称 URDF，勿传 `--kinematics-*` |
| xArm 5/6/7 | `≥ 1304`（如 1305） | 可能有 — 需从本机提取 YAML |
| Lite6 | `< 1006` | **一定没有** |
| Lite6 | `≥ 1006` | 可能有 |
| UF850 | 任意 | **一定有** |

示例 SN：`XI130506XXXXXX` → 型号码 `1305`（xArm6，需标定）。

```bash
# 仅当 SN 规则允许时才会导出；suffix 默认为 SN 最后 6 字符
python scripts/gen_kinematics_params.py <ip>

# 通用 FK/IK 验证（--robot 见上表「支持机型」；带 --ip 时自动解析 suffix）
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real
dynamics-sim-collision-check --robot xarm6 --ip <ip>   # 仿真模式串联自碰撞预检
# 其他机型：将 --robot 换成 xarm5 / xarm7 / uf850 / lite6。
```

UF850 / Lite6 / xArm5 / xArm7 默认动力学验证姿态来自
[`assets/configs/dynamics_validation_pose.yaml`](assets/configs/dynamics_validation_pose.yaml)（每机型 20 点均匀插值）；
可通过 `ufactory.dynamics.poses_config` 读取和扩展。

## xArm 6

xArm 6 仍是本仓库参考机型，但 v0.2.12 不发布各机型包装目录。请通过按任务组织的入口选择它，例如 `examples/visualization/view_robot.py --robot xarm6` 或 `examples/packaging/run.py --robot xarm6 ...`。完整的 v0.2.6 路径迁移表见 [examples/README_cn.md](examples/README_cn.md)。

## 项目结构

```
ufactory/config/          # 版本化运行时 YAML 加载与哈希
ufactory/safety/          # SafetyGate、ApprovedProgram、碰撞/时序端口
ufactory/cli/             # 控制台入口（抓放、装箱）
ufactory/quality/         # 本地 project-check 报告
ufactory/robots/          # 机型注册、资产路径、运行参数
ufactory/kinematics/      # 运动学校准与 FK/IK 验证
ufactory/dynamics/        # 动力学仿真、验证、报告与 CLI
ufactory/hardware/        # xArm SDK/session 与 hold 电流观察
ufactory/grippers/        # 夹爪命令转换与控制器
ufactory/trajectory/      # 轨迹 profile、segment、仿真/真机执行器
ufactory/manipulation/    # 任务坐标与可复用装箱实现
ufactory/simulation/      # 共享 Genesis 运行时所有权
ufactory/training/        # 场景库、验收配置、安全检查点 YAML/清单辅助
ufactory/visualization/   # GLB/PBR 可视化辅助
assets/                   # URDF、mesh、配置与场景资产
examples/                 # 可视化、运动学、抓放、装箱与 rl/pick_place 入口
scripts/                  # 用户辅助脚本与维护者工具（见 scripts/README.md）
tests/                    # 贡献者 Pytest 回归
```

私人实验归档仍在被忽略的根目录 `/rl/`；公开 RL 仅 `examples/rl/`。

## 参与贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、代码风格、资产流水线和 PR 流程。

本项目遵循 [Contributor Covenant](CODE_OF_CONDUCT.md) 行为准则。

## 开源协议

- 本仓库**代码**为 MIT — 详见 [LICENSE](LICENSE)。
- **机器人 URDF / 网格资产**含源自上游 UFACTORY / xArm ROS 软件包的内容；归属与许可说明见 [NOTICE](NOTICE)。请勿将整个 `assets/` 树视为纯原创 MIT 内容。

## 引用

如在研究中使用 genesis-ufactory，请引用：

```bibtex
@misc{genesis-ufactory,
  author = {Wang, Daniel},
  title = {genesis-ufactory: UFACTORY Robot Models for Genesis Simulation},
  year = {2026},
  note = {Preview 0.2.x on personal GitHub; planned 0.3.x move to a UFACTORY organization repository},
  publisher = {GitHub},
  url = {https://github.com/DanielWang123321/ufactory_genesis}
}
```
