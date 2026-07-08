# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.2.1-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.2.0%2B-lightgrey" alt="Genesis">
</p>

UFACTORY 机器人模型与 Genesis 仿真工具集 — 高保真 GLB 可视化、运动学校准、轨迹抓放示例与强化学习环境。

[English](README.md) | [贡献指南](CONTRIBUTING.md) | [变更日志](CHANGELOG.md)

## 目录

- [快速开始](#快速开始)
- [支持机型](#支持机型)
- [GLB 视觉预览](#glb-视觉预览)
- [轨迹抓放](#轨迹抓放)
- [展示场景](#展示场景xarm6--gripper-g2-物理装箱)
- [API 快速参考](#api-快速参考)
- [真机运动学校准](#真机运动学校准按-sn-判断)
- [xArm 6 — 参考机型](#xarm-6)
- [项目结构](#项目结构)
- [参与贡献](#参与贡献)
- [开源协议](#开源协议)
- [引用](#引用)

## 快速开始

已在 Python 3.13、Genesis 1.2.0、PyTorch 2.10.0+cu128 下验证。

```bash
# 1. 安装 Genesis（按平台选择：CPU / CUDA / macOS / AMD）
#    参考官方指南：https://genesis-world.readthedocs.io/
pip install "genesis-world>=1.2.0"

# 2. 安装 ufactory_genesis
pip install -r requirements.txt
pip install -e .

# 可选依赖：
#   pip install -e ".[real]"      # xArm SDK / 真机命令
#   pip install -e ".[rl]"        # RL 训练与评估示例
#   pip install -e ".[showcase]"  # 物理装箱展示 scipy 依赖

export NUMBA_CACHE_DIR=~/.cache/numba

# 预览 xArm 6 GLB 模型
python examples/view_robot_glb.py --robot xarm6
```

2024 年起新发货 xArm 均为 **XI1305** 硬件版本。短名 `xarm5` / `xarm6` / `xarm7` 会解析为 `xarm5_1305` / `xarm6_1305` / `xarm7_1305`；显式 `*_1305` 键名仍兼容。旧型号码（11、12、1300–1304）不在本仓库内置，请通过 `--urdf` 或 `prepare_robot_model_for_verification(robot_model=...)` 传入自有 URDF。

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

GLB 用于高精度 PBR 渲染；注册机械臂连杆使用 visual STL 作为碰撞网格。Gripper G2 与 Lite6 Gripper 保留随包 STL 路径，但文件内容与 xarm_ros2 上游 visual STL 一致；Bio Gripper G2 与 Lite6 Vacuum 使用 visual STL 碰撞网格。统一入口：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# 仅机械臂
python examples/view_robot_glb.py --robot <profile_key>

# Gripper G2（静态 / 可动开合）
python examples/view_robot_glb.py --robot xarm6 --gripper-g2
python examples/view_robot_glb.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2（静态）
python examples/view_robot_glb.py --robot uf850 --bio-gripper-g2

# Lite6 平行夹爪（静态 / 可动开合）
python examples/view_robot_glb.py --robot lite6 --lite6-gripper
python examples/view_robot_glb.py --robot lite6 --lite6-gripper --movable --gripper-demo

# Lite6 真空吸盘（静态）
python examples/view_robot_glb.py --robot lite6 --lite6-vacuum-gripper
```

各目录下 `view_*_glb.py`（如 `examples/xarm6/view_xarm6_glb.py`）等价于 `view_robot_glb.py --robot <key>`；xArm6 专用脚本额外提供 `--diagnose`。

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

v0.2.1 新增五机型共享的路点/LSPB 抓取-放置流程。下面这些入口都会调用 `examples/_grasp_place_traj.py`，先生成混合笛卡尔路点程序，再按 LSPB 轨迹采样执行。

| 机型 | 命令 | 说明 |
|------|------|------|
| xArm5 | `python examples/xarm5/xarm5_grasp_place_traj.py --headless --rate 50` | 仅仿真和 dry-run |
| xArm6 | `python examples/xarm6/xarm6_grasp_place_traj.py --headless --rate 50` | 仿真、dry-run、真机 `servo-cartesian` |
| xArm7 | `python examples/xarm7/xarm7_grasp_place_traj.py --headless --rate 50` | 仿真、dry-run、真机机械臂运动 |
| UF850 | `python examples/uf850/uf850_grasp_place_traj.py --headless --rate 50` | 仿真、dry-run、真机机械臂运动 |
| Lite6 | `python examples/lite6/lite6_grasp_place_traj.py --headless --rate 50` | Lite6 反装夹爪，30 mm 方块 |

同一组脚本也支持 Genesis 可视化和真机安全 dry-run：

```bash
# Genesis 三维窗口，GLB visual + STL collision。
python examples/xarm6/xarm6_grasp_place_traj.py --visual --rate 50

# 真机路径 dry-run：只打印分段和 servo 安全检查，不运动机械臂。
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --dry-run --rate 50 --z-min-mm 0

# 真机机械臂执行；确认物理夹爪安装并测试正常后，才添加 --real-gripper。
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --ip 192.168.1.xx --z-min-mm 0 --no-dry-run
```

更多命令、镜像模式和 SDK 仿真验证见 [examples/README_cn.md](examples/README_cn.md)。

## 展示场景（xArm6 + Gripper G2 物理装箱）

黄色桌面（臂固定在桌面长边）、真实物理抓取红色木块、放入开口快递纸箱。GLB 高模 G2 可动 combo + 碰撞/惯性一体。首次运行需生成纸箱贴图：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python scripts/generate_showcase_textures.py

# 完整展示（默认循环）
python examples/xarm6/xarm6_g2_showcase.py

# 单周期后保持画面；加快节奏
python examples/xarm6/xarm6_g2_showcase.py --no-loop --speed 1.5
```

| 参数 | 说明 |
|------|------|
| `--table-height` | 桌面顶面高度（米，默认 0.75） |
| `--speed` | 动作速度倍率（>1 更快） |
| `--loop` / `--no-loop` | 是否循环 pick-place（默认循环） |

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
| `ufactory.get_robot_runtime_profile(key)` | typed runtime profile |

高级 API 从所属功能模块导入：

| 功能域 | 规范导入路径 |
|--------|--------------|
| 机器人注册、路径、运行参数 | `ufactory.robots.registry`, `ufactory.robots.paths`, `ufactory.robots.runtime` |
| 运动学校准与 FK/IK 验证 | `ufactory.kinematics.calibration`, `ufactory.kinematics.validation` |
| 动力学仿真与真机验证 | `ufactory.dynamics` |
| 真机 SDK/session/hold 观测 | `ufactory.hardware.xarm`, `ufactory.hardware.session`, `ufactory.hardware.observe` |
| 夹爪命令转换与控制器 | `ufactory.grippers.g2`, `ufactory.grippers.bio_g2` |
| 轨迹与抓放辅助 | `ufactory.trajectory`, `ufactory.manipulation.frames` |
| GLB 可视化 | `ufactory.visualization.glb` |
| 策略部署 | `ufactory.deploy` |

```python
from ufactory.kinematics.calibration import build_calibrated_urdf
from ufactory.dynamics import dynamics_default_configs
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sdk_pos_mm
from ufactory.visualization.glb import enable_glb_pbr_surfaces
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

示例 SN：`XI130506D43A0A` → 型号码 `1305`（xArm6，需标定）。

```bash
# 仅当 SN 规则允许时才会导出；suffix 默认为 SN 最后 6 字符
python scripts/gen_kinematics_params.py <ip>

# 通用 FK/IK 验证（--robot 见上表「支持机型」；带 --ip 时自动解析 suffix）
python examples/fk_verify_robot.py --robot xarm6 --ip <ip>
python examples/ik_verify_robot.py --robot lite6 --ip <ip>
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-sim-check --robot uf850 --random-count 0 --z-min-mm 0 --require-reference
dynamics-sim-check --robot lite6 --random-count 0 --z-min-mm 0 --require-reference
dynamics-sim-check --robot xarm5 --random-count 0 --z-min-mm 0 --require-reference
dynamics-sim-check --robot xarm7 --random-count 0 --z-min-mm 0 --require-reference
dynamics-hardware-check --robot xarm6 --ip <ip>
dynamics-hardware-check --robot uf850 --ip 192.168.1.55 --z-min-mm 0
dynamics-sim-collision-check --robot uf850 --ip 192.168.1.55   # 仿真模式串联自碰撞预检
```

UF850 / Lite6 / xArm5 / xArm7 默认动力学验证姿态来自
[`assets/configs/dynamics_validation_pose.yaml`](assets/configs/dynamics_validation_pose.yaml)（每机型 20 点均匀插值）；
可通过 `ufactory.dynamics.poses_config` 读取和扩展。

## xArm 6

xArm 6 是本仓库参考机型，`examples/xarm6/` 保留兼容入口；新的通用入口优先使用 `--robot`，例如 `examples/view_robot_glb.py --robot xarm6 --diagnose` 与 `examples/packaging_showcase.py --robot xarm6 --gripper-g2`。

## 项目结构

```
ufactory/robots/          # 机型注册、资产路径、运行参数
ufactory/kinematics/      # 运动学校准与 FK/IK 验证
ufactory/dynamics/        # 动力学仿真、验证、报告与 CLI
ufactory/hardware/        # xArm SDK/session 与 hold 电流观察
ufactory/grippers/        # 夹爪命令转换与控制器
ufactory/trajectory/      # 轨迹 profile、segment、仿真/真机执行器
ufactory/manipulation/    # 任务坐标系辅助
ufactory/visualization/   # GLB/PBR 可视化辅助
ufactory/deploy/          # 策略部署辅助
assets/                   # URDF、mesh、配置与场景资产
examples/                 # 使用示例（预览、FK/IK、RL）
scripts/                  # 用户可用辅助脚本
tests/                    # 贡献者 Pytest 回归
```

## 参与贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、代码风格、资产流水线和 PR 流程。

本项目遵循 [Contributor Covenant](CODE_OF_CONDUCT.md) 行为准则。

## 开源协议

MIT — 详见 [LICENSE](LICENSE)。

## 引用

如在研究中使用 genesis-ufactory，请引用：

```bibtex
@misc{genesis-ufactory,
  author = {UFACTORY},
  title = {genesis-ufactory: UFACTORY Robot Models for Genesis Simulation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/DanielWang123321/ufactory_genesis}
}
```
