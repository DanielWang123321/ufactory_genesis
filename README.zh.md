# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.1.0-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.1.2-lightgrey" alt="Genesis">
</p>

UFACTORY 机器人模型与 Genesis 仿真工具集 — 高保真 GLB 可视化、运动学校准与强化学习环境。

[English](README.md) | [贡献指南](CONTRIBUTING.md) | [变更日志](CHANGELOG.md)

## 目录

- [快速开始](#快速开始)
- [支持机型](#支持机型)
- [GLB 视觉预览](#glb-视觉预览)
- [展示场景](#展示场景xarm6--gripper-g2-物理装箱)
- [API 快速参考](#api-快速参考)
- [真机运动学校准](#真机运动学校准按-sn-判断)
- [xArm 6 — 参考机型](#xarm-6)
- [文档](#文档)
- [项目结构](#项目结构)
- [参与贡献](#参与贡献)
- [开源协议](#开源协议)
- [引用](#引用)

## 快速开始

已在 Python 3.13、Genesis 1.1.2、PyTorch 2.12 下验证。

```bash
# 1. 安装 Genesis（按平台选择：CPU / CUDA / macOS / AMD）
#    参考官方指南：https://genesis-world.readthedocs.io/
pip install genesis-world==1.1.2

# 2. 安装 ufactory_genesis
pip install -r requirements.txt
pip install -e .

export NUMBA_CACHE_DIR=~/.cache/numba

# 预览 xArm 6 GLB 模型
python examples/view_robot_glb.py --robot xarm6_1305
```

## 支持机型

| profile key | 机型 | 自由度 | Gripper G2 | Bio Gripper G2 | Lite6 Gripper | Lite6 Vacuum |
|-------------|------|--------|:----------:|:--------------:|:-------------:|:------------:|
| `xarm5_1305` | xArm 5 | 5 | ✓ | ✓ | — | — |
| `xarm6_1305` | xArm 6 | 6 | ✓ | ✓ | — | — |
| `xarm7_1305` | xArm 7 | 7 | ✓ | ✓ | — | — |
| `uf850` | UF850 | 6 | ✓ | ✓ | — | — |
| `lite6` | Lite6 | 6 | — | — | ✓ | ✓ |

**Gripper G2** 与 **Bio Gripper G2** 为 xArm/UF850 共用配件；**Lite6 Gripper**（平行夹爪）与 **Lite6 Vacuum Gripper**（真空吸盘）仅适用于 Lite6。加载末端时各配件互斥，一次只能选一种。

机型能力、已知限制与资产维护见 [docs/multi_robot_compatibility.md](docs/multi_robot_compatibility.md)。

## GLB 视觉预览

GLB 用于高精度 PBR 渲染；碰撞与物理仍走 STL 网格。统一入口：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# 仅机械臂
python examples/view_robot_glb.py --robot <profile_key>

# Gripper G2（静态 / 可动开合）
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2 --movable --gripper-demo

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

```python
import ufactory
```

### 机器人注册表

| 函数 / 对象 | 说明 |
|------------|------|
| `ufactory.ROBOT_PROFILES` | 所有支持机型的 `RobotModelSpec` 字典 |
| `ufactory.get_robot_profile(key)` | 按 profile key 获取 `RobotModelSpec` |
| `ufactory.get_profile_key_for_robot_name(name)` | 机器人名称解析为 profile key |
| `ufactory.arm_link_names(profile)` | 获取某机型的连杆名称元组 |
| `ufactory.joint_names(profile)` | 获取某机型的关节名称元组 |

### 路径工具

| 函数 | 说明 |
|------|------|
| `ufactory.robot_urdf(key)` | 默认 URDF 绝对路径 |
| `ufactory.robot_visual_glb_urdf(key, ...)` | 带 GLB 视觉的 URDF，支持末端执行器 |
| `ufactory.robot_assets(name)` | 机器人资产目录 `Path` |
| `ufactory.xarm6_1305_urdf()` | 便捷函数：xArm6 1305 URDF |
| `ufactory.lite6_visual_glb_urdf(...)` | 便捷函数：Lite6 带夹爪选项的 GLB URDF |

### 运动学校准

| 函数 | 说明 |
|------|------|
| `ufactory.load_kinematics_yaml(path)` | 从运动学 YAML 加载关节偏移 |
| `ufactory.build_calibrated_urdf(base, kinematics)` | 生成含标定关节原点的 URDF |
| `ufactory.parse_sn_model_code(sn)` | 从序列号提取 4 位型号码 |
| `ufactory.has_per_unit_kinematics_calibration(sn, name)` | 判断某 SN 是否需要逐台标定 |

### GLB PBR 视觉

| 函数 | 说明 |
|------|------|
| `ufactory.enable_glb_pbr_surfaces()` | 修补 Genesis 以保留 GLB 的 PBR 材质 |
| `ufactory.glb_view_surface()` | 非 GLB 几何体的默认双面渲染表面 |

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
# 仅当 SN 规则允许时才会导出；旧款 xArm 会提示跳过
python scripts/gen_kinematics_params.py <ip> <suffix>

# 通用 FK/IK 验证（--robot 见上表「支持机型」）
python examples/fk_verify_robot.py --robot xarm6_1305 --ip <ip> --kinematics-suffix <suffix>
python examples/ik_verify_robot.py --robot lite6 --ip <ip> --kinematics-suffix <suffix>
```

## xArm 6

xArm 6 是本仓库参考机型，涵盖运动学/动力学验证与 reach、grasp-place 强化学习示例。完整步骤见 [docs/xarm6_verification.md](docs/xarm6_verification.md)。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/multi_robot_compatibility.md](docs/multi_robot_compatibility.md) | 多机型能力、资产管线、维护者重定位 |
| [docs/xarm6_verification.md](docs/xarm6_verification.md) | xArm6 FK/IK、动力学、RL、pytest |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 项目路线图 |

## 项目结构

```
ufactory/             # 核心 Python 包（机器人注册、路径、运动学、GLB）
assets/urdf/          # 各机型 URDF、STL 碰撞、GLB 视觉 mesh
assets/scenes/        # 仿真场景资产（贴图、道具）
examples/             # 使用示例（预览、FK/IK、RL）
scripts/              # 资产生成与维护脚本
tests/                # Pytest 测试集
docs/                 # 扩展文档
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
