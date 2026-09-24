# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.3.0-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.4.2-lightgrey" alt="Genesis">
  <a href="https://github.com/xArm-Developer/ufactory_genesis/actions/workflows/ci.yml"><img src="https://github.com/xArm-Developer/ufactory_genesis/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

UFACTORY 机械臂的 Genesis 模型与仿真工具，覆盖可视化、运动学、轨迹执行、装箱和仅用于仿真的强化学习。

[English](README.md) | [示例](examples/README_zh.md) | [变更记录](CHANGELOG.md) | [路线图](ROADMAP.md) | [安全说明](SECURITY.md)

> v0.3.0 是 [xArm-Developer](https://github.com/xArm-Developer/ufactory_genesis) 上的官方仓库。项目仍是 0.x：1.0.0 之前可以改名或删除入口。Genesis World 只验证 1.4.2。

## 支持范围

- Python 3.12/3.13。Genesis World 1.4.2 是已验证基线。不保证更高版本可用；钩子检查对不上时直接失败。
- 仅支持克隆源码后的可编辑安装；不支持 wheel、sdist 和远程资产下载。
- 支持可视化、校准后的正向/逆向运动学（FK/IK）、接触仿真、离线预检、控制器仿真和经过安全检查的真机执行。
- 真机装箱只启用 xArm6 + Gripper G2 和 Lite6 + Lite6 Gripper。

| 机械臂 | 可用末端 |
|--------|----------|
| `xarm5` | Gripper G2、Bio Gripper G2 |
| `xarm6` | Gripper G2、Bio Gripper G2 |
| `xarm7` | Gripper G2、Bio Gripper G2 |
| `uf850` | Gripper G2、Bio Gripper G2 |
| `lite6` | Lite6 Gripper、Lite6 Vacuum Gripper |

## 安装

```bash
git clone https://github.com/xArm-Developer/ufactory_genesis.git
cd ufactory_genesis
pip install -e ".[sim]"
export NUMBA_CACHE_DIR=~/.cache/numba
python examples/visualization/view_robot.py --robot xarm6
```

| 目标 | 可编辑安装命令 |
|------|----------------|
| 仿真与可视化 | `pip install -e ".[sim]"` |
| 固定位置强化学习 | `pip install -e ".[sim,rl]"` |
| 真机安全后端 | `pip install -e ".[sim,real]"` |
| 装箱贴图生成 | `pip install -e ".[sim,showcase]"` |
| 动力学参考检查 | `pip install -e ".[sim,dynamics]"` |

Windows 或无可用显卡时，应先安装 CPU 版 PyTorch，并传入 `--backend cpu`。强化学习仍只支持 Linux/NVIDIA；详见[示例指南](examples/README_zh.md)。

若 `view_robot.py` 查看器在 `Building visualizer...` 之后无窗口直接回到提示符，请设置 `MKL_THREADING_LAYER=SEQUENTIAL` 后重跑（PowerShell：`$env:MKL_THREADING_LAYER = "SEQUENTIAL"`）；`--headless` 不受影响。

## 主要工作流

```bash
# GLB 可视化
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable

# 抓放配置检查与仿真
ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j --print-config
ufactory-pick-place --robot xarm6 --mode sim --executor servo_j --visual

# YAML 驱动的装箱仿真
python scripts/generate_showcase_textures.py
ufactory-packaging-showcase --robot lite6 --mode sim --executor servo_cartesian --visual
```

完整的可视化、运动学、CPU/Windows、配置和真机示例见[示例指南](examples/README_zh.md)。

## 当前入口

下面是 v0.3.0 在 Genesis World 1.4.2 上可用的入口，不表示后续 0.x 或更高 Genesis 版本会继续保留。

命令：`ufactory-pick-place`、`ufactory-packaging-showcase`、`dynamics-sim-check`、`dynamics-sim-collision-check`、`dynamics-hardware-check`。`project-check` 和 `observe-hold-current` 是维护者工具。

示例：`examples/visualization`、`examples/kinematics`、`examples/pick_place`、`examples/packaging`、`examples/rl/pick_place`，以及抓放和装箱入口旁的 `runtime.example.yaml`。

`ufactory` 根导出：`get_robot_profile`、`robot_urdf`、`robot_visual_glb_urdf`、`robot_assets`、`RepositoryAssetStore`。

## v0.3.0 固定位置强化学习

公开的 xArm6 + Gripper G2 任务仅使用固定 `+Y` 布局：方块位于 `[0.300, 0, 0.015]` 米，目标位于 `[0.300, 0.300, 0.015]` 米。本版本不包含随机起点。

内置的随机种子 7 模型 `model_299_g2stable.pt` 已在 `g2_stable_v1_3_3` 接触物理配置下重新训练并通过全部发布检查：9 组评估种子/并行数量组合共 219/219 回合、独立固定场景库 64/64、0.02 动作噪声下 512/512。它不能证明随机布局泛化，也没有真机执行接口。

```bash
python -m examples.rl.pick_place.evaluate --expert --headless -B 1 --episodes 1
python -m examples.rl.pick_place.evaluate --headless -B 8 --episodes 8
```

训练方式、发布结果、模型校验和限制见[强化学习指南](examples/rl/pick_place/README_cn.md)。

## 真机安全

真机运动必须使用逐台精确校准、完成预测安全预检，并显式传入 `--confirm-real`。软件检查不能替代受训操作员、隔离工作区、风险评估和实体急停。连接硬件前请阅读 [SECURITY.md](SECURITY.md)。

## 文档

- [按任务组织的示例](examples/README_zh.md)
- [贡献与项目检查](CONTRIBUTING.md)
- [版本变更记录](CHANGELOG.md)
- [项目路线图](ROADMAP.md)
- [安全与信任边界](SECURITY.md)

## 许可与引用

代码采用 MIT 许可；机器人 URDF 和网格资产包含上游材料，其独立归属说明见 [NOTICE](NOTICE)。xArm、UF850 和 UFACTORY 是其各自所有者的商标。

研究使用时请引用：“genesis-ufactory: UFACTORY Robot Models for Genesis Simulation”，Daniel Wang，2026，`https://github.com/xArm-Developer/ufactory_genesis`。
