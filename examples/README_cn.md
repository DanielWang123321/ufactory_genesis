# examples/

[English](README.md)

示例脚本目录索引。安装、抓放模式与真机安全流程见仓库根目录 [README.zh.md](../README.zh.md)。

## 快速开始

| 文件 | 说明 |
|------|------|
| `view_robot_glb.py` | 统一 GLB 查看器，支持全部机型与末端执行器 |

```bash
python examples/view_robot_glb.py --robot xarm6
```

## 机器人可视化

各机型目录下的包装脚本等价于 `view_robot_glb.py --robot <key>`：

| 目录 | 预览命令 |
|------|----------|
| `xarm5/` | `python examples/xarm5/view_xarm5_glb.py` |
| `xarm6/` | `python examples/xarm6/view_xarm6_glb.py` |
| `xarm7/` | `python examples/xarm7/view_xarm7_glb.py` |
| `lite6/` | `python examples/lite6/view_lite6_glb.py` |
| `uf850/` | `python examples/uf850/view_uf850_glb.py` |

## 夹爪演示

| 目录 | 说明 |
|------|------|
| `gripper_g2/` | Gripper G2 可动视觉演示 |
| `bio_gripper_g2/` | Bio Gripper G2 可动视觉演示 |
| `lite6_gripper/` | Lite6 平行夹爪可动视觉演示 |

## 机械臂验证

| 文件 | 说明 |
|------|------|
| `verify_robot.py` | 通用正运动学/位置控制冒烟测试（所有机型） |
| `fk_verify_robot.py` | 通用正运动学验证（可与真机对比） |
| `ik_verify_robot.py` | 通用逆运动学验证（可与真机对比） |
| `packaging_showcase.py` | 通用展示场景入口（当前支持 xArm6 + Gripper G2） |

动力学验证使用安装后的控制台命令：

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip>
```

## 轨迹抓放

五个脚本是已安装命令 `ufactory-grasp-place`（`ufactory.cli.grasp_place`）的薄包装。完整的 `--mode` / `--executor` 说明见根目录 [README.zh.md](../README.zh.md#轨迹抓放)。

| 文件 | 说明 |
|------|------|
| `xarm5/xarm5_grasp_place_traj.py` | `--robot xarm5` 包装 |
| `xarm6/xarm6_grasp_place_traj.py` | `--robot xarm6` 包装 |
| `xarm7/xarm7_grasp_place_traj.py` | `--robot xarm7` 包装 |
| `uf850/uf850_grasp_place_traj.py` | `--robot uf850` 包装 |
| `lite6/lite6_grasp_place_traj.py` | `--robot lite6` 包装 |

```bash
python examples/xarm6/xarm6_grasp_place_traj.py --mode dry-run --executor servo_j
# 等价：
# ufactory-grasp-place --robot xarm6 --mode dry-run --executor servo_j

# 真机 + 运动学镜像窗口（详见根 README）：
# ufactory-grasp-place --robot xarm6 --mode real --executor servo_j \
#   --calibration path/to/exact.yaml --confirm-real --visual
```

## xArm 6 — 参考实现

xArm 6 拥有最完整的示例覆盖。

### 运动学

| 文件 | 说明 |
|------|------|
| `xarm6/verify_xarm6.py` | 正运动学 + 位置控制冒烟测试 |
| `xarm6/fk_verify.py` | 兼容入口，默认 `--robot xarm6` |
| `xarm6/ik_verify.py` | 兼容入口，默认 `--robot xarm6` |

### 强化学习

| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_reach_env.py` / `_train.py` | 到达任务环境与训练 |
| `xarm6/xarm6_reach_deploy.py` | 到达任务部署辅助：对齐 / 离线预检仍可用；**v0.2.5 已硬禁用**在线真机策略 `deploy` 与 `smoke-random`（见 [SECURITY.md](../SECURITY.md)） |
| `xarm6/xarm6_grasp_place_env.py` / `_train.py` / `_eval.py` | 抓放强化学习任务 |

### 展示场景

| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_g2_showcase.py` | xArm6 + Gripper G2 物理装箱演示 |

装箱方块/home 的名义坐标与抓放示例完全一致，纸箱/投放中心也使用抓放目标 XY `(0.300, 0.300)`。纸箱基座系尺寸为 `300 × 200 × 150 mm`，长边沿基座 X；展示基座绕 Z 轴 `+90°` 后，画面中长边沿世界 Y。Genesis 只对抓取/home 列使用 X `+2.5 mm` 补偿和 `1.5` 倍机械臂刚度，释放点仍是无补偿箱心；抓取只依靠接触摩擦且不绑定方块。闭合/搬运只驱动 G2 主关节；到达箱心后保留静止等待，再用 `0.2 s` 将计划夹口从 `22 mm` 缓释到 `29 mm`，随后短暂同步驱动六个联动关节，夹指可见间距增加 `3 mm` 后由主关节与右外指节继续张开，方块离开后完成全开。每个 `20 ms` 指令周期使用 32 个物理子步，接触摩擦不变。方块从箱口上方 50 mm 自然下落。dry-run、SDK 仿真和真机使用无补偿的名义轨迹。通用入口支持版本化布局、安全预检和单次真机执行：

```bash
# 仿真默认单次；有限多轮和无限循环需显式指定
python examples/packaging_showcase.py --mode sim --executor servo_j
python examples/packaging_showcase.py --mode sim --executor servo_j --cycles 3
python examples/packaging_showcase.py --mode sim --executor servo_j --loop

python examples/packaging_showcase.py --mode dry-run --executor servo_j
python examples/packaging_showcase.py --mode dry-run --executor servo_cartesian
XARM_IP=192.168.1.65 python examples/packaging_showcase.py --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real
```

## 内部模块

以 `_` 前缀的文件是内部共享模块，不是用户入口：

| 文件 | 用途 |
|------|------|
| `_bootstrap.py` | 将项目根目录加入 `sys.path` |
| `_robot_viewer.py` | 共享 Genesis GLB 查看器核心 |
| `_gripper_demo.py` | Gripper G2 开合控制 |
| `_bio_gripper_g2_demo.py` | Bio Gripper G2 开合控制 |
| `_lite6_gripper_demo.py` | Lite6 夹爪开合控制 |
| `_packaging_scene.py` | 展示场景构建器 |
| `_grasp_place_traj.py` | 遗留共享模块（测试仍引用）；用户路径为 `ufactory-grasp-place` |
| `_standalone_gripper_viewer.py` | 独立夹爪查看器辅助 |
