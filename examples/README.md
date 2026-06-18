# examples/

使用示例和教程。

## 快速开始

| 文件 | 说明 |
|------|------|
| `view_robot_glb.py` | **统一 GLB 查看器** — 支持全部机型和末端执行器 |

```bash
python examples/view_robot_glb.py --robot xarm6_1305
```

## 机器人可视化

每个机器人目录包含独立的 GLB 预览脚本：

| 目录 | 预览命令 |
|------|----------|
| `xarm5/` | `python examples/xarm5/view_xarm5_glb.py` |
| `xarm6/` | `python examples/xarm6/view_xarm6_glb.py` |
| `xarm7/` | `python examples/xarm7/view_xarm7_glb.py` |
| `lite6/` | `python examples/lite6/view_lite6_glb.py` |
| `uf850/` | `python examples/uf850/view_uf850_glb.py` |

## 夹爪演示

| 目录/文件 | 说明 |
|-----------|------|
| `gripper_g2/` | Gripper G2 可动视觉演示 |
| `bio_gripper_g2/` | Bio Gripper G2 可动视觉演示 |
| `lite6_gripper/` | Lite6 平行夹爪可动视觉演示 |

## 机械臂验证

| 文件 | 说明 |
|------|------|
| `verify_robot.py` | 通用 FK/PD 冒烟测试（所有机型） |
| `fk_verify_robot.py` | 通用 FK 验证（可与真机对比） |
| `ik_verify_robot.py` | 通用 IK 验证（可与真机对比） |

## xArm 6 — 参考实现

xArm 6 拥有最完整的示例覆盖：

### 运动学与动力学
| 文件 | 说明 |
|------|------|
| `xarm6/verify_xarm6.py` | FK + PD 冒烟测试 |
| `xarm6/verify_xarm6_dynamics.py` | 动力学验证 |
| `xarm6/fk_verify.py` | FK 与真机 SDK 对比 |
| `xarm6/ik_verify.py` | IK 与真机 SDK 对比 |
| `xarm6/run_fk_alignment_cycle.py` | FK 对齐循环 |

### 强化学习
| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_reach_env.py` / `_train.py` | Reach 任务环境与训练 |
| `xarm6/xarm6_grasp_place_env.py` / `_train.py` / `_eval.py` | Grasp-place 任务 |

### 展示场景
| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_g2_showcase.py` | xArm6 + Gripper G2 物理装箱演示 |

## 内部模块

以 `_` 前缀的文件是内部共享模块，被多个示例引用：

| 文件 | 用途 |
|------|------|
| `_bootstrap.py` | 添加项目根目录到 sys.path |
| `_robot_viewer.py` | 共享 Genesis GLB 查看器核心 |
| `_gripper_demo.py` | Gripper G2 开合控制 |
| `_bio_gripper_g2_demo.py` | Bio Gripper G2 开合控制 |
| `_lite6_gripper_demo.py` | Lite6 夹爪开合控制 |
| `_packaging_scene.py` | 展示场景构建器 |
