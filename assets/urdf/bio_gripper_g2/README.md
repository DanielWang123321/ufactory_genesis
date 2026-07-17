# Bio Gripper G2 Assets

柔性仿生夹爪（公共可动模块，可独立使用或搭配 xArm / UF850）。

## 行程（重要）

两指夹持面间距的真实机械范围：**闭合 71 mm → 张开 150 mm**。

- 驱动关节为右指 prismatic dof，左指 `mimic ×-1` 对称跟随。
- **关节零位 = 闭合（71 mm）**，dof 增大到 `0.0395 m` 时张开到 150 mm（单指行程 39.5 mm）。
- 间距与关节值近似线性：`gap ≈ 0.071 + 2 × right_joint_value`（米）。
- 这些数值集中在 `ufactory/grippers/bio_g2.py`（`CLOSE_POS=0`、`OPEN_POS=0.0395`、`STROKE=0.0395`、`CLOSED_GAP=0.071`、`OPEN_GAP=0.150`）以及模板 URDF 的关节 `limit` 中。

## 模板 URDF

`bio_gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的 standalone 模板文件；v0.1.6 中 mesh 路径已改为本仓库内的 `meshes/visual/*.stl`，可用于基础结构解析。它也是关节 `limit`（行程）的唯一真源。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口。

Bio G2 的 STL collision 文件使用 `bio_gripper_g2_*.stl` 唯一文件名。不要把组合 URDF 改回 `link_base.stl` / `left_finger.stl` / `right_finger.stl`：Genesis 经 MuJoCo 回读 URDF collision mesh 时会按 mesh basename 建资产表，机械臂自身也有 `link_base.stl`，短名会让 Bio G2 底座误用机械臂底座 collision。

组合 URDF 和 `bio_gripper_g2_movable_visual.urdf` 中的 Bio G2 STL collision 通过非零 `<collision><origin>` 对齐 canonical 视觉 GLB。所有机型共用同一套 collision origin（视觉几何已统一）；机型差异只体现在 `bio_gripper_g2_attach` joint origin。不要把这些 collision origin 改回 `0 0 0`，否则 collision 网格会相对 GLB 错位。

Genesis standalone collision 检查使用 `bio_gripper_g2_movable_visual.urdf`：文件名保留 visual 历史命名，但 `view_accessory_collision.py` 以 `vis_mode="collision"` 渲染 `<collision>` 中的 STL。实际机械臂搭配仍应优先使用 `robot_visual_glb_urdf(..., with_bio_gripper_g2=True)` 返回的组合 URDF。

## 目录结构

```
bio_gripper_g2/
├── bio_gripper_g2.urdf                      # standalone 模板 URDF
├── bio_gripper_g2_movable_visual.urdf       # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   └── visual/
│       ├── bio_gripper_g2_visual.glb        # 合并静态高模 GLB（全机型共用，原生 Draco）
│       ├── visual_glb/bio_gripper_g2_*.glb  # 可动分体高模 GLB（全机型共用，原生 Draco）
│       ├── bio_gripper_g2_left_finger.stl / bio_gripper_g2_right_finger.stl
│       └── bio_gripper_g2_link_base.stl     # URDF collision 引用的唯一命名 STL
```

视觉保持 link6 高模面数（静态 ~479k / 可动 ~273k tris），用原生 `draco_encoder`（非 DracoPy.encode）压 **视觉 GLB** 体积（合计目标 **< 1 MiB**）。collision STL 保持原密度，不做抽面。

## 加载方式

同一份可动夹爪定义有两种加载入口：

```python
from ufactory.robots.paths import (
    bio_gripper_g2_movable_visual_urdf,  # 独立（无机械臂）
    robot_visual_glb_urdf,               # 搭配机械臂
)

# 1) 独立可动夹爪
urdf = bio_gripper_g2_movable_visual_urdf()

# 2) 搭配 xArm5/6/7 或 UF850（movable=True 取可动版）
urdf = robot_visual_glb_urdf("xarm6_1305", with_bio_gripper_g2=True, movable=True)
# 静态高模（不可动）：去掉 movable=True
```

支持的 arm + movable Bio Gripper G2 collision 组合已经生成在各机械臂资产目录中：

| 机械臂 | 可动组合 URDF | collision viewer 命令 |
|--------|---------------|-----------------------|
| xArm5 | `xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf` | `python dev/ref_scripts/view_pose_collision.py --robot xarm5 --pose 4 --bio-gripper-g2 --movable --gripper-demo` |
| xArm6 | `xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf` | `python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --bio-gripper-g2 --movable --gripper-demo` |
| xArm7 | `xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf` | `python dev/ref_scripts/view_pose_collision.py --robot xarm7 --pose 4 --bio-gripper-g2 --movable --gripper-demo` |
| UF850 | `uf850_bio_gripper_g2_movable_visual.glb.urdf` | `python dev/ref_scripts/view_pose_collision.py --robot uf850 --pose 4 --bio-gripper-g2 --movable --gripper-demo` |

这些 URDF 文件名保留 `visual.glb` 历史命名；`view_pose_collision.py` 使用 `vis_mode="collision"`，实际渲染 `<collision>` 中的 `bio_gripper_g2_*.stl`，并通过各 link 的 collision origin 对齐同一机型的 GLB 参考位置。

## 控制（公共模块）

两种加载方式都用同一个控制类 `ufactory.grippers.bio_g2.BioGripperG2`，它会自动发现夹爪关节并镜像左指：

```python
from ufactory.grippers.bio_g2 import BioGripperG2

gripper = BioGripperG2(robot)        # robot 为加载上述任一 URDF 得到的 entity
if gripper.found:
    gripper.setup_pd()
    gripper.open()                   # 张开到 150 mm（OPEN_POS）
    gripper.close()                  # 闭合到 71 mm（CLOSE_POS）

# 开合循环：
for step in range(1000):
    gripper.control_pose(gripper.demo_target(step))
    scene.step()
```

预览命令：

```bash
# 独立
python examples/visualization/view_bio_gripper_g2.py
# 搭配机械臂（开合演示）
python examples/visualization/view_robot.py --robot xarm6_1305 --bio-gripper-g2 --movable --gripper-demo
```

Collision 检查命令：

```bash
# 仅夹爪，STL collision 网格循环开合
python dev/ref_scripts/view_accessory_collision.py --accessory bio-gripper-g2 --movable --gripper-demo

# xArm6 + Bio Gripper G2，STL collision 网格循环开合
python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --bio-gripper-g2 --movable --gripper-demo

# 固定开/合两态对比
python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --bio-gripper-g2 --movable --gripper-state open
python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --bio-gripper-g2 --movable --gripper-state closed
```

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF、共享 `visual_glb/`（取消按 link5/6/7 重复副本）与 Draco 压缩 GLB。
