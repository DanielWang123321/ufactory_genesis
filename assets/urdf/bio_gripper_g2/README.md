# Bio Gripper G2 Assets

柔性仿生夹爪（公共可动模块，可独立使用或搭配 xArm / UF850）。

## 行程（重要）

两指夹持面间距的真实机械范围：**闭合 71 mm → 张开 150 mm**。

- 驱动关节为右指 prismatic dof，左指 `mimic ×-1` 对称跟随。
- **关节零位 = 闭合（71 mm）**，dof 增大到 `0.0395 m` 时张开到 150 mm（单指行程 39.5 mm）。
- 间距与关节值近似线性：`gap ≈ 0.071 + 2 × right_joint_value`（米）。
- 单指运动速度范围：`41.67–333.33 mm/s`（对应电机 `500–4000 rpm`，双指相向开合速度 `83.33–666.67 mm/s`）；单指夹持力 `5–20 N`（由虚功原理折算驱动关节力上限 `40 N`）。
- 这些数值集中在 `ufactory/grippers/bio_g2.py`、`ufactory/simulation/bio_g2_model.py` 以及各 URDF 的关节 `limit`（`lower="0" upper="0.0395" effort="40" velocity="0.333333333333333"`）中。

## 模板 URDF

`bio_gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的 standalone 模板文件；v0.1.6 中 mesh 路径已改为本仓库内的 `meshes/visual/*.stl`，可用于基础结构解析。它也是关节 `limit`（行程）的唯一真源。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口。

Bio G2 的 STL collision 文件使用 `meshes/collision/bio_gripper_g2_*_convex_*.stl` 唯一文件名（底座 1 个凸包 + 左右手指各 2 个水密凸包，手指在基座系 `X=0.05 m` 处切分为指杆与指根，指杆裁剪至 `±35.5 mm` 夹持平面）。不要把组合 URDF 改回 `link_base.stl` / `left_finger.stl` / `right_finger.stl`：Genesis 经 MuJoCo 回读 URDF collision mesh 时会按 mesh basename 建资产表，机械臂自身也有 `link_base.stl`，短名会让 Bio G2 底座误用机械臂底座 collision。

组合 URDF 和 `bio_gripper_g2_movable_visual.urdf` 中的 Bio G2 凸包 collision 已将几何偏移烘焙至网格顶点（`<collision><origin xyz="0 0 0"/>`），可动手指 GLB `<visual><origin>` 通过 Y 轴微调偏移与 `71–150 mm` 物理夹持平面严格对齐；整爪总质量标定为 `0.79 kg`（底座含转接板 `0.765827 kg`、左指 `0.012105 kg`、右指 `0.012068 kg`）。

Genesis standalone 检查可使用 `bio_gripper_g2_movable_visual.urdf`。实际机械臂搭配优先使用 `robot_visual_glb_urdf(..., with_bio_gripper_g2=True)` 返回的组合 URDF。

## 目录结构

```
bio_gripper_g2/
├── bio_gripper_g2.urdf                      # standalone 模板 URDF
├── bio_gripper_g2_movable_visual.urdf       # 仅夹爪可动视觉 URDF（标定动力学与凸包碰撞）
├── meshes/
│   ├── collision/
│   │   ├── bio_gripper_g2_base_link_convex_0.stl
│   │   ├── bio_gripper_g2_left_finger_convex_0.stl / _1.stl
│   │   └── bio_gripper_g2_right_finger_convex_0.stl / _1.stl
│   └── visual/
│       ├── bio_gripper_g2_visual.glb        # 合并静态高模 GLB（全机型共用，原生 Draco）
│       ├── visual_glb/bio_gripper_g2_*.glb  # 可动分体高模 GLB（全机型共用，原生 Draco）
│       ├── bio_gripper_g2_left_finger.stl / bio_gripper_g2_right_finger.stl
│       └── bio_gripper_g2_link_base.stl     # 原始 STL 视觉网格
```

视觉保持 link6 高模面数（静态 ~479k / 可动 ~273k tris），用原生 `draco_encoder`（非 DracoPy.encode）压 **视觉 GLB** 体积（合计目标 **< 1 MiB**）。collision STL 保持原密度，不做抽面。

## 加载方式

同一份可动夹爪定义有两种加载入口：

```python
from ufactory.robots.paths import (
    bio_gripper_g2_movable_visual_urdf,  # 独立（无机械臂）
    robot_visual_glb_urdf,  # 搭配机械臂
)

# 1) 独立可动夹爪
urdf = bio_gripper_g2_movable_visual_urdf()

# 2) 搭配 xArm5/6/7 或 UF850（movable=True 取可动版）
urdf = robot_visual_glb_urdf("xarm6_1305", with_bio_gripper_g2=True, movable=True)
# 静态高模（不可动）：去掉 movable=True
```

支持的 arm + movable Bio Gripper G2 组合已经生成在各机械臂资产目录中：

| 机械臂 | 可动组合 URDF |
|--------|---------------|
| xArm5 | `xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf` |
| xArm6 | `xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf` |
| xArm7 | `xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf` |
| UF850 | `uf850_bio_gripper_g2_movable_visual.glb.urdf` |

这些 URDF 同时包含 `<visual>` GLB 与 `<collision>` 中的 `meshes/collision/bio_gripper_g2_*_convex_*.stl`（`<collision><origin xyz="0 0 0"/>`），并通过可动手指 `<visual><origin>` 的 Y 轴微调使视觉与碰撞夹持面在 `71–150 mm` 全行程对齐。

## 控制（公共模块）

几何级连杆控制使用 `ufactory.grippers.bio_g2.BioGripperG2`（自动发现夹爪关节并镜像左指），物理标定斜坡与力控使用 `ufactory.simulation.BioG2GripperController`：

```python
from ufactory.grippers.bio_g2 import BioGripperG2

gripper = BioGripperG2(robot)  # robot 为加载上述任一 URDF 得到的 entity
if gripper.found:
    gripper.setup_pd()
    gripper.set_pose(gripper.OPEN_POS)  # 张开到 150 mm（OPEN_POS）
    gripper.control_pose(gripper.CLOSE_POS)  # 闭合到 71 mm（CLOSE_POS）

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
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --movable --gripper-demo
```

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF、共享 `visual_glb/`（取消按 link5/6/7 重复副本）与 Draco 压缩 GLB。
