# Gripper G2 Assets

标准 xArm/UF850 平行夹爪。

## 模板 URDF

`gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的 standalone 模板文件；v0.1.6 中 mesh 路径已改为本仓库内的 `meshes/collision/*.stl`，可用于基础结构解析。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口。

Genesis standalone collision 检查使用 `gripper_g2_movable_visual.urdf`：文件名保留 visual 历史命名，但 `view_accessory_collision.py` 以 `vis_mode="collision"` 渲染 `<collision>` 中的 STL。实际机械臂搭配仍应优先使用 `robot_visual_glb_urdf(..., with_gripper_g2=True)` 返回的组合 URDF。

## 目录结构

```
gripper_g2/
├── gripper_g2.urdf                   # standalone 模板 URDF
├── gripper_g2_movable_visual.urdf    # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   ├── collision/                    # STL 碰撞网格（保持原密度，不做抽面）
│   │   ├── base_link.stl
│   │   ├── left_finger.stl / right_finger.stl
│   │   ├── left_inner_knuckle.stl / right_inner_knuckle.stl
│   │   └── left_outer_knuckle.stl / right_outer_knuckle.stl
│   └── visual/                       # GLB 视觉网格（link6 canonical + 原生 Draco）
│       ├── gripper_g2_static.glb     # 合并静态高模（全机型共用）
│       └── visual_glb/               # 可动分体 GLB（base + fingers/knuckles）
```

视觉面数保持不变，用原生 `draco_encoder` 压 **视觉 GLB**（合计目标 **< 1 MiB**）。xArm5/xArm7 通过 visual origin 吸收相对 link6 的法兰平移差；collision STL 与 collision origin 不动。

## 加载方式

机械臂场景使用组合 URDF：

```python
from ufactory.robots.paths import robot_visual_glb_urdf

# 静态 GLB
urdf = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True)

# 可动 GLB
urdf = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True, movable=True)
```

## Collision 检查

```bash
# 仅夹爪，STL collision 网格循环开合
python dev/ref_scripts/view_accessory_collision.py --accessory gripper-g2 --movable --gripper-demo

# xArm6 + Gripper G2，STL collision 网格循环开合
python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --gripper-g2 --movable --gripper-demo

# 固定开/合两态对比
python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --gripper-g2 --movable --gripper-state open
python dev/ref_scripts/view_pose_collision.py --robot xarm6 --pose 4 --gripper-g2 --movable --gripper-state closed
```

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护 standalone / 组合 URDF、共享 `visual_glb/`（link6 canonical）与 Draco 压缩 GLB。
