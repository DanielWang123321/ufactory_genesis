# Lite6 Gripper Assets

Lite6 平行夹爪。

## 模板 URDF

`lite6_gripper.urdf` 来自 xarm_ros2，mesh 路径已被展平但未被 Lite6 组合 URDF 直接引用。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口；组合 URDF 生成脚本会处理路径替换。

反装外观基准：`assets/urdf/lite6/lite6_gripper_visual.glb.urdf`（`python examples/view_robot_glb.py --robot lite6 --lite6-gripper`）。可动组合/standalone URDF 使用相同 vendor 手指 joint 帧（`0 0 0.0543`），shell 继续使用 `shell.glb`；可动手指 visual 使用 `finger1.stl` / `finger2.stl`，与碰撞指面完全重合，避免分解 GLB 的跨指间子网格在抓取时看起来插入方块。

指根 visual/collision 对齐由 STL 网格内置偏移加 `<origin>` 共同完成：`finger1.stl` / `finger2.stl` 本身约有 `y=±7.8 mm` 半侧宽度，可动 URDF 再给两指 visual 与 collision 分别加 `y=+10 mm` / `y=-10 mm`，表达 Lite6 反装夹爪真实 20-38 mm 两指开口。

轨迹抓放场景与默认可动入口均使用 `assets/urdf/lite6/lite6_gripper_movable_visual.glb.urdf`；standalone collision 调试使用本目录 `lite6_gripper_movable_visual.urdf`。

## 目录结构

```
lite6_gripper/
├── lite6_gripper.urdf                   # 模板 URDF
├── lite6_gripper_movable_visual.urdf    # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   ├── collision/                        # STL 碰撞网格
│   └── visual/                           # GLB 视觉网格
```

## Collision 检查

```bash
# 仅夹爪，STL collision 网格循环开合
python dev/ref_scripts/view_accessory_collision.py --accessory lite6-gripper --movable --gripper-demo

# Lite6 + Lite6 Gripper，STL collision 网格循环开合
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-gripper --movable --gripper-demo

# 固定开/合两态对比
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-gripper --movable --gripper-state open
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-gripper --movable --gripper-state closed
```
