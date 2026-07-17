# Lite6 Gripper Assets

Lite6 平行夹爪。Visual GLB 为高模 + 原生 Draco（合计 < 1MB）；collision / 手指 visual STL 保持原密度，不做抽面。

## 模板 URDF

`lite6_gripper.urdf` 来自 xarm_ros2，mesh 路径已被展平但未被 Lite6 组合 URDF 直接引用。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口；组合 URDF 生成脚本会处理路径替换。

反装外观基准：`assets/urdf/lite6/lite6_gripper_visual.glb.urdf`（`python examples/visualization/view_robot.py --robot lite6 --lite6-gripper`）。可动组合/standalone URDF 使用相同 vendor 手指 joint 帧（`0 0 0.0543`），shell 继续使用 `shell.glb`；可动手指 visual 使用 `finger1.stl` / `finger2.stl`，避免分解 GLB 的跨指间子网格在抓取时看起来插入方块。

可动 URDF 不再给手指 visual 额外添加横向 `<origin>` 偏移；`finger1.stl` / `finger2.stl` 的原始大面积内侧面闭合间隙约 22.4 mm、张开间隙约 40.2 mm，和 Lite6 反装夹爪的 20-38 mm 控制模型接近。抓放轨迹通过抬高 Lite6 抓取高度，让低位指根/限位凸台越过 30 mm 方块上沿，使方块由大面积内侧面夹持。

手指 collision 使用双凸 box 拼出 L 形（内侧垫 + 外侧指根竖板），凹区留空。这样即使 Genesis 对每个 geom 做默认凸化，也不会填平凹口导致顶面假接触；轨迹/包装场景不再对 Lite6 整机关闭 `convexify`/`decimate`/`watertighten`。

轨迹抓放场景与默认可动入口均使用 `assets/urdf/lite6/lite6_gripper_movable_visual.glb.urdf`；standalone collision 调试使用本目录 `lite6_gripper_movable_visual.urdf`。

## 目录结构

```
lite6_gripper/
├── lite6_gripper.urdf                   # 模板 URDF
├── lite6_gripper_movable_visual.urdf    # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   ├── collision/                        # STL（shell 碰撞；手指 STL 仍供 visual）
│   └── visual/                           # GLB 视觉网格
```

## Collision 检查

```bash
# 仅夹爪，collision 网格循环开合
python dev/ref_scripts/view_accessory_collision.py --accessory lite6-gripper --movable --gripper-demo

# Lite6 + Lite6 Gripper，collision 网格循环开合
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-gripper --movable --gripper-demo

# 固定开/合两态对比
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-gripper --movable --gripper-state open
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-gripper --movable --gripper-state closed
```

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF 与 Draco 压缩视觉 GLB（配件合计目标 < 1 MiB）。
