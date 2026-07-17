# Lite6 Vacuum Gripper Assets

Lite6 真空吸盘是静态配件，没有开合关节。Visual GLB 为高模 + 原生 Draco（< 1MB）；STL（含 collision / visual 引用）保持原密度，不做抽面。

## URDF

`lite6_vacuum_gripper.urdf` 来自 xarm_ros2，保留 vendor 的外部父 link，供结构和组合 URDF 参考。`lite6_vacuum_gripper_collision.urdf` 是可直接在 Genesis 中 standalone 加载的 collision 检查入口，不复制 STL，只引用 `meshes/visual/vacuum_gripper_lite.stl`。

## Collision 检查

```bash
# 仅吸盘，静态 STL collision 网格
python dev/ref_scripts/view_accessory_collision.py --accessory lite6-vacuum-gripper

# Lite6 + vacuum，静态 STL collision 网格
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-vacuum-gripper
```

## Source / License

- URDF / STL 来自上游 xarm_ros2（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护 collision 入口 URDF 与 Draco 压缩视觉 GLB。
