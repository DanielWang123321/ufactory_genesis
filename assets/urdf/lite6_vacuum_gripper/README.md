# Lite6 Vacuum Gripper Assets

Lite6 真空吸盘是静态配件，没有开合关节。

## URDF

`lite6_vacuum_gripper.urdf` 来自 xarm_ros2，保留 vendor 的外部父 link，供结构和组合 URDF 参考。`lite6_vacuum_gripper_collision.urdf` 是可直接在 Genesis 中 standalone 加载的 collision 检查入口，不复制 STL，只引用 `meshes/visual/vacuum_gripper_lite.stl`。

## Collision 检查

```bash
# 仅吸盘，静态 STL collision 网格
python dev/ref_scripts/view_accessory_collision.py --accessory lite6-vacuum-gripper

# Lite6 + vacuum，静态 STL collision 网格
python dev/ref_scripts/view_pose_collision.py --robot lite6 --pose 4 --lite6-vacuum-gripper
```
