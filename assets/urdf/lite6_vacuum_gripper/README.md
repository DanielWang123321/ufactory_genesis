# Lite6 Vacuum Gripper Assets

Lite6 真空吸盘是静态配件，没有开合关节。Visual GLB 为高模 + 原生 Draco（< 1MB）；STL（含 collision / visual 引用）保持原密度，不做抽面。

## URDF

`lite6_vacuum_gripper.urdf` 来自 xarm_ros2（`0.14 kg`、`link_tcp` 偏移 `0.061 m`），是组合源：`compose_robot_urdf("lite6", "lite6_vacuum_gripper")` 视觉改用 `lite6_vacuum_gripper_visual_link6.glb`，`visual="stl"` 保留 `vacuum_gripper_lite.stl`；碰撞均为 `meshes/visual/vacuum_gripper_lite.stl`。原 standalone collision 入口 `lite6_vacuum_gripper_collision.urdf` 已删除。

## Source / License

- URDF / STL 来自上游 xarm_ros2（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护 Draco 压缩视觉 GLB。
