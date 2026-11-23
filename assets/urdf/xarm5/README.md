# xArm5 1305 Assets

动力学 URDF：`xarm5_1305.urdf`；GLB 视觉 URDF：`xarm5_1305_visual.glb.urdf`（物理与前者相同）。末端组合不再提供静态 URDF，由 `ufactory.robots.assembly` 按 [`assembly.yaml`](../assembly.yaml) 运行时生成（如 `compose_robot_urdf("xarm5_1305", "gripper_g2")`）。

## 网格：全部引用 xArm6

xArm5 外观与 xArm6 相同，唯一区别在 J4：xArm6 的 J4 是谐波关节，xArm5 在同一位置是铝合金连接键，相当于 xArm6 的 J4 固定在 0。因此本目录不存网格，视觉和碰撞网格都引用 [xArm6](../xarm6/README.md) 的文件（`../xarm6/meshes/xarm6_1305/`）。运动学、惯量、连杆名和关节名仍是 xArm5 自己的。

| xArm5 连杆 | 引用的 xArm6 零件 | 碰撞凸包 |
|---|---|---|
| link_base / link1 / link2 | 同名连杆 | 2 / 2 / 5 |
| link3 | link3 + link4；link4 按 J4 = 0 摆放（xyz `0.0775 0.3425 0`，rpy `-1.5708 0 0`） | 5 + 5 |
| link4 | link5 | 2 |
| link5（法兰） | link6 | 1 |

- 共 22 个凸包。xArm5 的 J4、J5 即 xArm6 的 J5、J6。`tests/robots/test_asset_integrity.py` 校验各连杆位姿和网格摆放都与 J4 = 0 时的 xArm6 一致。
- 惯量不共用：xArm5 link3 为 2.121 kg，比 xArm6 link3 + link4 轻 1.07 kg（没有 J4 模组）。
- 碰撞凸包的分解方法与伪自碰撞说明见 [xArm6 碰撞网格](../xarm6/README.md#碰撞网格)。xArm5 姿态 `[90, -90, -60, 160, -90]°` 等价于 xArm6 的 `[90, -90, -60, 0, 160, -90]°`。

v0.3.8 之前，xArm5 有一套自己的网格，存在以下问题，均已删除，可从 Git 历史获取：

- 底座 STL 与碰撞凸包是样机 B 版；
- link5 STL 未建孔；
- link5 GLB 在法兰平面内偏了 1.0 mm。

## Source / License

- URDF 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护 GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
