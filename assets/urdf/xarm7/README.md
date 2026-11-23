# xArm7 1305 Assets

动力学 URDF：`xarm7_1305.urdf`；GLB 视觉 URDF：`xarm7_1305_visual.glb.urdf`（物理与前者相同）。末端组合不再提供静态 URDF，由 `ufactory.robots.assembly` 按 [`assembly.yaml`](../assembly.yaml) 运行时生成（如 `compose_robot_urdf("xarm7_1305", "gripper_g2")`）。

## 碰撞网格

```
meshes/xarm7_1305/
├── collision/*.stl   # 注册 URDF 的机械臂选择性多凸包碰撞 STL（23 个凸体）
├── visual/*.stl      # STL 视觉与 CAD 几何参考资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 共 26 个水密凸包，`link_base..link7` 分别为 `2/2/5/4/5/5/2/1` 个凸体，单凸体 `<= 128` 面。`link1..link6` 的 23 个凸包是本目录的 `collision/xarm7_link*_col_*.stl`，`link_base` 与 `link7` 的 3 个引用 xArm6（见“共用件”）。`visual/*.stl` 保留供 STL `<visual>` 与 CAD 几何参考使用，末端配件按各自资源保留独立 collision STL。分解方法与伪自碰撞说明见 [xArm6 碰撞网格](../xarm6/README.md#碰撞网格)。

视觉 GLB（`meshes/xarm7_1305/visual_glb/`）经原生 Draco 压缩，本目录合计约 0.65 MB。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## 共用件

底座 `link_base` 和法兰连杆 `link7` 的视觉与碰撞网格，引用 [xArm6](../xarm6/README.md#共用件) 的 `link_base` 和 `link6`：

- 底座：只用出货的 A 版；
- 法兰：xArm5/6/7 用的是同一零件。

v0.3.8 之前，本目录有这两个连杆的副本，其中底座 GLB 是样机 B 版。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护 GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
