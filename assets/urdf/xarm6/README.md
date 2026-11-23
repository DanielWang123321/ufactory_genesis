# xArm6 1305 Assets

公开树仅维护 **1305** 变体：动力学 URDF `xarm6_1305.urdf`；GLB 视觉 URDF：`xarm6_1305_visual.glb.urdf`（物理与前者相同）。末端组合不再提供静态 URDF，由 `ufactory.robots.assembly` 按 [`assembly.yaml`](../assembly.yaml) 运行时生成（如 `compose_robot_urdf("xarm6_1305", "gripper_g2")`）。轨迹场景 `visual_model='stl'` 使用 `compose_robot_urdf("xarm6_1305", "gripper_g2", visual="stl")`。

## 碰撞网格

```
meshes/xarm6_1305/
├── collision/*.stl   # 注册 URDF 的机械臂选择性多凸包碰撞 STL（22 个凸体）
├── visual/*.stl      # STL 视觉与 CAD 几何参考资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `collision/xarm6_link*_col_*.stl`（共 22 个水密凸包，`link_base..link6` 分别为 `2/2/5/5/5/2/1` 个凸体，单凸体 `<= 128` 面）；`visual/*.stl` 保留供 STL `<visual>` 与 CAD 几何参考使用，末端配件按各自资源保留独立 collision STL。

Genesis 1.4.x 默认 `convexify=True` 且 `decompose_robot_error_threshold=inf`，会把每个碰撞网格直接取成单凸包，填平弯曲连杆的凹面。单凸包下，`link2`/`link3`/`link4` 的包络外扩 P95 为 `40.3/30.0/35.5 mm`，J5 在 `165°～180°`（如 `[90, -90, -60, 0, 175, -90]°`）时会出现 `link4↔link6` 伪自碰撞和力矩饱和。在 URDF 中把凹连杆预分解为紧贴表面的多个水密凸包后，外扩 P95 降至 `6.5/5.8/4.6 mm`，同时保持纯凸 SupportField+MPR 快速路径。

视觉 GLB（`meshes/xarm6_1305/visual_glb/`）经原生 Draco 压缩；完整性测试预算 `< 1.5 MB`（典型约 0.7–1.1 MB）。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## 共用件

本目录的网格被以下臂直接引用，修改会同时影响它们（映射见 `tests/robots/test_asset_integrity.py` 的 `_ARM_PART_SOURCES`）：

- [xArm5](../xarm5/README.md)：全部连杆。xArm5 即 J4 固定在 0 的 xArm6。
- [xArm7](../xarm7/README.md)：底座 `link_base`，以及法兰连杆（xArm7 link7 = 本目录 `link6`）。

底座只用出货的 A 版：平顶 z = 150.35 mm，主体 Ø95.4。其中：

- STL 与上游 `xarm6_1305` 相同；
- GLB 来自 CAD 导出 `1305-xarm5-P01-95-3`。v0.3.8 用它替换了原来的样机 B 版 GLB `1305-defeature-P01-0429`。

B 版（主体 Ø98.0，顶部 Ø57.8 圆台高至 152.55 mm）只装在样机上。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护 GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
