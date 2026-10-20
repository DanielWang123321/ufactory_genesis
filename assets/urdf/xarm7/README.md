# xArm7 1305 Assets

动力学 URDF：`xarm7_1305.urdf`（及 G2 / bio_gripper / GLB 组合变体）。

## 碰撞网格

```
meshes/xarm7_1305/
├── visual/*.stl      # 注册 URDF 的 arm collision 与 STL 视觉资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `visual/*.stl`；末端配件按各自资源保留独立 collision STL。

Genesis 1.4.4 默认将每个机器人碰撞网格取单凸包后再简化，碰撞查看器显示加载后的几何。单凸包会填平源网格的凹处；源 STL 的准确外形不能直接保证仿真碰撞外形准确。历史与加载处理对照见 [xArm5 碰撞说明](../xarm5/README.md#碰撞网格)。

视觉 GLB（`meshes/xarm7_1305/visual_glb/`）经原生 Draco 压缩，单臂合计约 0.7–1.0 MB 量级。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
