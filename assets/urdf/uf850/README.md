# UF850 Assets

动力学 URDF：`uf850.urdf`（及 G2 / bio_gripper / GLB 组合变体）。

## 碰撞与视觉

```
meshes/uf850/
├── collision/*.stl   # vendor 简化壳（历史/对比资源）
├── visual/*.stl      # 注册 URDF 的 arm collision 与 STL 视觉资源
└── visual_glb/       # Draco 压缩视觉 GLB（单臂合计约 0.7–1.0 MB）
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `visual/*.stl`；末端配件按各自资源保留独立 collision STL。

## Source / License

- URDF / STL 源自上游 UFACTORY / xArm ROS 描述家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
