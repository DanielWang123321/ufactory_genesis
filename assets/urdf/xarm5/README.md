# xArm5 1305 Assets

动力学 URDF：`xarm5_1305.urdf`（及 G2 / bio_gripper / GLB 组合变体）。

## 碰撞网格

```
meshes/xarm5_1305/
├── collision/*.stl   # 动力学/Genesis 自碰撞（vendor 简化壳，由 OBJ 转为 STL）
└── visual/*.stl      # 仅视觉
```

`<collision>` 必须引用 `collision/`，不得指向 `visual/`。
