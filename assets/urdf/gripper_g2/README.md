# Gripper G2

UFACTORY Gripper G2 parallel gripper for the `xarm` tool flange (xArm5/6/7, UF850).

## Files

```
gripper_g2/
├── gripper_g2_movable_visual.urdf    # part URDF: composition source and standalone preview
└── meshes/
    ├── collision/
    │   ├── base_link_col_00.stl … base_link_col_09.stl   # base: 10 convex hulls
    │   ├── left_finger.stl / right_finger.stl             # fingers and knuckles: full-density STLs
    │   ├── left_inner_knuckle.stl / right_inner_knuckle.stl
    │   └── left_outer_knuckle.stl / right_outer_knuckle.stl
    └── visual/
        ├── base_link.stl             # full base mesh, the base's visual="stl" mesh
        ├── gripper_g2_static.glb     # single static GLB (gripper_visual="static")
        └── visual_glb/               # per-link GLBs: base, fingers, knuckles
```

The GLBs are Draco-compressed, 0.50 MB in total (test budget < 1 MB). With `visual="stl"` the base shows `visual/base_link.stl` (`stl_visuals` in [`assembly.yaml`](../assembly.yaml)); the fingers and knuckles show their collision STLs.

## Model

- Frame: all meshes are in the flange frame of the official SolidWorks model `Gripper G2 AG1200 20250710.STEP`, with the flange mounting face at Z = 0 and the flange axis through the origin. Every `<visual>` and `<collision>` origin is `0 0 0`.
- Reference heights: locating boss bottom −5.0 mm, base top 104.5 mm, outer / inner knuckle pivots 60.6 / 75.602 mm, `joint_tcp` 173.5 mm.
- Materials of `visual_glb/base.glb`, one primitive each: housing white; flange, centring boss and locating pin in the xArm / UF850 flange material (anodised aluminium: base colour 0.953, metallic 0.8, roughness 0.64); the +Y connector cover in the fingers' black (the contact-type cover for XX1305 and UF850 arm ends). With `visual="stl"` the base is one `White` mesh.
- Base below the flange face: the locating boss and the Ø6 locating pin at (0, −25) mm, r = 2.98 mm in both visual meshes. Only two of the 10 base hulls reach there: `base_link_col_00` (boss) and `base_link_col_01` (pin). The hulls are byte-identical to ufactory_mujoco's.
- Joints: `drive_joint` (revolute, 0.0572–0.8651 rad) drives the gripper; the other five finger and knuckle joints mimic it (multiplier 1).
- The `world` link raises the base 0.12 m for standalone preview. Composition drops it and attaches `xarm_gripper_base_link` to the flange with the fixed joint `gripper_fix`.

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2")                          # per-link GLBs, movable
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", gripper_visual="static")  # single static GLB
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", visual="stl")             # STL visuals
```

Composed URDFs are written to `.cache/ufactory/urdf/composed/`. Runtime configurations reference this assembly as `urdf: "assembly:gripper_g2"`.

```bash
python examples/visualization/view_gripper_g2.py    # standalone
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo
```

## Source / License

- Meshes derive from the upstream xArm ROS / xarm_ros2 family (see [NOTICE](../../../NOTICE)).
- This repository maintains the part URDF, the flange-frame meshes and the Draco-compressed GLBs.
