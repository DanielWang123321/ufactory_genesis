# Bio Gripper G2 Assets

柔性仿生夹爪（公共可动模块，可独立使用或搭配 xArm / UF850）。

## 行程（重要）

两指夹持面间距的真实机械范围：**闭合 71 mm → 张开 150 mm**。

- 驱动关节为右指 prismatic dof，左指 `mimic ×-1` 对称跟随。
- **关节零位 = 闭合（71 mm）**，dof 增大到 `0.0395 m` 时张开到 150 mm（单指行程 39.5 mm）。
- 间距与关节值近似线性：`gap ≈ 0.071 + 2 × right_joint_value`（米）。
- 单指运动速度范围：`41.67–333.33 mm/s`（对应电机 `500–4000 rpm`，双指相向开合速度 `83.33–666.67 mm/s`）；单指夹持力 `4–20 N`（由虚功原理折算等效广义驱动力上限 `40 N`）。
- 这些数值集中在 `ufactory/grippers/bio_g2.py`、`ufactory/simulation/bio_g2_model.py` 以及各 URDF 的关节 `limit`（`lower="0" upper="0.0395" effort="40" velocity="0.333333333333333"`）中。

## 模板 URDF

`bio_gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的 standalone 模板文件；v0.1.6 中 mesh 路径已改为本仓库内的 `meshes/visual/*.stl`，可用于基础结构解析。它也是关节 `limit`（行程）的唯一真源。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口。

Bio G2 的 STL collision 文件使用 `meshes/collision/bio_gripper_g2_*_convex_*.stl` 唯一文件名，均由视觉 GLB 生成（见“碰撞体”）。不要把零件 URDF 改回 `link_base.stl` / `left_finger.stl` / `right_finger.stl`：Genesis 经 MuJoCo 回读 URDF collision mesh 时会按 mesh basename 建资产表，机械臂自身也有 `link_base.stl`，短名会让 Bio G2 底座误用机械臂底座 collision。

`bio_gripper_g2_movable_visual.urdf`（组合源）中的 Bio G2 凸包 collision 已将几何偏移烘焙至网格顶点（`<collision><origin xyz="0 0 0"/>`），可动手指 GLB `<visual><origin>` 通过 Y 轴微调偏移与 `71–150 mm` 物理夹持平面严格对齐；整爪总质量标定为 `0.79 kg`（底座含转接板 `0.765827 kg`、左指 `0.012105 kg`、右指 `0.012068 kg`）。

Genesis standalone 检查可使用 `bio_gripper_g2_movable_visual.urdf`。搭配机械臂时由 `ufactory.robots.assembly` 运行时组合。xArm5/6/7 与 UF850 使用同一法兰标准，安装位姿只有一组，见 [`assembly.yaml`](../assembly.yaml) 的 `bio_gripper_g2.mount`。

## 安装与坐标系

- **零件坐标系不在转接板上。** 在 `bio_gripper_g2_base_link` 系中，转接板中心位于 `(-22.636, -0.956, -2.516) mm`，姿态 rpy 为 `(0.031, 0.254, 0.360)°`，原因是早期把 CAD 按质心对齐到了 G1 STL。手指关节、TCP、惯量与碰撞体都定义在这个坐标系中，因此保持不动，由安装位姿补偿。
- **安装位姿。** xyz `(22.642, 0.814, 2.516) mm`，rpy `(0, 0, -0.00628537) rad`。效果：转接板中心在法兰轴上，转接面中心落在 z=0，两颗 M6 螺钉尾端对准法兰 135°/225° 螺孔（Ø50 圆周，残差 < 0.02 mm）。
- **不含倾斜。** GLB 相对零件坐标系还有 0.25° 倾斜，这是早期摆放误差：两个平面贴合不可能倾斜。若在安装位姿里补这个倾斜，TCP 朝下时 xArm5（5 轴）会无解，xArm6 会接近腕部奇异。因此转接面边缘在视觉上残留 ±0.16 mm 的不平。
- **校核。** 视觉指尖距法兰轴 157.9（右）/ 158.5（左）mm，均值 158.2 mm，CAD 实测为 157.5 mm。TCP 在法兰系中为 `(157.6, 0.0, 57.5) mm`，抓取点（`GRASP_OFFSET_IN_BASE_LINK`）为 `(150.6, 0.0, 52.5) mm`。
- **与厂商模型的关系。** 厂商 xarm_ros2 的 G2 模型以转接板为原点，以恒等变换安装在 `link_eef` 上，网格形状与本仓库一致。但厂商模型的手指相对壳体位置与 CAD 相差约 12 mm（指尖 170.5 mm），以 CAD 为准。

## 碰撞体

全部由视觉 GLB 生成，表达在各连杆坐标系中（`<collision><origin>` 为零）。生成脚本属内部维护材料，不随仓库发布。

- **底座**：1 个凸包，取转接面以上的 GLB 顶点，去掉 M6 螺钉尾端。转接板比壳体后壁多伸出约 20 mm，凸包在此处填平，最多超出视觉 13 mm。
- **手指**：每指 5 个凸包，在手指坐标系中按下表切分（u = |y|，即到夹爪中面的距离）：

| 文件 | 部位 | 范围 (mm) |
|---|---|---|
| `*_finger_convex_0` | 夹持垫内层 | x ≥ 31.5，u ≤ 40.3；内侧面压平到 u = 35.5 |
| `*_finger_convex_1` | 夹持垫外层 | x ≥ 31.5，u ≥ 40.1，z ≤ 25.9 |
| `*_finger_convex_2` | 指杆 | −3 ≤ x ≤ 31.5 |
| `*_finger_convex_3` | 关节立板 | −25 ≤ x ≤ −3，u ≥ 39.6 |
| `*_finger_convex_4` | 关节横块 | −25 ≤ x ≤ −3，z ≤ 6.8 |

- `convex_0` 的内侧面就是夹持平面：关节零位时位于 u = 35.5（间距 71 mm），垂直于关节轴，覆盖 x 33…76、z 11…29。它必须是手指的第一个碰撞元素，因为 `collision_gripping_planes` 从这里读取夹持面。视觉夹持面带有 0.36° 的偏航残差，与碰撞夹持面最多相差 0.3 mm。
- 滑块段（x < −25）在全行程内都位于底座凸包内，因此不建碰撞体。
- 碰撞指尖与视觉指尖一致，距法兰轴 157.9 / 158.5 mm。夹持垫是鳍条镂空结构，凸包会把空腔填实；除空腔外，凸包表面最多超出视觉 3.9 mm，出现在关节横块的颈部，该处位于壳体内。

## 目录结构

```
bio_gripper_g2/
├── bio_gripper_g2.urdf                      # standalone 模板 URDF
├── bio_gripper_g2_movable_visual.urdf       # 零件 URDF（组合源；标定动力学与凸包碰撞）
├── meshes/
│   ├── collision/
│   │   ├── bio_gripper_g2_base_link_convex_0.stl
│   │   ├── bio_gripper_g2_left_finger_convex_0.stl … _4.stl
│   │   └── bio_gripper_g2_right_finger_convex_0.stl … _4.stl
│   └── visual/
│       ├── bio_gripper_g2_visual.glb        # 合并静态高模 GLB（全机型共用，原生 Draco）
│       ├── visual_glb/bio_gripper_g2_*.glb  # 可动分体高模 GLB（全机型共用，原生 Draco）
│       ├── bio_gripper_g2_left_finger.stl / bio_gripper_g2_right_finger.stl
│       └── bio_gripper_g2_link_base.stl     # 原始 STL 视觉网格
```

视觉保持 link6 高模面数（静态 ~479k / 可动 ~273k tris），用原生 `draco_encoder`（非 DracoPy.encode）压 **视觉 GLB** 体积（合计目标 **< 1 MiB**）。collision STL 保持原密度，不做抽面。

## 加载方式

同一份可动夹爪定义有两种加载入口：

```python
from ufactory.robots.assembly import compose_robot_urdf
from ufactory.robots.paths import bio_gripper_g2_movable_visual_urdf

# 1) 独立可动夹爪
urdf = bio_gripper_g2_movable_visual_urdf()

# 2) 搭配 xArm5/6/7 或 UF850（运行时组合，输出到 .cache/ufactory/urdf/composed/）
urdf = compose_robot_urdf("xarm6_1305", "bio_gripper_g2")
# 静态高模（不可动）：gripper_visual="static"
```

组合输出同时包含 `<visual>` GLB 与 `<collision>` 中的 `meshes/collision/bio_gripper_g2_*_convex_*.stl`（`<collision><origin xyz="0 0 0"/>`），并通过可动手指 `<visual><origin>` 的 Y 轴微调使视觉与碰撞夹持面在 `71–150 mm` 全行程对齐。

## 控制（公共模块）

几何级连杆控制使用 `ufactory.grippers.bio_g2.BioGripperG2`（自动发现夹爪关节并镜像左指），物理标定斜坡与力控使用 `ufactory.simulation.BioG2GripperController`：

```python
from ufactory.grippers.bio_g2 import BioGripperG2

gripper = BioGripperG2(robot)  # robot 为加载上述任一 URDF 得到的 entity
if gripper.found:
    gripper.setup_pd()
    gripper.set_pose(gripper.OPEN_POS)  # 张开到 150 mm（OPEN_POS）
    gripper.control_pose(gripper.CLOSE_POS)  # 闭合到 71 mm（CLOSE_POS）

# 开合循环：
for step in range(1000):
    gripper.control_pose(gripper.demo_target(step))
    scene.step()
```

预览命令：

```bash
# 独立
python examples/visualization/view_bio_gripper_g2.py
# 搭配机械臂（开合演示）
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --movable --gripper-demo
```

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护零件 URDF、安装位姿、共享 `visual_glb/`（取消按 link5/6/7 重复副本）与 Draco 压缩 GLB。
