# `ufactory.dynamics` — xArm 动力学验证子包

v0.2.0 中，动力学验证公开入口统一为 `ufactory.dynamics` 子包；旧顶层兼容模块已删除。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `probe` | Genesis 场景构建；PD 持位采样（Genesis 仿真里，为维持目标关节角，PD 控制器算出的关节力矩；含 armature + frictionloss 回读）；EE z 表；pre-snap-home；plane morph；6 个去硬编码物理检查（自由落体 / 重力补偿 / PD 阶跃 / 能量 / 质量矩阵 / 物理参数）。 |
| `reference` | Pinocchio 独立 CRBA + 重力 oracle 后端；armature 对齐质量辅助函数。 |
| `analysis` | L2a/L2b/L3a/L3b 静态分层机（重构后 L2b oracle = `pd_hold_tau vs pin_G(q_actual)`）；URDF 动力学验证；力矩对比 / 分类 / `build_dynamics_sample`。 |
| `report` | `DynamicsRunConfig` / `GenesisDynamicsSample` / `DynamicsSample` / `ValidationStatus`；JSONL/CSV 报告写入+读取+对比；仿真/运行时常量。 |
| `poses` | 运行时验证姿态访问；默认姿态由 `assets/configs/dynamics_validation_pose.yaml` 经 `poses_config` 展开。 |
| `cli` | 4 个 CLI 入口 + sim collision 串联 + 打印辅助。 |
| `__init__` | re-export 公开 API；`__all__` 完整。 |

## 报告 schema v4

`run_config.version = "4"`。CSV/JSONL 的主判定字段使用显式单位命名：

- `torque_l2_err_nm = sqrt(sum((genesis_tau_nm - sdk_tau_mean_nm)^2))`。
- 每关节字段使用 `genesis_tau_J{i}_nm`、`sdk_tau_mean_J{i}_nm`、`abs_err_J{i}_nm`、`rel_err_J{i}`。
- `status_reason`、`worst_joint`、`worst_abs_err_nm` 用于快速定位失败关节。
- L2/L3 静态层仍作为诊断字段保留；默认不影响 Overall，除非启用 `--strict-static`。

默认输出按机械臂身份归档：
`reports/dyn_ver_<SN或robot_key>/<YYYYMMDD_HHMM>/dyn_ver_<SN或robot_key>_<YYYYMMDD_HHMMSS>.*`。
硬件报告会额外生成同名 `_torque.png`，按关节绘制 Genesis 理论力矩与 SDK 采样均值。

`read_report_records` 版本感知：CSV 读 `schema_version` 列，JSONL 读 `run_config.version`；v1–v3 旧报告仍可解析并用于 `compare_report_records`。新写入始终使用 schema v4。

## CLI 入口

```
dynamics-sim-check             ufactory.dynamics.cli:cli_sim_check
dynamics-sim-collision-check   ufactory.dynamics.cli:cli_sim_collision_check
dynamics-hardware-check        ufactory.dynamics.cli:cli_hardware_check
dynamics-report-compare        ufactory.dynamics.cli:cli_report_compare
```

按重构计划，4 个入口名保持不变。

## 用法

```python
from ufactory.dynamics import (
    build_dynamics_sample, build_genesis_scene, build_static_pose_analysis,
    cli_hardware_check, compare_report_records, GenesisDynamicsSample,
    load_reference_backend, parse_strict_static_layers,
    read_report_records, validate_urdf_dynamics, write_csv_report,
    write_jsonl_report, write_torque_plot,
)
```

或直接驱动 dry-run：

```python
from ufactory.dynamics import cli_hardware_check
rc = cli_hardware_check([
    "--robot", "xarm6",
    "--dry-run",
    "--require-reference",
    "--strict-static",
    "--strict-static-layers", "l2b",
    "--poses", "0,1,14",
    "--z-min-mm", "0",
])
```

## 测试

`tests/dynamics/test_dynamics_*.py` 覆盖本子包。贡献者安装与分档运行见仓库根 [CONTRIBUTING.md](../../CONTRIBUTING.md)：

```bash
pip install -e ".[dynamics,dev]"

# PR 默认（秒级，无 GPU / 无子进程冒烟）
pytest -m "not hardware and not gpu and not integration and not display"

# 发版前完整仿真回归
pytest -m "not hardware"
```
