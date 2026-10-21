# scripts/

用户可用的辅助脚本，以及少量维护者发版工具。资产 vendor / relocalize / combo 生成流水线属于内部维护材料，不随公开面发布。

## 用户脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `gen_kinematics_params.py` | `ufactory.kinematics` | 从机器人控制柜提取逐台运动学 YAML（严格 v0.2.5 校准 schema） |

真机 hold 电流/力矩时间序列观察由 `observe-hold-current` 控制台入口（`ufactory.hardware.observe:cli_observe_hold_current`）提供，安装后可直接调用。

## 维护者脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `export_release_evidence_summary.py` | `ufactory.quality.evidence_summary` | 从本地 `project-check` JSON 报告生成脱敏 Release 附件摘要 |
