# Pick-place

[中文](README_zh.md) · [Examples guide](../README.md)

Multi-robot pick-place in Genesis physics simulation or offline trajectory and collision preflight. Install the package and run commands from the repository root as described in [Prerequisites](../README.md#prerequisites).

`examples/pick_place/run.py` delegates to the stable `ufactory-pick-place` console command. Both forms accept the same flags.

## Flags

| Flag | Values / notes |
|------|----------------|
| `--robot` | Required: `xarm5` / `xarm6` / `xarm7` / `uf850` / `lite6` |
| `--gripper` | Optional: `default` / `g2` / `bio_gripper_g2` / `lite6` (`bio_gripper_g2` supports microplate task on `xarm5`, `xarm6`, `xarm7`, `uf850`, `sim` mode only) |
| `--mode` | Required for simulation: `sim` (Genesis physics), `dry-run` (offline trajectory and collision preflight) |
| `--executor` | `servo_j` or `servo_cartesian`; Bio G2 defaults to `servo_cartesian`, required for other grippers |
| `--backend` | Optional: `cpu` / `gpu`; overrides `simulation.backend` (use `cpu` without a Genesis-supported GPU) |
| `--visual` | Optional: open the Genesis viewer in `sim` mode |
| `--config` | Optional strict partial overlay YAML |
| `--print-config` | Print resolved runtime YAML and exit |
| `--report` | Optional report output path |

## Commands

```bash
# xArm6 + Gripper G2: print resolved pick-place runtime YAML and exit
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j --print-config

# xArm6 + Gripper G2: offline trajectory and collision preflight (servo_j)
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper: pick-place physics simulation with viewer (servo_cartesian)
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual

# xArm6 + Bio Gripper G2: microplate pick-place physics simulation with viewer (servo_cartesian by default)
python examples/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sim --visual

# Lite6 + Lite6 Gripper: pick-place physics simulation on CPU backend (without a supported GPU)
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --backend cpu

# xArm6 + Gripper G2: offline preflight with partial configuration overlay
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j \
  --config examples/pick_place/runtime.example.yaml
```

## Configuration overlay

`runtime.example.yaml` is a strict partial overlay. Omitted robot, geometry, motion, safety, and simulation values continue to resolve from `assets/configs/runtime`.

## Real robot

For physical robot pick-place, see [Hardware & Real Robot](../README.md#hardware--real-robot).
