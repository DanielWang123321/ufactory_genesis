# Packaging Showcase Simulation

[中文](README_zh.md) · [Examples guide](../README.md)

Multi-robot box packing with trajectory control, in Genesis physics simulation or offline preflight. Install the package and run commands from the repository root as described in [Prerequisites](../README.md#prerequisites).

`examples/packaging/run.py` delegates to `ufactory-packaging-showcase`.

The cube is released 50 mm above the box rim and falls naturally. GPU regressions
allow rebound and require its final collision bounds to remain inside the box
(1 mm contact tolerance), rather than within 25 mm of the target center.
Grasp, transport, release and return-home checks remain in place.

## Flags

| Flag | Values / notes |
|------|----------------|
| `--robot` | Default `xarm6`; all five families supported in sim / dry-run |
| `--mode` | Required for simulation: `sim` (default), `dry-run` (offline preflight) |
| `--executor` | Default `servo_j`; also `servo_cartesian` |
| `--backend` | Optional: `cpu` / `gpu`; same meaning as [pick-place](../pick_place/README.md#flags) |
| `--cycles N` | Exact simulation cycle count (default 1) |
| `--speed` | Simulation playback multiplier (`>1` is faster) |
| `--table-height` | Simulation display height only; base-frame geometry unchanged |
| `--visual` | Optional: hold the final viewer frame |
| `--config` / `--print-config` / `--report` | Same roles as [pick-place](../pick_place/README.md#flags) |

## Commands

```bash
# xArm6 + Gripper G2: 1-cycle packaging physics simulation with viewer (servo_j)
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual

# xArm6 + Gripper G2: 1-cycle packaging physics simulation on CPU backend
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --backend cpu

# Lite6 + Lite6 Gripper: 3-cycle packaging physics simulation regression
python examples/packaging/run.py \
  --robot lite6 --mode sim --executor servo_j --cycles 3

# xArm7 + Gripper G2: packaging offline trajectory and collision preflight
python examples/packaging/run.py \
  --robot xarm7 --mode dry-run --executor servo_j

# xArm6 + Gripper G2: packaging physics simulation with partial configuration overlay
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j \
  --config examples/packaging/runtime.example.yaml
```

## Real robot

`sdk-sim` and `real` modes are covered in [Hardware & Real Robot](../README.md#hardware--real-robot). Real packaging is enabled for xArm6 + Gripper G2 and Lite6 + Lite6 Gripper.
