# Gripper G2 calibrated simulation

[中文](README_cn.md)

The standalone gripper, xArm5/6/7 and UF850 simulations use `g2_calibrated_v1`.
The public implementation is in `ufactory/simulation/`; these examples only start it.
Validated with local Genesis World 1.4.4. Hardware and SDK simulation accept
opening in millimetres; motion mirrors use the same joint geometry.

Run from an editable source checkout. Pick-place and packaging preflight also need
Pinocchio/Coal (`.[real]`).

```bash
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
export GS_CACHE_FILE_PATH="$PWD/.cache/genesis"
export QD_OFFLINE_CACHE_FILE_PATH="$PWD/.cache/quadrants"

# Hold a 30 mm free cube, then remove its support.
python examples/gripper_g2/grasp_cube.py --headless --backend gpu \
  --position-mm 0 --speed-percent 50 --force-percent 50 --csv reports/g2/cube.csv

# All 33 original dynamics cases; each case saves its CSV and result.
python examples/gripper_g2/verify.py --backend gpu --suite all --out reports/g2/validation
```

Commands use pad opening in **0–84 mm**, speed in **0–100%**, and force in
**1–100%**. Speed zero pauses feed and preserves holding preload; automatic
pick-place and packaging require positive speed. Force percentage selects the
calibrated motor effort; it is not a force in newtons. Speeds above the chart's
3500 rpm limit use its boundary force curve while retaining commanded motion speed.

The defaults are `kp=40`, `kv=5`, `ki=40`, 0.01 kg·m² gripper joint armature,
50 Hz control, 32 substeps, a 1.25 ms contact time constant, and 5 no-slip
iterations. Without the no-slip pass, pyramidal friction with soft contacts lets a
held cube creep about 2.4 mm/s at 50% force. Pick-and-place RL keeps
`noslip_iterations: 0`, because 5 iterations make the 20480-environment batch
about 7× slower. Only `drive_joint`
is actuated. The controller uses position/velocity feedback with conditional
integral anti-windup; contact forces only confirm workflow transitions. Scalar
`G2Controller` and Torch `G2BatchController` expose `command(position_mm,
speed_percent, force_percent)`, `update`, and `reset`; the batch implementation
supports selected-environment reset and independent torque limits.

The steady-force curve is built into `ufactory/simulation/g2_model.py`. The
reference is raw SDK position 230, inferred as 24.155872 mm; it is not a
documented specimen thickness. All 11 standard G2 URDFs directly encode the same
six-joint limits: open `0.057210812728026743 rad`, closed
`0.86509464599328212 rad`, maximum velocity `2.608067454724 rad/s`.
The nominal stroke is **0–84 mm**. Individual hardware deviations of 1–2 mm
belong to device calibration; this change adds no device calibration.

`ufactory/grippers/g2.py` defines the linkage geometry and scalar conversions;
the Torch paths use the same formula and parameters. Opening/angle conversion
functions keep their signatures and now return or accept actual joint radians.
Scalar conversions work without Torch, Genesis or trimesh. Simulations load the
project URDF directly. `derive_urdf()` and `derive_g2_urdf` have been removed.

## Pick-place and packaging

```bash
python examples/pick_place/run.py --robot xarm6 --mode sim --executor servo_cartesian \
  --position-mm 0 --speed-percent 50 --force-percent 50 \
  --report reports/g2/pick.preflight.json --csv reports/g2/pick.csv --sim-report reports/g2/pick.json
python examples/packaging/run.py --robot xarm6 --mode sim --executor servo_j --cycles 2 \
  --speed-percent 50 --force-percent 50 \
  --report reports/g2/pack.preflight.json --csv reports/g2/pack.csv --sim-report reports/g2/pack.json
```

Use `xarm5`, `xarm7` or `uf850` with either executor. YAML commands are
`simulation.g2.position_mm`, `speed_percent` and `force_percent`; CLI values override
YAML. Packaging's `--speed` scales the arm path and observation windows, while
`--speed-percent` controls the gripper. Multiple cycles save `.cycle-N` files.

A grasp requires both normal forces above 0.2 N and their average within ±5% of
the opening-dependent target for two consecutive frames. Opening requires
84±1 mm and both normal forces at most 0.2 N for two frames. The next control
cycle starts the lift/retreat, or packaging's existing drop observation window.
There is no `grip-settle`. Both deadlines are `max(2 s, reference travel / speed
+ 1 s)`. A timeout stops subsequent arm motion and saves failure JSON and raw CSV,
including default files under `reports/g2/failures/` if paths were omitted.

Scenes and executable programs are built before preflight. Conditional grasp
preflight checks the calibrated linkage at the rigid object's width; arm motion
can proceed only after actual bilateral force confirmation. Collision margins
and transport acceptance thresholds are unchanged.

`--report` stores preflight, `--sim-report` stores execution status, actual phase
durations, force/opening confirmation samples, next-action delay and separate
timing/transport results. `--csv` stores every control sample. The CLI returns
failure on timeout or failed timing/transport acceptance, including the 5 mm slip
and (for pick-and-place) 10 mm placement limits.
Deterministic G2 tasks enable `simulation.noslip_iterations = 5`,
with a dedicated CPU regression requiring at most 1 mm of slip.

## Demos and RL

```bash
python examples/visualization/view_gripper_g2.py --headless --speed-percent 50 --force-percent 50
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo \
  --headless --position-mm 0 --speed-percent 50 --force-percent 50

python -m examples.rl.pick_place.train --num-envs 32 --max-iterations 5 --log-dir reports/g2/rl-short
python -m examples.rl.pick_place.train --num-envs 32 --max-iterations 1 \
  --resume reports/g2/rl-short/model_4.pt --log-dir reports/g2/rl-resume
python -m examples.rl.pick_place.evaluate --allow-physics-transfer --headless -B 1 \
  --episodes 16 --summary-json reports/g2/legacy-evaluation.json
```

The existing RL task remains xArm6 with the same policy action, observation and
reward interfaces. Policy actions keep issuing position commands; YAML supplies
speed and force. Training artifacts record effective parameters and current asset/configuration
hashes. These hashes are recomputed for the fixed URDFs; historical checkpoints
and manifests are preserved and compatibility checks remain strict. Bundled weights retain the historical `g2_stable_v1_3_3` profile and files.
Evaluating them requires `--allow-physics-transfer`, which still validates original
checkpoint/config integrity. Cross-version full training resume is rejected.
Historical success rates do not describe the new physics.

The G2 tests occupy seven files: calibration, constraints, batch control,
confirmation, replay, CPU transport and GPU acceptance. The GPU suite has
22 cases: nine pick-place, four fault injections, five packaging (seven cycles),
two RL batches (1/4 environments), and two dynamic demos (standalone/xArm6).
Run the GPU suite serially, followed by the CPU transport regression:

```bash
python -m pytest tests/simulation/test_g2_gpu.py -q
python -m pytest tests/simulation/test_g2_transport_cpu.py -q
project-check fast
```

`project-check fast` and `project-check deep` retain their existing markers.
The matrix saves raw records under `reports/g2/`; timing and transport quality are
reported separately. The 33 independent cases retain the original central-speed
10%, endpoint 1 mm, stable-force `max(2 N, 5%)`, and torque-limit thresholds.
