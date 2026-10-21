# 2. Pick & Place: xArm6 + Gripper G2

Move a 30 mm cube from `(300, 0) mm` to `(300, 300) mm` in simulation. Train an end-to-end policy from scratch with pure PPO. Every episode starts at EE `(300, 0, 300) mm` with a 50 mm gripper gap.

[中文说明](README_cn.md) · [RL examples](../README.md) · [Previous: Reach](../reach/README.md)

Environment **v3** uses 70 observations and four actions `Δx, Δy, Δz, Δgap`, a downward gripper orientation, a 20 ms control period, 32 physics substeps, and an 8 s episode limit. Cartesian increments are at most 5 mm per step; gap increments are at most 3 mm. Commands accumulate; zero actions hold the commands. All channels remain available throughout the episode, subject to workspace bounds, URDF joint limits, IK checks, and a 25 mm setpoint leash. Gripper dynamics use the calibrated [G2Controller](../../../ufactory/simulation/g2_controller.py).

Observations include joint positions and velocities, EE and object state, relative target position, actual finger-pad centers relative to the cube, commanded and measured gap, EE command error, finger contacts, table support, robot-table loads, previous actions, and task history. There is no phase observation. Older policies have incompatible observations and control semantics and must be retrained.

Discounted potential differences and one-time grasp, lift, and low-release bonuses guide approach, grasping, lifting, transport, release, and hold. A pre-grasp gap potential varies continuously with actual pad-to-cube distance before placement, encouraging an open approach followed by closing and gradual preload unwinding during descent. Placement uses the cube's lowest point under its actual orientation, with a 2 mm clearance target before release and a 0–2 mm resting band once low release is confirmed. Once low release at the target is confirmed (`place_rewarded`), approach, pre-grasp, and grasp potentials smoothly hand off to contact-free opening progress and settled hold steps, eliminating re-closing incentive. Held descent faster than 20 mm/s within 20 mm of the table, movement after touchdown, table overload, high release, and robot-table collisions incur penalties. Repeated opening, closing, or lifting cannot collect the one-time bonuses again.

The scene geometry and home pose are defined in `ufactory/training/tasks.py` and `assets/configs/runtime/tasks/pick_place.yaml`. Runtime YAML overrides support `task.parameters.control_dt_s`, `workspace_lower_m`, `workspace_upper_m`, `object_mass_kg`, gripper damping/friction, solver settings, and G2 speed/force percentages. The 8 s episode, model, cube size and layout are fixed. Unsupported overrides and unused recipe fields raise an error. Existing v3 checkpoints remain loadable with hash, action-contract and safety checks.

Training metrics, checkpoint manifests and caches use shared helpers in `ufactory/training/runner.py` and `runtime.py`.

## Strict success criteria

- Pre-lift drag ≤10 mm and tilt ≤15°. Confirm bilateral contact for at least three steps, with 0.2 N contact entry and 0.05 N exit thresholds.
- Lift the held cube's lowest point to ≥65 mm (80 mm center height for an upright cube). Finish within 20 mm of the target with tilt ≤15°.
- Release at **0–3 mm** actual clearance, with measured gap ≥35 mm. Both fingers must stay below 0.05 N for three steps to confirm detachment. Dropping from height and landing at the target fails.
- Additional EE descent from touchdown until both fingers detach must be **≤1 mm**. Include the sample before touchdown; retain tracking during single-finger contact and brief contact gaps. Recontact starts another touchdown/release interval while retaining the episode’s worst pressing and highest release.
- After release, require table support and object speed <30 mm/s for at least **0.2 s**. Robot-table collision fails.

Contact loads use entity filtering and `valid_mask`, separating object support from robot-table loads; see the [Genesis contact interface](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_solver/collision_detection.html). Reports include `release_height_mm`, `release_confirmed`, `press_mm`, `max_table_force_n`, `max_place_table_force_n`, `max_held_table_force_n`, and `max_robot_table_force_n`. Read release height together with its confirmation flag.

## Baselines and training

Run from the project root in `py313`. Requesting GPU raises an error if Genesis falls back to CPU.

```bash
python -m examples.rl.pick_place.evaluate --zero-policy -B 16 --episodes 32 --quiet-episodes
python -m examples.rl.pick_place.evaluate --expert -B 16 --episodes 32 \
  --min-success-rate 1.0 --max-press-mm 1.0 --quiet-episodes

# One seed: full-task PPO from scratch, with every reset at home
python -m examples.rl.pick_place.train -e pick_place_e2e_s1 --seed 1 \
  -B 2048 --max-iterations 2000 --eval-interval 50 --eval-episodes 64

# Train three seeds, evaluate 64 independent episodes each, record B=1 GLB replays
python -m examples.rl.pick_place.run_seeds
```

Defaults: seeds 1, 2, 3; learning rate `3e-4`; `gamma=0.995`; initial action standard deviation `0.5`; entropy coefficient `0.005`. Keep the existing `[256, 256]` networks and 32 rollout steps per iteration. Stop after two consecutive passing periodic evaluations, up to 2000 iterations. Disable this with `--early-stop-evaluations 0`; `training_summary.json` records the actual iteration count. The scripted expert is an independent evaluation policy and never supplies PPO training data. Existing experiment directories cause an error to preserve their artifacts.

Outputs go to `outputs/rl/pick_place/pick_place_e2e_s1/` (or `s2/s3`). Rank checkpoints by strict success rate, valid low-touchdown rate, maximum pressing, then mean landing error. Save configuration, source hashes, checkpoint manifests, training logs, and JSON/CSV evaluations. Each seed must achieve **≥98%** success and **≤1 mm** maximum pressing. Combined results and process logs go to `outputs/rl/pick_place/e2e_validation/`.

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt \
  -B 32 --episodes 64 --seed 18001 --min-success-rate 0.98 --max-press-mm 1.0 \
  --out outputs/rl/pick_place/pick_place_e2e_s1/test.json

python -m examples.rl.pick_place.visualize -e pick_place_e2e_s1 -B 1
python -m examples.rl.pick_place.visualize -e pick_place_e2e_s1 -B 1 --episodes 1 \
  --headless --record outputs/rl/pick_place/pick_place_e2e_s1/replay.mp4
```

Video defaults to GLB PBR at 50 FPS, with accompanying JSON/CSV metrics. Three-seed GPU training (`pick_place_e2e_s1..s3`), 64-episode strict evaluation, and `B=1` GLB replay verification pass under the unified `impratio = 100.0, noslip_iterations = 0` profile (100% success, maximum pressing 0.084 mm, mean landing error 4.44 mm).
