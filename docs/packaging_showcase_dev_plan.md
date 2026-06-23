# Packaging Showcase — Development Plan

Status: yellow table + long-edge layout applied (2025-06); startup snap fixed via deferred viewer; red block applied.

## Current layout

| Constant | Value | Meaning |
|----------|-------|---------|
| `TABLE_TOP_SIZE` | `(1.2, 0.8, 0.04)` | X=1.2 m (long), Y=0.8 m (short) |
| `ROBOT_XY` | `(0.22, -0.38)` | Robot base on **long** edge (negative Y) |
| `HOME_XY` | `(0.0, 0.30)` | Home offset from base (along +Y into table) |
| `OBJ_SPAWN_XY` | `(0.40, 0.05)` | Red block spawn |
| `BOX_CENTER_XY` | `(0.55, 0.26)` | Cardboard box center |
| Camera | `packaging_camera()` | Shared by viewer + `capture_showcase_keyframes.py` |

## Completed

### Initial pose bug (fixed)

**Root cause (first fix, incomplete):** After `scene.build()`, the arm stayed at URDF default qpos (all zeros). `main()` ran settle steps before `init_showcase_robot()`, so the viewer showed a collapsed / wrong pose on first startup.

**Root cause (startup snap, fixed 2025-06):** `show_viewer=True` during `scene.build()` rendered URDF zero qpos before IK home was applied. At q=0 the finger *center* is only ~27 mm above the table, but finger pads extend ~61 mm below center — visually penetrating the tabletop. IK home raises the TCP to ~1.05 m, producing a visible snap.

**Fix:** Build headless (`show_viewer=False`), stiffen mimic constraints, run `init_showcase_robot()` + `hold_robot_home()`, then `start_deferred_viewer()` so the first visible frame is already at home. All six gripper mimic joints are set explicitly to `GRIPPER_OPEN` during init.

### Red block (done)

`examples/_packaging_scene.py` — block uses `color_rough(BLOCK_COLOR)`.

### Yellow table (done)

`add_table()` uses `color_rough(TABLE_TOP_COLOR)` for the top and darker `TABLE_LEG_COLOR` on legs (no wood texture).

### Long-side layout (done)

Robot mounted on the 1.2 m long edge (negative Y); workspace (block, box) on table interior; `HOME_XY` reaches along +Y; `packaging_camera()` centralizes viewer and headless capture cameras.

**Verify:** Run full pick-place cycle; physics grasp/place may need further tuning (target place error &lt; 60 mm).

### Gripper open/close 5× speed (done)

Default showcase uses `GRIPPER_SPEED_FACTOR=5.0`: linear `grasp_close` / `grasp_open` ramps and higher drive_joint PD (`kp=10`, `kv=25`, `force=±5 N`). Arm motion timing unchanged.

### Object-aware partial close (done)

Grasp/squeeze/carry use `grasp_gripper_drive()` with `GRIPPER_GAP_CALIBRATION_OFFSET_M` (block width − 5.3 mm → drive ≈ 0.60) for flush visual contact without finger penetration.

### Pick approach axis separation (done)

Grasp approach no longer uses `move_to` diagonal XYZ from home to pre-grasp. Sequence is:

1. **Phase 1** `move_xy(obj_xy)` — horizontal transit at current Z (home height)
2. **Phase 2** `move_z(grasp_z, obj_xy)` — vertical descent only

Removed separate `pre_grasp_z` hover segment (was grasp+100 mm). Place/transit/return still use `move_to` diagonal interpolation.

### Arm Cartesian speed + trapezoid accel (done)

All arm Cartesian moves (`move_xy`, `move_z`, `move_xy_at_base_z`) use distance-driven trapezoid profiles:

| Constant | Value |
|----------|-------|
| `SHOWCASE_CARTESIAN_SPEED_MMS` | 100 mm/s cruise |
| `SHOWCASE_CARTESIAN_ACCEL_MMS2` | 1000 mm/s² accel/decel |

`--speed` scales both v and a. Gripper open/close keeps `GRIPPER_SPEED_FACTOR` (independent timing). Hold/settle steps unchanged.

---

## Keyframe debug workflow

Use to compare startup poses before/after layout or init changes.

### Headless pose capture (recommended)

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python scripts/capture_showcase_keyframes.py
# optional: --out-dir debug/showcase_keyframes --settle-steps 40 --png
```

**Output:** `debug/showcase_keyframes/keyframes.json` (always). PNG frames when `--png` is passed (requires BatchRenderer; may fail on tight GPU memory).

| Frame | When |
|-------|------|
| `01_post_build` | Right after `scene.build()` — URDF zero qpos |
| `02_post_stiffen_mimic` | After mimic equality stiffening, still zero qpos |
| `03_post_init_home` | After `init_showcase_robot()` |
| `04_after_settle` | After home-hold settle steps |
| `05_cycle_phase0_done` | After Phase 0 inside cycle |

Metadata: `keyframes.json` (link6 position, finger Z above table mm, arm qpos deg, gripper q).

**Expected:** `01_post_build` finger_above_table_mm ≈ +30 (pads visually in table); `03_post_init_home` ≈ +190 (home pose).

```bash
python examples/xarm6/xarm6_g2_showcase.py --capture-keyframes --no-loop
```

Runs init + settle + Phase 0 in the viewer (deferred open — first frame is home).

### What to look for

- **01 vs 03:** Large finger Z jump confirms zero-qpos vs home (old startup snap).
- **03 vs 04:** Should be stable (small PD tracking only).
- **04 vs 05:** Phase 0 should not snap to a different pose.
- **Interactive:** No gripper-through-table flash on window open.

---

## Suggested follow-ups

1. Regenerate marketing / checklist screenshots with new layout  
2. Optional: PNG capture via `--png` when GPU memory allows BatchRenderer

## Related files

- `examples/xarm6/xarm6_g2_showcase.py` — main demo, init fix, `--capture-keyframes`
- `examples/_packaging_scene.py` — layout, table, box, block
- `scripts/capture_showcase_keyframes.py` — headless keyframes
- `scripts/generate_showcase_textures.py` — texture assets
