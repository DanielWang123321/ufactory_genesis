# Packaging Showcase — Development Plan

Status: yellow table + long-edge layout applied (2025-06); initial pose bug fixed; red block applied.

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

**Root cause:** After `scene.build()`, the arm stayed at URDF default qpos (all zeros). `main()` ran 40 `scene.step()` settle steps **before** `run_pick_place_cycle()` called `_init_home_qpos()`, so the viewer showed a collapsed / wrong pose on first startup.

**Fix:** `init_showcase_robot()` runs immediately after `build_packaging_scene()`; settle steps use `hold_robot_home()` so the arm stays at IK home while the block settles.

### Red block (done)

`examples/_packaging_scene.py` — block uses `color_rough(BLOCK_COLOR)`.

### Yellow table (done)

`add_table()` uses `color_rough(TABLE_TOP_COLOR)` for the top and darker `TABLE_LEG_COLOR` on legs (no wood texture).

### Long-side layout (done)

Robot mounted on the 1.2 m long edge (negative Y); workspace (block, box) on table interior; `HOME_XY` reaches along +Y; `packaging_camera()` centralizes viewer and headless capture cameras.

**Verify:** Run full pick-place cycle; physics grasp/place may need further tuning (target place error &lt; 60 mm).

---

## Keyframe debug workflow

Use to compare startup poses before/after layout or init changes.

### Headless PNG capture (recommended)

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
conda run -n py313 python scripts/capture_showcase_keyframes.py
# optional: --out-dir debug/showcase_keyframes --settle-steps 40
```

**Output:** `debug/showcase_keyframes/`

| Frame | When |
|-------|------|
| `01_post_build` | Right after `scene.build()` — URDF zero qpos |
| `02_post_settle_no_home` | After idle steps **without** home init (documents old bug) |
| `03_post_init_home` | After `init_showcase_robot()` |
| `04_after_phase0_hold` | After 40-step home hold |
| `05_cycle_phase0_done` | After Phase 0 inside cycle |

Metadata: `keyframes.json` (link6 position, arm qpos deg, gripper q).

### Interactive viewer

```bash
python examples/xarm6/xarm6_g2_showcase.py --capture-keyframes --no-loop
```

Runs init + settle + Phase 0 in the viewer; for PNGs use the headless script above.

### What to look for

- **02 vs 03:** Large joint change confirms delayed-init bug.
- **03 vs 04:** Should be stable (small PD tracking only).
- **04 vs 05:** Phase 0 should not snap to a different pose if fix is correct.

---

## Suggested follow-ups

1. Regenerate marketing / checklist screenshots with new layout  
2. Optional: remove frame `02` from capture script once bug is well documented  

## Related files

- `examples/xarm6/xarm6_g2_showcase.py` — main demo, init fix, `--capture-keyframes`
- `examples/_packaging_scene.py` — layout, table, box, block
- `scripts/capture_showcase_keyframes.py` — headless keyframes
- `scripts/generate_showcase_textures.py` — texture assets
