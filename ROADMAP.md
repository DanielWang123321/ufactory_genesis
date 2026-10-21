# Roadmap

## 0.2.x (ended — personal GitHub preview)

- Preview surface was hosted at `DanielWang123321/ufactory_genesis` and moved to the organization repository in v0.3.0.
- Public APIs could break between minor versions; pin a git tag for secondary work on that line.
- No compatibility shims for historical renames (for example `grasp_place` → `pick_place`).
- Public CI covers the **fast** CPU check only. Simulation and hardware evidence stay on maintainer machines; optional sanitized summaries may attach to GitHub Releases.

## 0.3.x (current — xArm-Developer/ufactory_genesis)

The official repository is [xArm-Developer/ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis). This project is still 0.x. Entry points may be renamed or removed before 1.0.0. Genesis World 1.4.4 is the validated baseline. 1.4.2 is the minimum; 1.4.2 and 1.4.3 still run and emit one warning. A newer release must pass the hook checks or it fails closed. This repository does not promise that today's interfaces keep working on later Genesis versions, or that 1.4.2 contact numbers match 1.4.4.

### Official repository migration

Repository ownership has been transferred to `xArm-Developer/ufactory_genesis`. The previous personal URL redirects to the organization repository, and feature development on the 0.2 line has ended.

### Current entry points

Current commands, example directories and `ufactory` root exports are listed below. Later 0.x releases may rename or remove them. `ufactory.training`, `ufactory.quality`, and modules left off the list can change without a deprecation window.

- Commands: `ufactory-pick-place`, `ufactory-packaging-showcase`, `dynamics-sim-check`, `dynamics-sim-collision-check`, `dynamics-hardware-check`, `dynamics-report-compare`. `project-check` and `observe-hold-current` are maintainer tools.
- Examples: `examples/visualization`, `examples/kinematics`, `examples/pick_place`, `examples/packaging`, `examples/rl/reach`, `examples/rl/pick_place`, including the `runtime.example.yaml` files beside the pick-place and packaging entries. See the [examples guide](examples/README.md) for usage.
- Library exports from `ufactory`: `get_robot_profile`, `robot_urdf`, `robot_visual_glb_urdf`, `robot_assets`, `RepositoryAssetStore`.

### User promotion checklist

- [ ] English and Chinese example guides run a viewer + one simulation example on a clean machine in about 10 minutes.
- [ ] Clone size remains acceptable (Draco-compressed visual GLBs; `assets/` on the order of tens of MB).
- [ ] Public CI green on the official repo; Releases attach sanitized sim/hardware evidence summaries.
- [ ] Support channel documented (Issues and/or internal docs).
- [ ] Reference baseline (Python / Genesis / Torch) and maintainer-verified robot list published.

## Out of scope for public CI

Full GPU matrices and real-robot acceptance do not run on GitHub Actions. They remain maintainer checks (`project-check sim` / `sdk-sim` / `hardware`).
