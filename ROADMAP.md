# Roadmap

## 0.2.x (ended — personal GitHub preview)

- Preview surface was hosted at `DanielWang123321/ufactory_genesis` and moved to the organization repository in v0.3.0.
- Public APIs could break between minor versions; pin a git tag for secondary work on that line.
- No compatibility shims for historical renames (for example `grasp_place` → `pick_place`).
- Public CI covers the **fast** CPU check only. Simulation and hardware evidence stay on maintainer machines; optional sanitized summaries may attach to GitHub Releases.

## 0.3.x (current — xArm-Developer/ufactory_genesis)

The official repository is [xArm-Developer/ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis). This project is still 0.x. Entry points may be renamed or removed before 1.0.0. Genesis World is validated at 1.4.2 only; a newer release must pass the hook checks or it fails closed. This repository does not promise that today's interfaces keep working on later Genesis versions.

### Official repository migration

1. Done: target organization is `xArm-Developer`, repository name `ufactory_genesis`.
2. Done: repository ownership was transferred. The previous personal URL redirects to the organization repository.
3. Update `pyproject.toml` URLs, README badges, Citation, and the CONTRIBUTING clone URL. LICENSE / authors stay with the current maintainer until a company process says otherwise.
4. Align any PyPI project homepage with the official repository.
5. Stop feature development on the personal 0.2 line after cutover.

### Current entry points

README lists the commands, example directories, and `ufactory` root exports that work in v0.3.0. That list is not a compatibility promise. `ufactory.training`, `ufactory.quality`, and modules left off the list can change without a deprecation window.

### User promotion checklist

- [ ] Official bilingual README Quick Start runs viewer + one sim example on a clean machine in about 10 minutes.
- [ ] Clone size remains acceptable (Draco-compressed visual GLBs; `assets/` on the order of tens of MB).
- [ ] Public CI green on the official repo; Releases attach sanitized sim/hardware evidence summaries.
- [ ] Support channel documented (Issues and/or internal docs).
- [ ] Reference baseline (Python / Genesis / Torch) and maintainer-verified robot list published.

## Out of scope for public CI

Full GPU matrices and real-robot acceptance do not run on GitHub Actions. They remain maintainer checks (`project-check sim` / `sdk-sim` / `hardware`).
