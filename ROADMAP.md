# Roadmap

## 0.2.x (ended — personal GitHub preview)

- Preview surface was hosted at `DanielWang123321/ufactory_genesis` and moved to the organization repository in v0.3.0.
- Public APIs could break between minor versions; pin a git tag for secondary work on that line.
- No compatibility shims for historical renames (for example `grasp_place` → `pick_place`).
- Public CI covers the **fast** CPU check only. Simulation and hardware evidence stay on maintainer machines; optional sanitized summaries may attach to GitHub Releases.

## 0.3.x (current — xArm-Developer/ufactory_genesis)

The official repository is [xArm-Developer/ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis). Public APIs may still change in later 0.3.x versions. The API freeze below has not started.

### Official repository migration

1. Done: target organization is `xArm-Developer`, repository name `ufactory_genesis`.
2. Done: repository ownership was transferred. The previous personal URL redirects to the organization repository.
3. Update `pyproject.toml` URLs, README badges, Citation, and the CONTRIBUTING clone URL. LICENSE / authors stay with the current maintainer until a company process says otherwise.
4. Align any PyPI project homepage with the official repository.
5. Stop feature development on the personal 0.2 line after cutover.

### Public API freeze (first promise)

After 0.3 ships, the default-stable surface is:

- Documented CLI entry points (`ufactory-pick-place`, packaging CLI, dynamics CLIs listed in README).
- Task-oriented `examples/` layouts and their beside-entry overlays.
- Symbols exported from `ufactory` root and the README **API Quick Reference** list.

Still evolving without freeze promises:

- `ufactory.training` and other library APIs not listed as stable in 0.3 docs.
- Internal `ufactory.quality` and undocumented private modules.

**0.2 → 0.3** may include one cleanup Breaking change set with a one-page migration note (no runtime shims required).
**After 0.3**, Breaking changes require at least one minor-version deprecation window except for urgent safety fixes.

### User promotion checklist

- [ ] Official bilingual README Quick Start runs viewer + one sim example on a clean machine in about 10 minutes.
- [ ] Clone size remains acceptable (Draco-compressed visual GLBs; `assets/` on the order of tens of MB).
- [ ] Public CI green on the official repo; Releases attach sanitized sim/hardware evidence summaries.
- [ ] Support channel documented (Issues and/or internal docs).
- [ ] Reference baseline (Python / Genesis / Torch) and maintainer-verified robot list published.

## Out of scope for public CI

Full GPU matrices and real-robot acceptance do not run on GitHub Actions. They remain maintainer checks (`project-check sim` / `sdk-sim` / `hardware`).
