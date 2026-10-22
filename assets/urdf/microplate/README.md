# SBS 96-Well Microplate Assets

Standard ANSI/SLAS 96-well microplate model (`127.76 × 85.48 × 14.35 mm`, `0.055 kg`) used by the Bio Gripper G2 pick-and-place simulation (`ufactory.simulation.bio_g2_microplate`).

## Files

- `SBS_96_well_plate.json`: Parametric labware geometry definition from upstream [ramonaoptics/labware](https://github.com/ramonaoptics/labware/tree/b2de87e717584b2e27f436ee03c17b827085e87b) (`b2de87e`).
- `sbs_96_well.glb`: Visual GLB generated from `SBS_96_well_plate.json`.
- `sbs_96_well_convex.stl`: Watertight convex-hull collision mesh (`127.76 × 85.48 × 14.35 mm`).
- `sbs_96_well.urdf`: Standalone Genesis URDF combining the visual GLB, convex collision STL, and calibrated inertia (`0.055 kg`).
- `LICENSE.txt` / `manifest.json`: Upstream BSD-3-Clause license and source checksums.

