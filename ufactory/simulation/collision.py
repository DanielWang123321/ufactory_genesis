"""Geometry-pair collision filtering for URDF entities, supported since Genesis 1.4.2."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ufactory.simulation.compat import GenesisCompatibilityError


def apply_collision_geometry_filters(robot: Any, excluded_geometry_pairs: Iterable[tuple[str, str]]) -> None:
    """Exclude requested URDF box pairs after adding entities, before ``scene.build()``.

    IDs follow Pinocchio's ``<link>_<collision index>`` convention. Box geoms
    have a one-to-one mapping to URDF collision elements; decomposed meshes
    do not, so this entry deliberately rejects them. Reserved contact bits
    are unique within the scene, retaining contacts between different robots.
    """

    pairs = tuple(tuple(pair) for pair in excluded_geometry_pairs)
    if not pairs:
        return
    if robot.scene.is_built:
        raise ValueError("collision geometry filters must be applied before scene.build()")
    if getattr(robot, "_ufactory_collision_geometry_filters", None) == pairs:
        return

    geometries = {f"{link.name}_{index}": geom for link in robot.links for index, geom in enumerate(link.geoms)}
    selected = {name for pair in pairs for name in pair}
    missing = selected - geometries.keys()
    if missing:
        raise ValueError(f"excluded collision geometries missing from Genesis entity: {sorted(missing)}")
    for name in selected:
        geom = geometries[name]
        if geom.type.name != "BOX":
            raise ValueError(f"excluded collision geometry must be a URDF box: {name}")
        desc = getattr(geom, "desc", None)
        if desc is None or not hasattr(desc, "contype") or not hasattr(desc, "conaffinity"):
            raise GenesisCompatibilityError("Genesis geometry descriptions must expose contype and conaffinity")
        if desc.contype != 1 or desc.conaffinity & 0xFFFF != 1:
            raise ValueError(f"excluded URDF geometry already has custom collision masks: {name}")

    # Primitive defaults occupy bits 0-15. Replace each selected geom's type
    # bit 0 with an unused bit. Propagate that replacement to existing affinity
    # masks, preserving their original compatibility, then exclude neighbours.
    used_types = 0xFFFF
    for entity in robot.scene.rigid_solver.entities:
        for geom in entity.geoms:
            used_types |= int(geom.contype)
    available = [1 << bit for bit in range(16, 32) if not used_types & (1 << bit)]
    if len(selected) > len(available):
        raise ValueError("not enough unused contact type bits for collision geometry filters")
    bits = dict(zip(sorted(selected), available, strict=False))
    new_bits = sum(bits.values())
    masks = {name: int(geometries[name].conaffinity) | new_bits for name in selected}
    for first, second in pairs:
        masks[first] &= ~bits[second]
        masks[second] &= ~bits[first]
    selected_ids = {id(geometries[name]) for name in selected}
    for entity in robot.scene.rigid_solver.entities:
        for geom in entity.geoms:
            if id(geom) not in selected_ids and geom.conaffinity & 1:
                geom.desc.conaffinity = int(geom.conaffinity) | new_bits
    for name in selected:
        geometries[name].desc.contype = bits[name]
        geometries[name].desc.conaffinity = masks[name]
    robot._ufactory_collision_geometry_filters = pairs


def configure_lite6_collision_filters(robot: Any) -> None:
    """Apply the bundled Lite6 policy to visualizers that have no runtime YAML."""

    if not any(link.name == "uflite_finger1" for link in robot.links):
        return
    from ufactory.config import load_runtime_config

    apply_collision_geometry_filters(robot, load_runtime_config("lite6").safety.excluded_geometry_pairs)
