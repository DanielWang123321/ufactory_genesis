from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import pytest
import numpy as np

from ufactory.config import load_runtime_config
from ufactory.simulation.collision import apply_collision_geometry_filters


class Geom:
    def __init__(self, contype=1, conaffinity=1):
        self.type = SimpleNamespace(name="BOX")
        self.desc = SimpleNamespace(contype=np.int32(contype), conaffinity=np.int32(conaffinity))

    @property
    def contype(self):
        return self.desc.contype

    @property
    def conaffinity(self):
        return self.desc.conaffinity


def robot(scene):
    links = [SimpleNamespace(name=f"uflite_finger{i}", geoms=[Geom(), Geom()]) for i in (1, 2)]
    entity = SimpleNamespace(scene=scene, links=links, geoms=[geom for link in links for geom in link.geoms])
    scene.rigid_solver.entities.append(entity)
    return entity


def compatible(first, second):
    return bool((first.contype & second.conaffinity) | (second.contype & first.conaffinity))


def test_only_roots_of_the_same_robot_are_filtered():
    scene = SimpleNamespace(is_built=False, rigid_solver=SimpleNamespace(entities=[]))
    first, second = robot(scene), robot(scene)
    objects = SimpleNamespace(geoms=[Geom(0xFFFF, 0xFFFF), Geom(), Geom(2, 0), Geom(0, 1), Geom(2, 1)])
    scene.rigid_solver.entities.append(objects)
    all_geoms = first.geoms + second.geoms + objects.geoms
    before = {frozenset((a, b)): compatible(a, b) for a, b in combinations(all_geoms, 2)}
    excluded = load_runtime_config("lite6").safety.excluded_geometry_pairs
    for entity in (first, second):
        apply_collision_geometry_filters(entity, excluded)
        apply_collision_geometry_filters(entity, excluded)  # idempotent before build
    ignored = {frozenset((entity.geoms[0], entity.geoms[2])) for entity in (first, second)}
    for a, b in combinations(all_geoms, 2):
        pair = frozenset((a, b))
        assert compatible(a, b) is (before[pair] and pair not in ignored)


@pytest.mark.parametrize("failure", ["missing", "mesh", "custom-mask", "built"])
def test_invalid_filter_fails_before_changing_any_mask(failure):
    scene = SimpleNamespace(is_built=False, rigid_solver=SimpleNamespace(entities=[]))
    entity = robot(scene)
    excluded = load_runtime_config("lite6").safety.excluded_geometry_pairs
    if failure == "missing":
        excluded = (("uflite_finger1_0", "unknown_0"),)
    elif failure == "mesh":
        entity.geoms[0].type.name = "MESH"
    elif failure == "custom-mask":
        entity.geoms[2].desc.contype = 2
    else:
        scene.is_built = True
    before = [(geom.contype, geom.conaffinity) for geom in entity.geoms]
    with pytest.raises(ValueError):
        apply_collision_geometry_filters(entity, excluded)
    assert [(geom.contype, geom.conaffinity) for geom in entity.geoms] == before
