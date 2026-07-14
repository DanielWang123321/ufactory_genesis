"""Version and private-hook contract tests for Genesis 1.2.2+."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import pytest

import ufactory.simulation.compat as compat


@pytest.fixture(autouse=True)
def reset_unvalidated_warning(monkeypatch):
    monkeypatch.setattr(compat, "_WARNED_UNVALIDATED", False)


def test_version_below_minimum_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.1")

    with pytest.raises(compat.GenesisCompatibilityError, match=r"Genesis>=1\.2\.2 is required"):
        compat.require_genesis_version()


def test_validated_version_is_accepted_without_warning(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.2")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compat.require_genesis_version() == compat.VALIDATED_GENESIS_VERSION
    assert caught == []


def test_newer_version_warns_only_once(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.3")

    with pytest.warns(RuntimeWarning, match="only 1.2.2 has completed") as caught:
        compat.require_genesis_version()
        compat.require_genesis_version()
    assert len(caught) == 1


def test_newer_version_with_complete_capabilities_passes_and_warns(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.3")
    import genesis as gs

    with pytest.warns(RuntimeWarning, match="only 1.2.2 has completed") as caught:
        assert compat.require_genesis_capabilities(gs, pbr=True, deferred_viewer=True) is gs
    assert len(caught) == 1


def test_invalid_version_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "not-a-version")

    with pytest.raises(compat.GenesisCompatibilityError, match="Cannot parse"):
        compat.require_genesis_version()


def _parse_mesh_glb(path, group_by_material, scale, is_mesh_zup, surface):
    return path, group_by_material, scale, is_mesh_zup, surface


def _surface_uvs_to_trimesh_visual(surface, uvs=None, n_verts=None):
    return surface, uvs, n_verts


class _CompatibleMesh:
    @classmethod
    def from_trimesh(
        cls,
        mesh,
        scale=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=2,
        metadata=None,
        surface=None,
        is_mesh_zup=True,
    ):
        return (
            cls,
            mesh,
            scale,
            convexify,
            decimate,
            decimate_face_num,
            decimate_aggressiveness,
            metadata,
            surface,
            is_mesh_zup,
        )


def test_pbr_hook_contract_accepts_required_signatures(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.2")
    gs = SimpleNamespace(Mesh=_CompatibleMesh)
    gltf = SimpleNamespace(parse_mesh_glb=_parse_mesh_glb)
    mesh = SimpleNamespace(surface_uvs_to_trimesh_visual=_surface_uvs_to_trimesh_visual)

    compat.require_pbr_hooks(gs, gltf, mesh)


def test_pbr_hook_contract_rejects_changed_signature_before_patch(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.3")
    gs = SimpleNamespace(Mesh=_CompatibleMesh)
    gltf = SimpleNamespace(parse_mesh_glb=lambda path: path)
    mesh = SimpleNamespace(surface_uvs_to_trimesh_visual=_surface_uvs_to_trimesh_visual)

    with pytest.warns(RuntimeWarning):
        with pytest.raises(compat.GenesisCompatibilityError, match="missing parameters"):
            compat.require_pbr_hooks(gs, gltf, mesh)


def test_viewer_contract_rejects_missing_private_registry(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.2")

    with pytest.raises(compat.GenesisCompatibilityError, match="_scene_registry"):
        compat.load_deferred_viewer_api(SimpleNamespace())


def test_ik_scratch_contract_rejects_missing_solver_batch(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.2")
    robot = SimpleNamespace(_IK_qpos_orig=None, n_qs=6, _solver=SimpleNamespace())

    with pytest.raises(compat.GenesisCompatibilityError, match=r"robot\._solver\._B"):
        compat.ensure_ik_scratch(robot)


def test_ik_scratch_contract_allocates_expected_shape(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.2.2")
    robot = SimpleNamespace(_IK_qpos_orig=None, n_qs=6, _solver=SimpleNamespace(_B=4))
    calls = []
    qd = SimpleNamespace(field=lambda **kwargs: calls.append(kwargs) or "scratch")

    compat.ensure_ik_scratch(robot, gs_module=SimpleNamespace(qd_float="float"), qd_module=qd)

    assert robot._IK_qpos_orig == "scratch"
    assert calls == [{"dtype": "float", "shape": (6, 4)}]


def test_installed_genesis_122_matches_runtime_and_hook_contracts():
    if str(compat.require_genesis_version()) != "1.2.2":
        pytest.skip("the full installed-contract assertion targets the validated 1.2.2 baseline")
    import genesis as gs
    import genesis.utils.gltf as gltf_utils
    import genesis.utils.mesh as mesh_utils

    assert compat.require_genesis_runtime(gs) is gs
    compat.require_pbr_hooks(gs, gltf_utils, mesh_utils)
    viewer = compat.load_deferred_viewer_api(gs)
    assert viewer.default_aspect_ratio > 0.0
    assert viewer.default_height_ratio > 0.0
