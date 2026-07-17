"""Deterministic tests for the scoped PBR patch lifecycle."""

from __future__ import annotations

import threading

import pytest

import ufactory.visualization.glb as glb


@pytest.fixture(autouse=True)
def clean_patch_state(monkeypatch):
    monkeypatch.setattr(glb, "_REFCOUNT", 0)
    monkeypatch.setattr(glb, "_ORIGINALS", {})
    monkeypatch.setattr(glb, "_LEGACY_CONTEXTS", [])
    glb._queue().clear()


def test_nested_scope_installs_once_and_restores_after_outer_exit(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(glb, "require_genesis_version", lambda: None)
    monkeypatch.setattr(glb, "_install_patch", lambda: events.append("install"))
    monkeypatch.setattr(glb, "_restore_patch", lambda: events.append("restore"))

    with glb.glb_pbr_surfaces():
        glb._queue().append("outer")
        with glb.glb_pbr_surfaces():
            glb._queue().append("inner")
        assert glb._REFCOUNT == 1
        assert glb._queue() == []
    assert events == ["install", "restore"]
    assert glb._REFCOUNT == 0


def test_exception_restores_and_clears_material_queue(monkeypatch):
    restored: list[bool] = []
    monkeypatch.setattr(glb, "require_genesis_version", lambda: None)
    monkeypatch.setattr(glb, "_install_patch", lambda: None)
    monkeypatch.setattr(glb, "_restore_patch", lambda: restored.append(True))

    with pytest.raises(ValueError, match="load failed"):
        with glb.glb_pbr_surfaces():
            glb._queue().append(object())
            raise ValueError("load failed")
    assert restored == [True]
    assert glb._queue() == []


def test_incompatible_hook_fails_before_any_global_patch(monkeypatch):
    gs = pytest.importorskip("genesis")
    gltf_utils = pytest.importorskip("genesis.utils.gltf")
    mesh_utils = pytest.importorskip("genesis.utils.mesh")

    original_parse = gltf_utils.parse_mesh_glb
    original_from_trimesh = gs.Mesh.__dict__["from_trimesh"]
    original_surface_visual = mesh_utils.surface_uvs_to_trimesh_visual
    monkeypatch.setattr(glb, "require_genesis_version", lambda: None)

    def _reject(*_args):
        raise RuntimeError("future PBR hook is incompatible")

    monkeypatch.setattr(glb, "require_pbr_hooks", _reject)

    with pytest.raises(RuntimeError, match="future PBR hook"):
        with glb.glb_pbr_surfaces():
            pytest.fail("incompatible hook must fail before entering the scope")

    assert gltf_utils.parse_mesh_glb is original_parse
    assert gs.Mesh.__dict__["from_trimesh"] is original_from_trimesh
    assert mesh_utils.surface_uvs_to_trimesh_visual is original_surface_visual
    assert glb._ORIGINALS == {}
    assert glb._REFCOUNT == 0


def test_concurrent_scopes_share_patch_but_have_thread_local_queues(monkeypatch):
    events: list[str] = []
    barrier = threading.Barrier(2)
    observed: list[tuple[str, ...]] = []
    monkeypatch.setattr(glb, "require_genesis_version", lambda: None)
    monkeypatch.setattr(glb, "_install_patch", lambda: events.append("install"))
    monkeypatch.setattr(glb, "_restore_patch", lambda: events.append("restore"))

    def worker(name: str) -> None:
        with glb.glb_pbr_surfaces():
            glb._queue().append(name)
            barrier.wait()
            observed.append(tuple(glb._queue()))

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert sorted(observed) == [("a",), ("b",)]
    assert events == ["install", "restore"]
