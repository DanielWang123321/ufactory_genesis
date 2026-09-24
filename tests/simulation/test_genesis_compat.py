"""Version and private-hook contract tests for the Genesis 1.4.2 baseline."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import pytest

import ufactory.simulation.compat as compat


@pytest.fixture(autouse=True)
def reset_unvalidated_warning(monkeypatch):
    monkeypatch.setattr(compat, "_WARNED_UNVALIDATED", False)


def test_version_below_minimum_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.1")

    with pytest.raises(compat.GenesisCompatibilityError, match=r"Genesis>=1\.4\.2 is required"):
        compat.require_genesis_version()


def test_validated_version_is_accepted_without_warning(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.2")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compat.require_genesis_version() == compat.VALIDATED_GENESIS_VERSION
    assert caught == []


def test_older_version_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.3.3")

    with pytest.raises(compat.GenesisCompatibilityError, match=r"Genesis>=1\.4\.2 is required"):
        compat.require_genesis_version()


def test_newer_version_warns_only_once(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.3")

    with pytest.warns(RuntimeWarning, match="only 1.4.2 is the project's reference baseline") as caught:
        compat.require_genesis_version()
        compat.require_genesis_version()
    assert len(caught) == 1


def test_newer_version_with_complete_capabilities_passes_and_warns(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.3")
    gs = pytest.importorskip("genesis")

    with pytest.warns(RuntimeWarning, match="only 1.4.2 is the project's reference baseline") as caught:
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
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.2")
    gs = SimpleNamespace(Mesh=_CompatibleMesh)
    gltf = SimpleNamespace(parse_mesh_glb=_parse_mesh_glb)
    mesh = SimpleNamespace(surface_uvs_to_trimesh_visual=_surface_uvs_to_trimesh_visual)

    compat.require_pbr_hooks(gs, gltf, mesh)


def test_pbr_hook_contract_rejects_changed_signature_before_patch(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.3")
    gs = SimpleNamespace(Mesh=_CompatibleMesh)
    gltf = SimpleNamespace(parse_mesh_glb=lambda path: path)
    mesh = SimpleNamespace(surface_uvs_to_trimesh_visual=_surface_uvs_to_trimesh_visual)

    with pytest.warns(RuntimeWarning):
        with pytest.raises(compat.GenesisCompatibilityError, match="missing parameters"):
            compat.require_pbr_hooks(gs, gltf, mesh)


def test_present_hold_hook_contract_accepts_required_methods(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.2")

    class _Viewer:
        def _render(self, camera_node=None, renderer=None, normal=False):
            return ()

        def flip(self):
            return None

        def on_resize(self, width, height):
            return None

        def set_visible(self, visible=True):
            return None

        def activate(self):
            return None

    class _Renderer:
        def render(self, scene, flags, seg_node_map=None, *, is_first_pass=True, force_skip_shadows=False):
            return ()

    monkeypatch.setitem(__import__("sys").modules, "genesis.ext.pyrender.viewer", SimpleNamespace(Viewer=_Viewer))
    monkeypatch.setitem(__import__("sys").modules, "genesis.ext.pyrender.renderer", SimpleNamespace(Renderer=_Renderer))

    compat.require_present_hold_hooks()


def test_present_hold_hook_contract_rejects_missing_refresh(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.2")

    class _Viewer:
        def _render(self, camera_node=None, renderer=None, normal=False):
            return ()

        def on_resize(self, width, height):
            return None

        def set_visible(self, visible=True):
            return None

        def activate(self):
            return None

    class _Renderer:
        def render(self, scene, flags, seg_node_map=None, *, is_first_pass=True, force_skip_shadows=False):
            return ()

    monkeypatch.setitem(__import__("sys").modules, "genesis.ext.pyrender.viewer", SimpleNamespace(Viewer=_Viewer))
    monkeypatch.setitem(__import__("sys").modules, "genesis.ext.pyrender.renderer", SimpleNamespace(Renderer=_Renderer))

    with pytest.raises(compat.GenesisCompatibilityError, match="flip"):
        compat.require_present_hold_hooks()


def test_viewer_contract_rejects_missing_private_registry(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.2")

    with pytest.raises(compat.GenesisCompatibilityError, match="_scene_registry"):
        compat.load_deferred_viewer_api(SimpleNamespace())


def test_forward_kinematics_calls_solver_query():
    calls = []
    solver = SimpleNamespace(
        forward_kinematics_query=lambda robot, qpos, envs_idx=None: (
            calls.append((robot, qpos, envs_idx)) or ("pos", "quat")
        )
    )
    robot = SimpleNamespace(solver=solver)
    result = compat.forward_kinematics(robot, "dummy_qpos", envs_idx=0)
    assert result == ("pos", "quat")
    assert calls == [(robot, "dummy_qpos", 0)]


def test_forward_kinematics_missing_solver_raises():
    robot = SimpleNamespace(solver=None, _solver=None)
    with pytest.raises(compat.GenesisCompatibilityError, match="Robot has no solver"):
        compat.forward_kinematics(robot, "dummy_qpos")


def test_installed_genesis_matches_runtime_and_hook_contracts():
    gs = pytest.importorskip("genesis")
    gltf_utils = pytest.importorskip("genesis.utils.gltf")
    mesh_utils = pytest.importorskip("genesis.utils.mesh")
    try:
        version = compat.require_genesis_version()
    except compat.GenesisCompatibilityError as exc:
        pytest.skip(f"genesis-world metadata unavailable: {exc}")
    if str(version) != "1.4.2":
        pytest.skip("the full installed-contract assertion targets the reference 1.4.2 baseline")

    assert compat.require_genesis_runtime(gs) is gs
    compat.require_pbr_hooks(gs, gltf_utils, mesh_utils)
    compat.require_present_hold_hooks()
    viewer = compat.load_deferred_viewer_api(gs)
    assert viewer.default_aspect_ratio > 0.0
    assert viewer.default_height_ratio > 0.0


def test_start_deferred_viewer_resolves_options_from_scene_options(monkeypatch):
    from ufactory.visualization.viewer import start_deferred_viewer

    viewer_options = SimpleNamespace(res=(800, 600), run_in_thread=True, realtime_factor=1.0, refresh_rate=60)
    built = False

    class DummyViewer:
        def __init__(self, opts, ctx):
            self.opts = opts
            self.lock = "dummy_lock"

        def build(self, scene):
            nonlocal built
            built = True

    monkeypatch.setattr(
        "ufactory.visualization.viewer.load_deferred_viewer_api",
        lambda gs: SimpleNamespace(viewer_type=DummyViewer, default_aspect_ratio=1.33, default_height_ratio=0.8),
    )
    visualizer = SimpleNamespace(viewer=None, _context="dummy_ctx", reset=lambda: None)
    scene = SimpleNamespace(visualizer=visualizer, options=SimpleNamespace(viewer=viewer_options))

    start_deferred_viewer(scene)
    assert built is True
    assert visualizer.viewer_lock == "dummy_lock"


def test_start_deferred_viewer_primes_visual_states_before_build(monkeypatch):
    from ufactory.visualization.viewer import start_deferred_viewer

    calls: list[str] = []

    class DummyContext:
        def update(self, *, force_render: bool = False):
            calls.append(f"prime:force_render={force_render}")

    class DummyViewer:
        def __init__(self, opts, ctx):
            self.lock = "dummy_lock"

        def build(self, scene):
            calls.append("build")

    monkeypatch.setattr(
        "ufactory.visualization.viewer.load_deferred_viewer_api",
        lambda gs: SimpleNamespace(viewer_type=DummyViewer, default_aspect_ratio=1.33, default_height_ratio=0.8),
    )
    viewer_options = SimpleNamespace(res=(800, 600), run_in_thread=True, realtime_factor=1.0, refresh_rate=60)
    visualizer = SimpleNamespace(viewer=None, _context=DummyContext(), reset=lambda: calls.append("reset"))
    scene = SimpleNamespace(visualizer=visualizer, options=SimpleNamespace(viewer=viewer_options))

    start_deferred_viewer(scene)

    # reset() 不再由 start_deferred_viewer 显式调用，
    # 已由 viewer.build() → visualizer.build() 内部完成。
    assert calls == ["prime:force_render=True", "build"]
