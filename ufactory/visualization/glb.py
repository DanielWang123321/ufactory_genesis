"""Scoped Genesis 1.2.2+ GLB PBR preservation patch."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Iterator

from ufactory.simulation.compat import require_genesis_version, require_pbr_hooks

_LOCK = threading.RLock()
_LOCAL = threading.local()
_REFCOUNT = 0
_ORIGINALS: dict[str, Any] = {}
_LEGACY_CONTEXTS: list[Any] = []


def _queue() -> list[Any]:
    queue = getattr(_LOCAL, "surface_queue", None)
    if queue is None:
        queue = []
        _LOCAL.surface_queue = queue
    return queue


def _install_patch() -> None:
    import trimesh
    import genesis as gs
    import genesis.utils.gltf as gltf_utils
    import genesis.utils.mesh as mesh_utils

    require_pbr_hooks(gs, gltf_utils, mesh_utils)
    original_parse = gltf_utils.parse_mesh_glb
    original_bound_from_trimesh = gs.Mesh.from_trimesh
    original_from_trimesh_descriptor = gs.Mesh.__dict__["from_trimesh"]
    original_surface_visual = mesh_utils.surface_uvs_to_trimesh_visual
    _ORIGINALS.update(
        parse_mesh_glb=original_parse,
        from_trimesh_bound=original_bound_from_trimesh,
        from_trimesh_descriptor=original_from_trimesh_descriptor,
        surface_uvs_to_trimesh_visual=original_surface_visual,
        gltf_utils=gltf_utils,
        mesh_utils=mesh_utils,
        mesh_class=gs.Mesh,
    )

    def pbr_factor(texture: Any) -> float | None:
        if texture is None or getattr(texture, "color", None) is None:
            return None
        return float(texture.color[0])

    def surface_uvs_to_pbr(surface: Any, uvs: Any = None, n_verts: Any = None) -> Any:
        metallic = pbr_factor(getattr(surface, "metallic_texture", None))
        roughness = pbr_factor(getattr(surface, "roughness_texture", None))
        if metallic is None or metallic <= 0.01:
            return original_surface_visual(surface, uvs, n_verts)
        if roughness is None:
            roughness = 0.5
        rgba = surface.get_rgba()
        if not isinstance(rgba, gs.textures.ColorTexture):
            return original_surface_visual(surface, uvs, n_verts)
        color = tuple(float(channel) for channel in rgba.color)
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=color,
            metallicFactor=metallic,
            roughnessFactor=roughness,
            doubleSided=True if surface.double_sided is None else bool(surface.double_sided),
            alphaMode="OPAQUE" if len(color) < 4 or color[3] >= 1.0 else "BLEND",
        )
        if uvs is not None:
            uvs = uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            return trimesh.visual.TextureVisuals(uv=uvs, material=material)
        return trimesh.visual.TextureVisuals(material=material)

    def parse_mesh_glb(path: Any, group_by_material: Any, scale: Any, is_mesh_zup: Any, surface: Any) -> Any:
        meshes = original_parse(path, group_by_material, scale, is_mesh_zup, surface)
        queue = _queue()
        for mesh in meshes:
            part_surface = mesh.surface.model_copy(deep=True)
            if part_surface.double_sided is None:
                part_surface.double_sided = True
            queue.append(part_surface)
        return meshes

    def from_trimesh(
        cls: Any,
        mesh: Any,
        scale: Any = None,
        convexify: bool = False,
        decimate: bool = False,
        decimate_face_num: int = 500,
        decimate_aggressiveness: int = 2,
        metadata: Any = None,
        surface: Any = None,
        is_mesh_zup: bool = True,
        **kwargs: Any,
    ) -> Any:
        del cls
        queue = _queue()
        mesh_path = str((metadata or {}).get("mesh_path", "")).lower()
        if queue and mesh_path.endswith((".glb", ".gltf")):
            surface = queue.pop(0)
        return original_bound_from_trimesh(
            mesh,
            scale=scale,
            convexify=convexify,
            decimate=decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            metadata=metadata,
            surface=surface,
            is_mesh_zup=is_mesh_zup,
            **kwargs,
        )

    gltf_utils.parse_mesh_glb = parse_mesh_glb
    gs.Mesh.from_trimesh = classmethod(from_trimesh)
    mesh_utils.surface_uvs_to_trimesh_visual = surface_uvs_to_pbr


def _restore_patch() -> None:
    if not _ORIGINALS:
        return
    _ORIGINALS["gltf_utils"].parse_mesh_glb = _ORIGINALS["parse_mesh_glb"]
    setattr(_ORIGINALS["mesh_class"], "from_trimesh", _ORIGINALS["from_trimesh_descriptor"])
    _ORIGINALS["mesh_utils"].surface_uvs_to_trimesh_visual = _ORIGINALS["surface_uvs_to_trimesh_visual"]
    _ORIGINALS.clear()


@contextmanager
def glb_pbr_surfaces() -> Iterator[None]:
    """Install the patch for a scoped GLB load and always restore originals."""

    global _REFCOUNT
    require_genesis_version()
    with _LOCK:
        if _REFCOUNT == 0:
            _install_patch()
        _REFCOUNT += 1
    try:
        yield
    finally:
        _queue().clear()
        with _LOCK:
            _REFCOUNT -= 1
            if _REFCOUNT == 0:
                _restore_patch()


def enable_glb_pbr_surfaces() -> None:
    """Legacy explicit acquire; new code must prefer ``with glb_pbr_surfaces()``."""

    context = glb_pbr_surfaces()
    context.__enter__()
    _LEGACY_CONTEXTS.append(context)


def disable_glb_pbr_surfaces() -> None:
    """Release the most recent legacy acquire."""

    if not _LEGACY_CONTEXTS:
        return
    context = _LEGACY_CONTEXTS.pop()
    context.__exit__(None, None, None)


def glb_view_surface() -> Any:
    """Fallback surface for non-GLB geometries."""

    import genesis as gs

    return gs.surfaces.Default(double_sided=True)
