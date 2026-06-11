"""Helpers for loading GLB visual URDFs with per-part PBR materials in Genesis."""

from __future__ import annotations

_glb_surface_queue: list = []
_patched = False


def enable_glb_pbr_surfaces() -> None:
    """Preserve metallic/roughness from GLB files (Genesis URDF loader overwrites them by default)."""
    global _patched
    if _patched:
        return

    import trimesh
    import genesis as gs
    import genesis.utils.gltf as gltf_utils
    import genesis.utils.mesh as mu

    _orig_parse = gltf_utils.parse_mesh_glb
    _orig_from_trimesh = gs.Mesh.from_trimesh
    _orig_surface_uvs_to_trimesh_visual = mu.surface_uvs_to_trimesh_visual

    def _pbr_factor(texture):
        if texture is None:
            return None
        color = getattr(texture, "color", None)
        if color is None:
            return None
        return float(color[0])

    def _surface_uvs_to_trimesh_pbr_visual(surface, uvs=None, n_verts=None):
        """Keep GLB metallic/roughness when Genesis converts surfaces for the rasterizer."""
        metallic = _pbr_factor(getattr(surface, "metallic_texture", None))
        roughness = _pbr_factor(getattr(surface, "roughness_texture", None))
        # Plastic shells (metallic≈0) must stay on vertex-color visuals; forcing them
        # through factor-only PBR + UVs causes harsh/incorrect shadowing on links.
        if metallic is None or metallic <= 0.01:
            return _orig_surface_uvs_to_trimesh_visual(surface, uvs, n_verts)
        if roughness is None:
            roughness = 0.5

        rgba = surface.get_rgba()
        if not isinstance(rgba, gs.textures.ColorTexture):
            return _orig_surface_uvs_to_trimesh_visual(surface, uvs, n_verts)

        color = tuple(float(c) for c in rgba.color)
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

    def _parse_mesh_glb(path, group_by_material, scale, is_mesh_zup, surface):
        meshes = _orig_parse(path, group_by_material, scale, is_mesh_zup, surface)
        for mesh in meshes:
            part_surface = mesh.surface.model_copy(deep=True)
            if part_surface.double_sided is None:
                part_surface.double_sided = True
            _glb_surface_queue.append(part_surface)
        return meshes

    def _from_trimesh(
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
        **kwargs,
    ):
        mesh_path = str((metadata or {}).get("mesh_path", "")).lower()
        if _glb_surface_queue and mesh_path.endswith((".glb", ".gltf")):
            surface = _glb_surface_queue.pop(0)
        return _orig_from_trimesh(
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

    gltf_utils.parse_mesh_glb = _parse_mesh_glb
    gs.Mesh.from_trimesh = classmethod(_from_trimesh)
    mu.surface_uvs_to_trimesh_visual = _surface_uvs_to_trimesh_pbr_visual
    _patched = True


def glb_view_surface():
    """Fallback surface when GLB PBR patch is enabled (only used for non-GLB geoms)."""
    import genesis as gs

    return gs.surfaces.Default(double_sided=True)
