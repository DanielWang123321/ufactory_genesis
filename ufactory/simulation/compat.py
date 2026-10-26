"""Genesis version and private-hook compatibility checks.

Genesis 1.4.2 remains the minimum and 1.4.5 is the maximum; newer releases are
rejected until verified (keep ``MAX_GENESIS_VERSION`` in sync with the ``[sim]``
extra). Genesis 1.4.5 is the reference baseline with maintainer simulation
checks. 1.4.2 to 1.4.4 still run and emit one warning;
their contact results are not part of that verification. On 1.4.5 the
SupportField azimuth index underflow is patched at runtime (see
``patch_support_field_direction_cell``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import inspect
import textwrap
import threading
from types import ModuleType
from typing import Any, Callable
import warnings

from packaging.version import InvalidVersion, Version


MIN_GENESIS_VERSION = Version("1.4.2")
VALIDATED_GENESIS_VERSION = Version("1.4.5")  # reference / pinned baseline alias
MAX_GENESIS_VERSION = Version("1.4.5")  # keep in sync with the [sim] extra upper bound

_WARNING_LOCK = threading.Lock()
_WARNED_UNVALIDATED = False


class GenesisCompatibilityError(RuntimeError):
    """Genesis is missing or does not satisfy the project's runtime contract."""


@dataclass(frozen=True)
class DeferredViewerAPI:
    viewer_type: type
    default_aspect_ratio: float
    default_height_ratio: float


def require_genesis_version() -> Version:
    """Return the installed Genesis version after enforcing the minimum and maximum."""

    try:
        raw_version = metadata.version("genesis-world")
    except metadata.PackageNotFoundError as exc:
        raise GenesisCompatibilityError(
            f"Genesis is not installed; install genesis-world>={MIN_GENESIS_VERSION},<={MAX_GENESIS_VERSION}."
        ) from exc
    try:
        version = Version(raw_version)
    except InvalidVersion as exc:
        raise GenesisCompatibilityError(f"Cannot parse installed Genesis version: {raw_version!r}.") from exc
    if version < MIN_GENESIS_VERSION:
        raise GenesisCompatibilityError(f"Genesis>={MIN_GENESIS_VERSION} is required; found {raw_version}.")
    if version > MAX_GENESIS_VERSION:
        raise GenesisCompatibilityError(
            f"Genesis<={MAX_GENESIS_VERSION} is required; found {raw_version}. "
            "Newer releases are not verified by this project."
        )
    if version != VALIDATED_GENESIS_VERSION:
        _warn_unvalidated_version(version)
    return version


def _warn_unvalidated_version(version: Version) -> None:
    global _WARNED_UNVALIDATED
    with _WARNING_LOCK:
        if _WARNED_UNVALIDATED:
            return
        warnings.warn(
            f"Genesis {version} satisfies the minimum version, but only {VALIDATED_GENESIS_VERSION} "
            "is the project's reference baseline with maintainer sim/hardware verification. "
            "Compatibility hooks will be checked before use.",
            RuntimeWarning,
            stacklevel=3,
        )
        _WARNED_UNVALIDATED = True


def _require_callable(owner: Any, name: str, context: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise GenesisCompatibilityError(f"Genesis {context} requires callable {name!r}.")
    return value


def _require_parameters(function: Callable[..., Any], required: set[str], context: str) -> None:
    try:
        parameters = set(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise GenesisCompatibilityError(f"Cannot inspect Genesis {context} signature.") from exc
    missing = sorted(required - parameters)
    if missing:
        raise GenesisCompatibilityError(f"Genesis {context} is incompatible; missing parameters: {', '.join(missing)}.")


def require_genesis_runtime(gs_module: ModuleType | Any | None = None) -> Any:
    """Validate the public runtime and kinematics entry points used by the project."""

    require_genesis_version()
    if gs_module is None:
        try:
            import genesis as gs_module
        except ImportError as exc:
            raise GenesisCompatibilityError("The genesis package cannot be imported.") from exc

    for name in ("init", "destroy", "Scene"):
        _require_callable(gs_module, name, "runtime")
    for name in ("options", "morphs", "surfaces"):
        if getattr(gs_module, name, None) is None:
            raise GenesisCompatibilityError(f"Genesis runtime requires module attribute {name!r}.")
    for name in ("constraint_solver", "friction_cone", "contact_resolution"):
        if getattr(gs_module, name, None) is None:
            raise GenesisCompatibilityError(f"Genesis runtime requires module attribute {name!r}.")
    rigid_options = _require_callable(gs_module.options, "RigidOptions", "rigid options")
    _require_parameters(
        rigid_options,
        {"constraint_solver", "friction_cone", "contact_resolution", "noslip_iterations"},
        "RigidOptions",
    )

    try:
        from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
        from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    except ImportError as exc:
        raise GenesisCompatibilityError("Genesis rigid-entity kinematics API is unavailable.") from exc
    forward = _require_callable(RigidSolver, "forward_kinematics_query", "forward kinematics")
    inverse = _require_callable(RigidEntity, "inverse_kinematics", "inverse kinematics")
    _require_parameters(forward, {"entity", "qpos"}, "RigidSolver.forward_kinematics_query")
    _require_parameters(
        inverse,
        {
            "link",
            "pos",
            "quat",
            "init_qpos",
            "dofs_idx_local",
            "damping",
            "respect_joint_limit",
            "return_error",
            "max_solver_iters",
        },
        "RigidEntity.inverse_kinematics",
    )
    if getattr(gs_module, "__name__", None) == "genesis":
        patch_support_field_direction_cell()
        patch_mesh_decimate(getattr(gs_module, "Mesh", None))
    return gs_module


# Dedented source of SupportField._func_direction_cell releases that carry the azimuth index underflow
# (Genesis 1.4.5, commit 375fd7eb). Only these exact implementations are replaced.
_DIRECTION_CELL_UNDERFLOW_SHA256 = frozenset({"42e904138606e51fd8b66fbc707f63e1ced0b623b3ec4d94f63d64d2b5effabd"})
_SUPPORT_FIELD_PATCH_LOCK = threading.Lock()
_MESH_DECIMATE_PATCH_ATTR = "_ufactory_decimate_post_process_patched"


def patch_mesh_decimate(mesh_cls: Any | None = None) -> bool:
    """Ensure ``Mesh.decimate`` rechecks the face count after ``process(validate=True)``.

    Genesis 1.4.2-1.4.5 checks ``len(self._mesh.faces) > decimate_face_num`` before calling
    ``self._mesh.process(validate=True)``. When ``process(validate=True)`` drops degenerate faces and brings the face
    count to or below ``decimate_face_num`` (e.g. ``hull_012_cup_01.stl`` from 504 to 499 faces under SciPy 1.18.1),
    ``fast_simplification.simplify`` raises ``ValueError: ``target_count`` must be less than the number of faces``.
    Running ``process(validate=True)`` before delegating lets the existing guard inside ``Mesh.decimate`` skip
    ``fast_simplification.simplify`` when the cleaned mesh already satisfies ``decimate_face_num``, while still running
    ``clear_visuals()`` and the ``@mutates_tmesh`` cache release.
    """

    with _SUPPORT_FIELD_PATCH_LOCK:
        if mesh_cls is None:
            try:
                from genesis.engine.mesh import Mesh as mesh_cls
            except ImportError:
                return False
        if mesh_cls is None:
            return False
        original = getattr(mesh_cls, "decimate", None)
        if not callable(original) or getattr(original, _MESH_DECIMATE_PATCH_ATTR, False):
            return False

        def _decimate_with_post_process_guard(self: Any, decimate_face_num: int, decimate_aggressiveness: int) -> Any:
            tmesh = getattr(self, "_mesh", None)
            if tmesh is not None and tmesh.vertices.shape[0] > 3 and len(tmesh.faces) > decimate_face_num:
                tmesh.process(validate=True)
            return original(self, decimate_face_num, decimate_aggressiveness)

        setattr(_decimate_with_post_process_guard, _MESH_DECIMATE_PATCH_ATTR, True)
        mesh_cls.decimate = _decimate_with_post_process_guard
        return True


def _direction_cell_lacks_lower_wrap(source: str) -> bool:
    return "if i >= support_res:" in source and "if i < 0:" not in source


def patch_support_field_direction_cell(support_field: ModuleType | Any | None = None) -> bool:
    """Restore the lower-bound wrap of the SupportField azimuth index.

    Genesis 1.4.5 replaced ``% support_res`` with an upper-bound check only. For a direction whose azimuth is just
    above -pi, rounding yields index -1 and the lookup reads the previous geom's support table, or the wrong chart of
    the same geom, which can create false penetrations. Genesis 1.4.4 and older are unaffected and left untouched.

    Must run before the first scene build compiles the collision kernels. Returns True when this call patched.
    """

    with _SUPPORT_FIELD_PATCH_LOCK:
        if support_field is None:
            try:
                from genesis.engine.solvers.rigid.collider import support_field
            except ImportError:
                return False
        current = getattr(support_field, "_func_direction_cell", None)
        if current is None:
            return False
        try:
            source = textwrap.dedent(inspect.getsource(current))
        except (OSError, TypeError) as exc:
            raise GenesisCompatibilityError("Cannot inspect Genesis SupportField._func_direction_cell source.") from exc
        if not _direction_cell_lacks_lower_wrap(source):
            return False
        if hashlib.sha256(source.encode()).hexdigest() not in _DIRECTION_CELL_UNDERFLOW_SHA256:
            raise GenesisCompatibilityError(
                "Genesis SupportField._func_direction_cell lacks the lower-bound azimuth wrap, but its implementation "
                "is not a known release; the compatibility patch cannot be applied safely."
            )
        support_field._func_direction_cell = _build_direction_cell_wrap_fix()
        return True


def _build_direction_cell_wrap_fix() -> Callable[..., Any]:
    # Kept in its own module: Quadrants resolves type annotations at decoration time, which postponed annotations
    # (``from __future__ import annotations`` here) would break.
    from ufactory.simulation._support_field_fix import func_direction_cell

    return func_direction_cell


def require_genesis_capabilities(
    gs_module: ModuleType | Any | None = None,
    *,
    pbr: bool = False,
    deferred_viewer: bool = False,
) -> Any:
    """Validate requested Genesis capabilities without initializing a scene.

    Hardware-facing commands call this before connecting to a controller so a
    future Genesis release cannot fail late after external state is involved.
    """

    gs_module = require_genesis_runtime(gs_module)
    if pbr:
        try:
            import genesis.utils.gltf as gltf_utils
            import genesis.utils.mesh as mesh_utils
        except ImportError as exc:
            raise GenesisCompatibilityError("Genesis GLB PBR utility modules are unavailable.") from exc
        require_pbr_hooks(gs_module, gltf_utils, mesh_utils)
    if deferred_viewer:
        load_deferred_viewer_api(gs_module)
        require_present_hold_hooks()
    return gs_module


def require_pbr_hooks(gs_module: Any, gltf_utils: Any, mesh_utils: Any) -> None:
    """Validate every Genesis hook patched by the scoped GLB PBR integration."""

    require_genesis_version()
    parse_mesh_glb = _require_callable(gltf_utils, "parse_mesh_glb", "GLB parser")
    surface_to_visual = _require_callable(mesh_utils, "surface_uvs_to_trimesh_visual", "surface-to-trimesh conversion")
    mesh_class = getattr(gs_module, "Mesh", None)
    from_trimesh = getattr(mesh_class, "from_trimesh", None) if mesh_class is not None else None
    if not callable(from_trimesh) or "from_trimesh" not in getattr(mesh_class, "__dict__", {}):
        raise GenesisCompatibilityError("Genesis Mesh.from_trimesh classmethod hook is unavailable.")
    _require_parameters(
        parse_mesh_glb,
        {"path", "group_by_material", "scale", "is_mesh_zup", "surface"},
        "parse_mesh_glb",
    )
    _require_parameters(
        from_trimesh,
        {
            "mesh",
            "scale",
            "convexify",
            "decimate",
            "decimate_face_num",
            "decimate_aggressiveness",
            "metadata",
            "surface",
            "is_mesh_zup",
        },
        "Mesh.from_trimesh",
    )
    _require_parameters(surface_to_visual, {"surface", "uvs", "n_verts"}, "surface_uvs_to_trimesh_visual")


def require_present_hold_hooks() -> None:
    """Validate every Genesis hook patched by the present-hold viewer integration."""

    require_genesis_version()
    try:
        from genesis.ext.pyrender.renderer import Renderer
        from genesis.ext.pyrender.viewer import Viewer
    except ImportError as exc:
        raise GenesisCompatibilityError("Genesis pyrender viewer/renderer modules are unavailable.") from exc
    _require_callable(Viewer, "_render", "present-hold")
    _require_callable(Viewer, "flip", "present-hold")
    _require_callable(Viewer, "on_resize", "present-hold")
    _require_callable(Viewer, "set_visible", "present-hold")
    _require_callable(Viewer, "activate", "present-hold")
    render = _require_callable(Renderer, "render", "present-hold")
    _require_parameters(render, {"scene", "flags", "is_first_pass"}, "Renderer.render")


def load_deferred_viewer_api(gs_module: Any) -> DeferredViewerAPI:
    """Load and validate the private Viewer entry points used for late startup."""

    require_genesis_version()
    if not isinstance(getattr(gs_module, "_scene_registry", None), list):
        raise GenesisCompatibilityError("Genesis deferred Viewer requires the private _scene_registry list.")
    try:
        from genesis.vis.viewer import Viewer
        from genesis.vis.visualizer import VIEWER_DEFAULT_ASPECT_RATIO, VIEWER_DEFAULT_HEIGHT_RATIO
    except ImportError as exc:
        raise GenesisCompatibilityError("Genesis deferred Viewer modules are unavailable.") from exc
    _require_parameters(Viewer, {"options", "context"}, "Viewer")
    try:
        aspect_ratio = float(VIEWER_DEFAULT_ASPECT_RATIO)
        height_ratio = float(VIEWER_DEFAULT_HEIGHT_RATIO)
    except (TypeError, ValueError) as exc:
        raise GenesisCompatibilityError("Genesis Viewer default ratios are invalid.") from exc
    if aspect_ratio <= 0.0 or height_ratio <= 0.0:
        raise GenesisCompatibilityError("Genesis Viewer default ratios must be positive.")
    return DeferredViewerAPI(Viewer, aspect_ratio, height_ratio)


def forward_kinematics(
    robot: Any,
    qpos: Any,
    envs_idx: Any | None = None,
) -> tuple[Any, Any]:
    """Evaluate link positions and orientations for an entity at the specified configuration.

    Wraps Genesis solver forward_kinematics_query. Returns (links_pos, links_quat).
    """
    solver = getattr(robot, "solver", None)
    if solver is None:
        solver = getattr(robot, "_solver", None)
    if solver is not None and hasattr(solver, "forward_kinematics_query"):
        return solver.forward_kinematics_query(robot, qpos, envs_idx=envs_idx)
    if hasattr(robot, "forward_kinematics"):
        return robot.forward_kinematics(qpos=qpos)
    raise GenesisCompatibilityError("Robot has no solver with forward_kinematics_query; build the scene first.")
