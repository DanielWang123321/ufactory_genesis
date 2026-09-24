"""Present-hold and first-frame gating for the Genesis interactive viewer.

Genesis' low-level ``Renderer.render`` skips a frame while shadow textures are
not yet in the GL context (it returns ``()``) and documents that the caller
should keep the previous frame. ``Viewer.on_draw`` still clears the back buffer
first, and ``Viewer.refresh`` always flips, so a skipped frame presents a
cleared (black or garbage) buffer. This patch makes a skipped on-screen scene
render suppress the flip, so the previous front buffer stays visible.

``Viewer.start`` also maps the window (``set_visible(True)``) before the first
successful present, and ``Viewer.refresh`` flips even on ticks where ``on_draw``
never ran (``_time_event`` is scheduled one full interval late). Either path can
put an undefined or black buffer on screen. This patch presents only after a
complete on-screen scene render, and defers ``set_visible`` / ``activate()``
until that first present so GL-config retries and empty frames stay off screen.
The deferred show still runs ``on_draw`` immediately: ``Viewer.start`` uses that
draw, inside its config ``try``, to reject a PyOpenGL platform that cannot see
the window context (EGL queried against a GLX window). Skipping the draw lets
initialization succeed and the later refresh crashes outside that handler.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from ufactory.simulation.compat import require_present_hold_hooks


_HOLD_FLAG = "_uf_hold_present"
_DRAWN_FLAG = "_uf_scene_drawn"
_READY_FLAG = "_uf_frame_ready"
_WANT_VISIBLE = "_uf_want_visible"
_PENDING_ACTIVATE = "_uf_pending_activate"
_LOCK = threading.RLock()
_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def is_skipped_render_result(result: object) -> bool:
    """True when ``Renderer.render`` skipped the scene pass (empty tuple)."""

    return isinstance(result, tuple) and len(result) == 0


def should_present_frame(viewer: Any) -> bool:
    """False unless this tick drew a complete on-screen scene frame.

    ``Viewer.refresh`` flips unconditionally, but ``on_draw`` is driven by
    ``_time_event`` / resize and is often absent on a tick. Presenting then
    swaps an undefined back buffer. Require a successful scene render first.
    """

    return bool(getattr(viewer, _DRAWN_FLAG, False)) and not getattr(viewer, _HOLD_FLAG, False)


def is_frame_ready(viewer: Any) -> bool:
    """True after the first complete on-screen frame has been presented."""

    return bool(getattr(viewer, _READY_FLAG, False))


def wrap_render_method(original: Callable[..., Any]) -> Callable[..., Any]:
    def _render(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        render_flags = getattr(self, "render_flags", None) or {}
        on_screen = not render_flags.get("offscreen", False)
        requested = bool(render_flags.get("rgb") or render_flags.get("depth") or render_flags.get("seg"))
        if on_screen and requested:
            skipped = is_skipped_render_result(result)
            setattr(self, _HOLD_FLAG, skipped)
            setattr(self, _DRAWN_FLAG, not skipped)
        return result

    return _render


def _promote_visibility(viewer: Any) -> None:
    """Show and raise the window once a complete frame is in the front buffer."""

    if not is_frame_ready(viewer):
        return
    if getattr(viewer, _WANT_VISIBLE, False):
        setattr(viewer, _WANT_VISIBLE, False)
        original_set_visible = _ORIGINALS.get("set_visible")
        if original_set_visible is not None:
            original_set_visible(viewer, True)
    if getattr(viewer, _PENDING_ACTIVATE, False):
        setattr(viewer, _PENDING_ACTIVATE, False)
        original_activate = _ORIGINALS.get("activate")
        if original_activate is not None:
            original_activate(viewer)


def wrap_flip_method(
    original: Callable[..., Any], *, on_presented: Callable[[Any], None] | None = None
) -> Callable[..., Any]:
    def flip(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_present_frame(self):
            return None
        result = original(self, *args, **kwargs)
        setattr(self, _DRAWN_FLAG, False)
        was_ready = is_frame_ready(self)
        setattr(self, _READY_FLAG, True)
        if not was_ready and on_presented is not None:
            on_presented(self)
        return result

    return flip


def wrap_set_visible_method(original: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the window unmapped until the first complete frame is presented."""

    def set_visible(self: Any, visible: bool = True) -> Any:
        if not visible:
            setattr(self, _WANT_VISIBLE, False)
            return original(self, False)
        if not is_frame_ready(self):
            setattr(self, _WANT_VISIBLE, True)
            # Probe while still unmapped. Viewer.start calls set_visible(True)
            # inside the GL-config try/except; the following refresh does not
            # draw, because the frame timer is not scheduled yet. Without this
            # draw, an EGL PyOpenGL platform is accepted for a GLX window.
            probe = getattr(self, "on_draw", None)
            if callable(probe):
                probe()
            return None
        return original(self, True)

    return set_visible


def wrap_activate_method(original: Callable[..., Any]) -> Callable[..., Any]:
    """Raise the window only after it is actually mapped with a good frame."""

    def activate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_frame_ready(self):
            setattr(self, _PENDING_ACTIVATE, True)
            return None
        return original(self, *args, **kwargs)

    return activate


def wrap_on_resize_method(original: Callable[..., Any]) -> Callable[..., Any]:
    """Avoid map-time FBO teardown before the first frame; later resizes hold the lock."""

    def on_resize(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_frame_ready(self):
            if args:
                try:
                    self._viewport_size = (int(args[0]), int(args[1]))
                except Exception:
                    pass
            return None
        lock = getattr(self, "render_lock", None)
        if lock is None:
            return original(self, *args, **kwargs)
        with lock:
            return original(self, *args, **kwargs)

    return on_resize


def _install_patch() -> None:
    require_present_hold_hooks()
    from genesis.ext.pyrender.viewer import Viewer

    _ORIGINALS.clear()
    _ORIGINALS.update(
        viewer_cls=Viewer,
        render=Viewer._render,
        flip=Viewer.flip,
        on_resize=Viewer.on_resize,
        set_visible=Viewer.set_visible,
        activate=Viewer.activate,
    )
    Viewer._render = wrap_render_method(_ORIGINALS["render"])
    Viewer.flip = wrap_flip_method(_ORIGINALS["flip"], on_presented=_promote_visibility)
    Viewer.on_resize = wrap_on_resize_method(_ORIGINALS["on_resize"])
    Viewer.set_visible = wrap_set_visible_method(_ORIGINALS["set_visible"])
    Viewer.activate = wrap_activate_method(_ORIGINALS["activate"])


def _restore_patch() -> None:
    if not _ORIGINALS:
        return
    viewer_cls = _ORIGINALS["viewer_cls"]
    viewer_cls._render = _ORIGINALS["render"]
    viewer_cls.flip = _ORIGINALS["flip"]
    viewer_cls.on_resize = _ORIGINALS["on_resize"]
    viewer_cls.set_visible = _ORIGINALS["set_visible"]
    viewer_cls.activate = _ORIGINALS["activate"]
    _ORIGINALS.clear()


def install_present_hold() -> None:
    """Install the present-hold and first-frame gating wrappers once per process."""

    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _install_patch()
        _INSTALLED = True


def uninstall_present_hold() -> None:
    """Restore the original methods (tests only)."""

    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            return
        _restore_patch()
        _INSTALLED = False
