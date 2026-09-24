"""Present-hold and first-frame gating tests for the Genesis viewer patch."""

from __future__ import annotations

import ufactory.visualization.render_patch as render_patch


class _Viewer:
    def __init__(self, *, offscreen: bool = False) -> None:
        self.render_flags = {"offscreen": offscreen, "rgb": True, "depth": False, "seg": False}
        self.flips = 0
        self.resizes = 0
        self.visible_calls: list[bool] = []
        self.activations = 0
        self.lock_depths: list[int] = []
        self._lock_depth = 0
        self._viewport_size = (800, 600)

    def render(self, camera_node=None, renderer=None, normal=False):
        return self._next_render_result

    def flip(self) -> str:
        self.flips += 1
        return "flipped"

    def on_resize(self, width: int, height: int) -> str:
        self.resizes += 1
        return f"{width}x{height}"

    def set_visible(self, visible: bool = True) -> bool:
        self.visible_calls.append(visible)
        return visible

    def activate(self) -> str:
        self.activations += 1
        return "activated"

    @property
    def render_lock(self):
        viewer = self

        class _Lock:
            def __enter__(self):
                viewer._lock_depth += 1
                viewer.lock_depths.append(viewer._lock_depth)
                return self

            def __exit__(self, *_exc):
                viewer._lock_depth -= 1
                return False

        return _Lock()


def test_is_skipped_render_result():
    assert render_patch.is_skipped_render_result(()) is True
    assert render_patch.is_skipped_render_result(None) is False
    assert render_patch.is_skipped_render_result(([],)) is False


def test_render_wrapper_marks_hold_only_on_empty_first_pass():
    viewer = _Viewer()
    wrapped = render_patch.wrap_render_method(_Viewer.render)

    viewer._next_render_result = ()
    assert wrapped(viewer) == ()
    assert viewer._uf_hold_present is True
    assert viewer._uf_scene_drawn is False

    viewer._next_render_result = None
    assert wrapped(viewer) is None
    assert viewer._uf_hold_present is False
    assert viewer._uf_scene_drawn is True


def test_render_wrapper_ignores_offscreen_and_unrequested_passes():
    offscreen = _Viewer(offscreen=True)
    wrapped = render_patch.wrap_render_method(_Viewer.render)
    offscreen._next_render_result = ()
    wrapped(offscreen)
    assert getattr(offscreen, "_uf_hold_present", False) is False

    unrequested = _Viewer()
    unrequested.render_flags["rgb"] = False
    unrequested._next_render_result = ()
    wrapped(unrequested)
    assert getattr(unrequested, "_uf_hold_present", False) is False


def test_refresh_skips_flip_when_present_hold():
    viewer = _Viewer()
    viewer._uf_scene_drawn = True
    viewer._uf_hold_present = True
    wrapped = render_patch.wrap_flip_method(_Viewer.flip)

    assert wrapped(viewer) is None
    assert viewer.flips == 0
    assert render_patch.is_frame_ready(viewer) is False


def test_refresh_skips_flip_when_on_draw_did_not_run():
    """refresh() flips even on ticks where _time_event never called on_draw."""

    viewer = _Viewer()
    viewer._uf_hold_present = False
    viewer._uf_scene_drawn = False
    wrapped = render_patch.wrap_flip_method(_Viewer.flip)

    assert wrapped(viewer) is None
    assert viewer.flips == 0
    assert render_patch.is_frame_ready(viewer) is False


def test_refresh_flips_when_frame_presented():
    viewer = _Viewer()
    viewer._uf_hold_present = False
    viewer._uf_scene_drawn = True
    wrapped = render_patch.wrap_flip_method(_Viewer.flip)

    assert wrapped(viewer) == "flipped"
    assert viewer.flips == 1
    assert viewer._uf_scene_drawn is False
    assert render_patch.is_frame_ready(viewer) is True


def test_set_visible_true_is_deferred_until_first_frame():
    viewer = _Viewer()
    wrapped = render_patch.wrap_set_visible_method(_Viewer.set_visible)

    assert wrapped(viewer, True) is None
    assert viewer.visible_calls == []
    assert viewer._uf_want_visible is True

    viewer._uf_frame_ready = True
    assert wrapped(viewer, True) is True
    assert viewer.visible_calls == [True]


def test_deferred_set_visible_runs_draw_probe():
    viewer = _Viewer()
    viewer.draws = 0

    def on_draw() -> None:
        viewer.draws += 1

    viewer.on_draw = on_draw
    wrapped = render_patch.wrap_set_visible_method(_Viewer.set_visible)

    assert wrapped(viewer, True) is None
    assert viewer.draws == 1
    assert viewer.visible_calls == []
    assert viewer._uf_want_visible is True


def test_deferred_set_visible_probe_error_propagates():
    viewer = _Viewer()

    def on_draw() -> None:
        raise RuntimeError("no context")

    viewer.on_draw = on_draw
    wrapped = render_patch.wrap_set_visible_method(_Viewer.set_visible)

    try:
        wrapped(viewer, True)
    except RuntimeError as exc:
        assert str(exc) == "no context"
    else:
        raise AssertionError("probe error was swallowed")
    assert viewer.visible_calls == []
    assert viewer._uf_want_visible is True


def test_set_visible_false_hides_immediately():
    viewer = _Viewer()
    viewer._uf_want_visible = True
    wrapped = render_patch.wrap_set_visible_method(_Viewer.set_visible)

    assert wrapped(viewer, False) is False
    assert viewer.visible_calls == [False]
    assert viewer._uf_want_visible is False


def test_activate_is_deferred_until_frame_ready():
    viewer = _Viewer()
    wrapped = render_patch.wrap_activate_method(_Viewer.activate)

    assert wrapped(viewer) is None
    assert viewer.activations == 0
    assert viewer._uf_pending_activate is True

    viewer._uf_frame_ready = True
    assert wrapped(viewer) == "activated"
    assert viewer.activations == 1


def test_first_successful_flip_promotes_visibility_and_activate(monkeypatch):
    viewer = _Viewer()
    promoted: list[tuple[str, bool | None]] = []
    monkeypatch.setitem(
        render_patch._ORIGINALS,
        "set_visible",
        lambda self, visible=True: promoted.append(("set_visible", visible)),
    )
    monkeypatch.setitem(
        render_patch._ORIGINALS,
        "activate",
        lambda self: promoted.append(("activate", None)),
    )
    viewer._uf_want_visible = True
    viewer._uf_pending_activate = True
    wrapped = render_patch.wrap_flip_method(_Viewer.flip, on_presented=render_patch._promote_visibility)

    viewer._uf_hold_present = False
    viewer._uf_scene_drawn = True
    assert wrapped(viewer) == "flipped"
    assert promoted == [("set_visible", True), ("activate", None)]
    assert viewer._uf_want_visible is False
    assert viewer._uf_pending_activate is False


def test_skipped_flip_does_not_mark_ready_or_promote(monkeypatch):
    viewer = _Viewer()
    promoted: list[bool] = []
    monkeypatch.setitem(
        render_patch._ORIGINALS,
        "set_visible",
        lambda self, visible=True: promoted.append(visible),
    )
    viewer._uf_want_visible = True
    wrapped = render_patch.wrap_flip_method(_Viewer.flip, on_presented=render_patch._promote_visibility)

    viewer._uf_hold_present = True
    viewer._uf_scene_drawn = True
    assert wrapped(viewer) is None
    assert render_patch.is_frame_ready(viewer) is False
    assert promoted == []


def test_config_retry_keeps_window_hidden_until_first_frame():
    viewer = _Viewer()
    show = render_patch.wrap_set_visible_method(_Viewer.set_visible)
    flip = render_patch.wrap_flip_method(_Viewer.flip, on_presented=render_patch._promote_visibility)

    # GL config attempt 1: warm-up refresh skips, then start() tries to show and validates.
    viewer._uf_hold_present = True
    flip(viewer)
    assert show(viewer, True) is None
    flip(viewer)  # validation refresh also skips
    # Failure path: hide + recreate. Must stay off screen the whole time.
    assert show(viewer, False) is False
    assert viewer.visible_calls == [False]

    # GL config attempt 2: a complete frame finally presents.
    viewer._uf_hold_present = False
    viewer._uf_scene_drawn = True
    assert flip(viewer) == "flipped"
    assert show(viewer, True) is True
    assert viewer.visible_calls == [False, True]


def test_on_resize_before_first_frame_skips_fbo_teardown():
    viewer = _Viewer()
    wrapped = render_patch.wrap_on_resize_method(_Viewer.on_resize)

    assert wrapped(viewer, 1024, 768) is None
    assert viewer.resizes == 0
    assert viewer._viewport_size == (1024, 768)

    viewer._uf_frame_ready = True
    assert wrapped(viewer, 800, 600) == "800x600"
    assert viewer.resizes == 1
    assert viewer.lock_depths == [1]


def test_install_present_hold_is_idempotent(monkeypatch):
    installs: list[int] = []
    restores: list[int] = []
    monkeypatch.setattr(render_patch, "_INSTALLED", False)
    monkeypatch.setattr(render_patch, "_install_patch", lambda: installs.append(1))
    monkeypatch.setattr(render_patch, "_restore_patch", lambda: restores.append(1))

    render_patch.install_present_hold()
    render_patch.install_present_hold()
    assert installs == [1]
    assert restores == []

    render_patch.uninstall_present_hold()
    render_patch.uninstall_present_hold()
    assert restores == [1]
