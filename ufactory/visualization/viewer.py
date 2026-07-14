"""Deferred Genesis interactive viewer startup."""

from __future__ import annotations

import sys

from ufactory.simulation.compat import load_deferred_viewer_api


def start_deferred_viewer(scene, *, kinematic_mirror: bool = False) -> None:
    """Open the interactive viewer after the scene has been initialized and warmed up.

    ``kinematic_mirror`` disables Genesis' simulation-time pacer.  A real-robot
    mirror is already paced by the servo callback, so letting ``Viewer.update``
    sleep to the same scene rate creates a second, drifting 50 Hz clock beside
    the safety-critical sender.
    """
    import genesis as gs

    visualizer = scene.visualizer
    if visualizer.viewer is not None:
        if kinematic_mirror:
            visualizer.viewer.realtime_factor = None
        return

    viewer_api = load_deferred_viewer_api(gs)

    live_other_scenes = [
        scene_ref() for scene_ref in gs._scene_registry if scene_ref() is not None and scene_ref() is not scene
    ]
    if live_other_scenes:
        gs.raise_exception(
            "Interactive viewer not supported when managing multiple scenes. Please set `show_viewer=False` "
            "or call `del scene`."
        )

    viewer_options = scene.viewer_options
    if kinematic_mirror:
        viewer_options.realtime_factor = None
        # An in-process mirror uploads poses at up to 15 Hz. Repainting the unchanged frame at
        # the simulation default (50-60 Hz) only competes with the real sender.
        viewer_options.refresh_rate = min(int(viewer_options.refresh_rate), 15)
    if viewer_options.res is None:
        try:
            screen_height, _screen_width, screen_scale = gs.utils.try_get_display_size()
        except Exception as exc:
            gs.raise_exception_from("No display detected. Use `show_viewer=False` for headless mode.", exc)
        viewer_height = (screen_height * screen_scale) * viewer_api.default_height_ratio
        viewer_width = viewer_height / viewer_api.default_aspect_ratio
        viewer_options.res = (int(viewer_width), int(viewer_height))
    if viewer_options.run_in_thread is None:
        if sys.platform == "linux":
            viewer_options.run_in_thread = True
        elif sys.platform == "darwin":
            viewer_options.run_in_thread = False
        elif sys.platform == "win32":
            viewer_options.run_in_thread = True
    if sys.platform == "darwin" and viewer_options.run_in_thread:
        gs.raise_exception("Running viewer in background thread is not supported on MacOS.")

    viewer = viewer_api.viewer_type(viewer_options, visualizer._context)
    visualizer._viewer = viewer
    if getattr(visualizer, "_rasterizer", None) is not None:
        visualizer._rasterizer._viewer = viewer
        visualizer._rasterizer._offscreen = False
    viewer.build(scene)
    visualizer.viewer_lock = viewer.lock
    visualizer.reset()
