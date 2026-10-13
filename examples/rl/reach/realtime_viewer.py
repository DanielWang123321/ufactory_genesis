"""Main-thread viewer startup and wall-clock pacing for reach visualization."""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch

import genesis as gs
from ufactory.visualization.render_patch import install_present_hold, should_present_frame
from ufactory.visualization.viewer import prime_visual_states

from .runtime import OUTPUT_ROOT


class RealtimeClock:
    def __init__(self, period_s: float) -> None:
        self.period_s = period_s
        self.deadline = time.perf_counter() + period_s

    def wait(self) -> None:
        now = time.perf_counter()
        if now < self.deadline:
            time.sleep(self.deadline - now)
            now = time.perf_counter()
        # An overrun delays the next deadline; there is no catch-up stepping.
        self.deadline = max(self.deadline, now) + self.period_s


def prepare_x11_background(window) -> None:
    """Give Xwayland the ready scene before mapping allocates its display surface.

    GLX swaps on an unmapped window fill GL buffers, but Xwayland's initial
    mapped surface can still be black until the next swap. A background pixmap
    makes even that first surface show the scene. Pyglet's Xlib bindings avoid
    adding another window-system dependency.
    """
    import ctypes

    from OpenGL.GL import GL_BGRA, GL_FRONT, GL_UNSIGNED_BYTE, glReadBuffer, glReadPixels
    from pyglet.libs.x11 import xlib

    width, height = window.get_framebuffer_size()
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE)
    # X images start at the top left; GL images start at the bottom left.
    pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)[::-1].tobytes()
    data = ctypes.create_string_buffer(pixels)
    visual = window.config.get_visual_info()
    connection = window._x_display
    pixmap = xlib.XCreatePixmap(connection, window._window, width, height, visual.depth)
    gc = xlib.XCreateGC(connection, pixmap, 0, None)
    ximage = xlib.XCreateImage(
        connection,
        visual.visual,
        visual.depth,
        xlib.ZPixmap,
        0,
        ctypes.cast(data, ctypes.c_char_p),
        width,
        height,
        32,
        width * 4,
    )
    try:
        if not ximage:
            raise RuntimeError("Could not prepare the initial X11 scene image")
        xlib.XPutImage(connection, pixmap, gc, ximage, 0, 0, 0, 0, width, height)
        # The rendering child covers the top-level window, so both need the image.
        for xid in (window._window, window._view):
            xlib.XSetWindowBackgroundPixmap(connection, xid, pixmap)
        xlib.XSync(connection, False)
    finally:
        if ximage:
            # Python owns the pixel buffer; free only XCreateImage's structure.
            xlib.XFree(ximage)
        xlib.XFreeGC(connection, gc)
        # Windows retain the pixmap until their background is cleared or destroyed.
        xlib.XFreePixmap(connection, pixmap)


def warmup(env, policy, *, steps: int = 3) -> None:
    """Compile physics and inference, then restore the scene, episode and RNG states."""
    scene_state = env.scene.get_state()
    episode_state = env.training_state_dict()
    holds = env.hold_steps_left.clone()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        observation = env.get_observations()
        with torch.inference_mode():
            for _ in range(steps):
                observation, *_ = env.step(policy(observation))
    finally:
        env.scene.reset(scene_state)
        env.load_training_state_dict(episode_state)
        env.hold_steps_left.copy_(holds)
        env.extras = {"observations": {}}
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])


def start_viewer(env):
    """Create one hidden GLX window, fill both buffers, then map it once."""
    from genesis.vis.viewer import Viewer

    scene = env.scene
    visualizer = scene.visualizer
    options = scene.options.viewer.model_copy(deep=True)
    # Fit all displayed environments, including the workspace around each base.
    span = np.ptp(scene.envs_offset, axis=0)
    radius = float(np.linalg.norm((span[:2] + 1.5) / 2.0))
    distance = max(2.4, radius / math.sin(math.radians(options.camera_fov / 2.0)))
    lookat = np.asarray((0.15, 0.0, 0.4))
    direction = np.asarray((1.5, -1.5, 1.1))
    if env.num_envs > 1:
        options.camera_lookat = tuple(lookat)
        options.camera_pos = tuple(lookat + direction / np.linalg.norm(direction) * distance)

    rasterizer = visualizer._rasterizer
    # Headless scene.build creates a hidden 1x1 offscreen context. It is no
    # longer needed once rendering is handed to the interactive viewer.
    if rasterizer._renderer is not None:
        rasterizer._renderer.delete()
        rasterizer._renderer = None
    install_present_hold()
    prime_visual_states(scene)
    viewer = Viewer(options, visualizer._context)
    visualizer._viewer = viewer
    rasterizer._viewer = viewer
    rasterizer._offscreen = False
    headless = os.environ.get("GS_HEADLESS")
    os.environ["GS_HEADLESS"] = "1"
    try:
        viewer.build(scene)
        visualizer.viewer_lock = viewer.lock
        window = viewer._pyrender_viewer
        save_directory = OUTPUT_ROOT / "visualization"
        save_directory.mkdir(parents=True, exist_ok=True)
        window.viewer_flags["save_directory"] = str(save_directory)
        presented = 0
        for _ in range(8):
            prime_visual_states(scene)
            window.on_draw()
            if should_present_frame(window):
                window.flip()
                presented += 1
                if presented == 2:
                    break
        if presented < 2:
            raise RuntimeError("Viewer did not draw a complete scene while hidden")
        # Wait for GL commands before the compositor can read either buffer.
        from OpenGL.GL import glFinish

        glFinish()
        prepare_x11_background(window)
    finally:
        if headless is None:
            os.environ.pop("GS_HEADLESS", None)
        else:
            os.environ["GS_HEADLESS"] = headless
    window.set_visible(True)
    # Publish immediately after mapping; do not wait for the first policy tick.
    viewer.update(auto_refresh=True, force=True)
    from pyglet.libs.x11 import xlib

    for xid in (window._window, window._view):
        xlib.XSetWindowBackgroundPixmap(window._x_display, xid, 0)
    xlib.XFlush(window._x_display)
    window.activate()
    viewer._last_refresh_time = None
    return viewer


def shutdown(env=None) -> None:
    """Stop redraw callbacks, release Genesis and finish X11 window destruction."""
    import pyglet
    from pyglet.libs.x11 import xlib

    windows = []
    viewer = None
    if env is not None:
        visualizer = env.scene.visualizer
        viewer = visualizer.viewer
        if viewer is not None and viewer._pyrender_viewer is not None:
            window = viewer._pyrender_viewer
            windows.append(window)
            pyglet.clock.unschedule(type(window)._time_event)
        rasterizer = visualizer._rasterizer
        if rasterizer is not None and rasterizer._renderer is not None:
            platform = rasterizer._renderer._platform
            window = getattr(platform, "_window", None)
            if window is not None:
                windows.append(window)
    try:
        if viewer is not None:
            viewer.stop()
    finally:
        try:
            gs.destroy()
        finally:
            # Pyglet queues XDestroyWindow without flushing the connection.
            # Keep each Display alive until the server has processed that request.
            for window in windows:
                xlib.XSync(window._x_display, False)
