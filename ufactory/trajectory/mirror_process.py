"""Isolated high-fidelity Genesis mirror for real packaging execution."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any

from ufactory.trajectory.segments import Program, Segment


def _paced_start_hold(mirror: Any, hold_s: float) -> None:
    if hold_s <= 0.0:
        return
    tick_s = 1.0 / float(mirror.rate)
    steps = max(1, int(round(float(hold_s) / tick_s)))
    deadline = time.monotonic()
    for _ in range(steps):
        mirror.hold_step()
        deadline += tick_s
        sleep_s = deadline - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def _packaging_mirror_worker(
    program: Program,
    config_path: str | None,
    urdf_path: str,
    start_hold_s: float,
    commands: Any,
    ready: Any,
    stop: Any,
) -> None:
    ready_sent = False
    try:
        from ufactory.cli.packaging import _build_packaging_context
        from ufactory.config import load_runtime_config
        from ufactory.simulation import GenesisRuntimeManager
        from ufactory.trajectory.mirror_executor import KinematicCarryTracker, TrajKinematicMirror
        from ufactory.trajectory.packaging import packaging_layout
        from ufactory.visualization import start_deferred_viewer

        config = load_runtime_config(
            "xarm6",
            task="packaging_showcase",
            config_path=None if config_path is None else Path(config_path),
        )
        with GenesisRuntimeManager(config.simulation):
            ctx = _build_packaging_context(config, Path(urdf_path), show_viewer=False)
            mirror = TrajKinematicMirror(ctx, program)
            mirror.prime_to_home()
            # This process contains no hardware sender, so retain the scene's
            # full 60 Hz viewer repaint rate.
            start_deferred_viewer(ctx.scene)
            _paced_start_hold(mirror, start_hold_s)
            tracker = KinematicCarryTracker(
                mirror,
                grasp_gap_m=packaging_layout(config).grasp_gap_m,
                grasp_segment_label="grip",
                release_segment_label="release",
                approach_freeze_labels=("descend",),
            )
            print("[visual-warmup] compiling isolated kinematic mirror object updates...", flush=True)
            started = time.monotonic()
            tracker.warm_up()
            print(f"[visual-warmup] complete elapsed_s={time.monotonic() - started:.2f}", flush=True)
            ready.send(("ready", ""))
            ready_sent = True

            viewer = ctx.scene.visualizer.viewer
            finished = False
            processed = 0
            while not stop.is_set() and viewer.is_alive():
                if finished:
                    frame_started = time.monotonic()
                    tracker.hold_step()
                    sleep_s = (1.0 / 30.0) - (time.monotonic() - frame_started)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    continue
                try:
                    item = commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                if stop.is_set():
                    break
                if item is None:
                    finished = True
                    ready.send(("finished", str(processed)))
                    print("Viewer open. Press Ctrl+C to exit...", flush=True)
                    continue
                segment_idx, tick_idx = item
                tracker.on_tick(program.segments[int(segment_idx)], int(tick_idx), update_visualizer=True)
                processed += 1
    except BaseException:
        details = traceback.format_exc()
        if not ready_sent:
            try:
                ready.send(("error", details))
            except Exception:
                pass
        else:
            print(f"[visual-process] failed:\n{details}", file=sys.stderr, flush=True)
    finally:
        try:
            ready.close()
        except Exception:
            pass


class PackagingMirrorProcess:
    """Send approved trajectory ticks to a Genesis viewer in another process."""

    def __init__(
        self,
        program: Program,
        *,
        config_path: Path | None,
        urdf_path: Path,
        start_hold_s: float = 0.5,
    ) -> None:
        self._program = program
        self._segment_indices = {id(segment): idx for idx, segment in enumerate(program.segments)}
        context = mp.get_context("spawn")
        self._commands = context.Queue()
        self._ready_recv, ready_send = context.Pipe(duplex=False)
        self._stop = context.Event()
        self._process = context.Process(
            target=_packaging_mirror_worker,
            args=(
                program,
                None if config_path is None else str(config_path),
                str(urdf_path),
                float(start_hold_s),
                self._commands,
                ready_send,
                self._stop,
            ),
            name="genesis-packaging-mirror",
            daemon=True,
        )
        self._ready_send = ready_send
        self._started = False
        self._finished = False
        self._processed_frames: int | None = None

    def start(self, *, timeout_s: float = 180.0) -> None:
        print("[visual-process] starting isolated high-fidelity Genesis mirror...", flush=True)
        self._process.start()
        self._started = True
        self._ready_send.close()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._ready_recv.poll(0.1):
                status, details = self._ready_recv.recv()
                if status == "ready":
                    print("[visual-process] ready; renderer isolated from servo sender", flush=True)
                    return
                raise RuntimeError(f"isolated Genesis mirror failed to start:\n{details}")
            if not self._process.is_alive():
                raise RuntimeError(f"isolated Genesis mirror exited during startup (code={self._process.exitcode})")
        raise TimeoutError(f"isolated Genesis mirror did not become ready within {timeout_s:.0f} s")

    def on_tick(self, segment: Segment, tick_idx: int) -> None:
        """Post a tiny integer-only update; no Genesis work runs in this process."""
        segment_idx = self._segment_indices.get(id(segment))
        if segment_idx is None:
            raise ValueError("mirror received a segment outside the approved program")
        if self._process.is_alive():
            self._commands.put_nowait((segment_idx, int(tick_idx)))

    def hold_until_closed(self) -> None:
        """Tell the child execution ended, then wait for its window to close."""
        self.finish()
        if not self._started:
            return
        try:
            while self._process.is_alive():
                self._process.join(timeout=0.2)
        except KeyboardInterrupt:
            pass

    def finish(self, *, timeout_s: float = 30.0) -> int:
        """Drain posted frames and return the number processed by the child."""
        if not self._started:
            return 0
        if self._processed_frames is not None:
            return self._processed_frames
        if not self._finished:
            self._finished = True
            if self._process.is_alive():
                self._commands.put_nowait(None)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._ready_recv.poll(0.1):
                status, details = self._ready_recv.recv()
                if status == "finished":
                    processed = int(details)
                    self._processed_frames = processed
                    print(f"[visual-process] synchronized frames={processed}", flush=True)
                    return processed
                if status == "error":
                    raise RuntimeError(f"isolated Genesis mirror failed:\n{details}")
            if not self._process.is_alive():
                raise RuntimeError(
                    f"isolated Genesis mirror exited before synchronization (code={self._process.exitcode})"
                )
        raise TimeoutError(f"isolated Genesis mirror did not synchronize within {timeout_s:.0f} s")

    def close(self) -> None:
        self._stop.set()
        if self._started and self._process.is_alive():
            try:
                self._commands.put_nowait(None)
            except Exception:
                pass
            self._process.join(timeout=5.0)
        if self._started and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        try:
            self._commands.close()
            self._commands.cancel_join_thread()
        except Exception:
            pass
        try:
            self._ready_recv.close()
        except Exception:
            pass
