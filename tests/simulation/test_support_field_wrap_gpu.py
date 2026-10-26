"""SupportField azimuth lookup at theta ~= -pi stays inside the queried geom's table (Genesis 1.4.5 regression)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu

# Runs in its own process: Genesis allows one gs.init per process.
_SCRIPT = textwrap.dedent(
    """
    import json, math, os, sys, tempfile
    import numpy as np
    import trimesh
    import genesis as gs
    import quadrants as qd
    import genesis.utils.array_class as array_class
    from genesis.engine.solvers.rigid.collider import support_field
    from ufactory.simulation.compat import require_genesis_runtime

    if not hasattr(support_field, "_func_direction_cell"):
        print(json.dumps({"skip": "no SupportField._func_direction_cell"}))
        sys.exit(0)
    require_genesis_runtime(gs)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")

    tmp = tempfile.mkdtemp()
    scene = gs.Scene(show_viewer=False)
    meshes = (trimesh.creation.box(extents=(0.07, 0.07, 0.03)), trimesh.creation.icosphere(subdivisions=1, radius=0.03))
    for k, mesh in enumerate(meshes):
        path = os.path.join(tmp, f"geom{k}.obj")
        mesh.export(path)
        scene.add_entity(gs.morphs.Mesh(file=path, pos=(0.2 * k, 0.0, 0.1), fixed=True))
    scene.build()
    collider_info = scene.sim.rigid_solver.collider.collider_info

    dirs = []
    for eps in np.logspace(-18, -9, 37):
        dirs += [[-1.0, -eps, 0.1 * eps], [-1.0, 0.1 * eps, eps]]
    dirs = np.asarray(dirs) / np.linalg.norm(dirs, axis=1, keepdims=True)
    d_in = qd.ndarray(dtype=gs.qd_float, shape=dirs.shape)
    d_in.from_numpy(dirs)
    cells = qd.ndarray(dtype=gs.qd_int, shape=(len(dirs),))
    ref = qd.ndarray(dtype=gs.qd_int, shape=(len(dirs),))

    @qd.kernel
    def query(i_g: qd.i32, d_in: qd.types.ndarray(), cells: qd.types.ndarray(), ref: qd.types.ndarray(),
              collider_info: array_class.ColliderInfo):
        for k in range(d_in.shape[0]):
            d = qd.Vector([d_in[k, 0], d_in[k, 1], d_in[k, 2]], dt=gs.qd_float)
            cells[k] = support_field._func_direction_cell(i_g, d, collider_info)
            res = collider_info.support_field.support_res[None]
            band_start, band_rows = res // 4, res // 2
            i_chart = 0
            d_chart = d
            if qd.abs(d[2]) > qd.abs(d[1]):
                i_chart = 1
                d_chart = qd.Vector([d[0], -d[2], d[1]], dt=gs.qd_float)
            i = gs.qd_int(qd.math.floor((qd.atan2(d_chart[1], d_chart[0]) + math.pi) / math.pi / 2 * res))
            i = (i % res + res) % res
            j = gs.qd_int(qd.math.clamp(qd.math.floor(qd.acos(d_chart[2]) / math.pi * res) - band_start, 0, band_rows - 1))
            ref[k] = collider_info.support_field.support_cell_start[i_g] + (i_chart * res + i) * band_rows + j

    query(1, d_in, cells, ref, collider_info)
    cells, ref = cells.to_numpy(), ref.to_numpy()
    print(json.dumps({"n": len(dirs), "wrong": int((cells != ref).sum())}))
    """
)


def test_direction_cell_wraps_azimuth_near_minus_pi(tmp_path: Path) -> None:
    # Write to a file: Python 3.12 does not populate linecache for ``python -c``,
    # and ``@qd.kernel`` requires ``inspect.getsourcelines`` to locate source.
    script_path = tmp_path / "support_field_wrap_probe.py"
    script_path.write_text(_SCRIPT, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (result.stdout[-3000:], result.stderr[-3000:])
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if "skip" in report:
        pytest.skip(report["skip"])
    assert report["n"] > 0
    assert report["wrong"] == 0, report
