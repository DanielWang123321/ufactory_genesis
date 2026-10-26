"""Genesis 1.4.5 SupportField._func_direction_cell with the azimuth index wrapped on both sides.

Installed by ``ufactory.simulation.compat.patch_support_field_direction_cell`` only when the running Genesis carries
the underflow. The body matches Genesis 1.4.5 except for the restored ``if i < 0`` wrap. Importing this module
requires Genesis and Quadrants.
"""

import math

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import quadrants as qd


@qd.func
def func_direction_cell(i_g: int, d_mesh: qd.types.vector(3), collider_info: array_class.ColliderInfo):
    support_res = collider_info.support_field.support_res[None]
    band_start, band_rows = support_res // 4, support_res // 2

    i_chart = gs.qd_int(0)
    d_chart = d_mesh
    if qd.abs(d_mesh[2]) > qd.abs(d_mesh[1]):
        i_chart = 1
        d_chart = qd.Vector([d_mesh[0], -d_mesh[2], d_mesh[1]], dt=gs.qd_float)

    theta = 0.0
    phi = 0.0
    if qd.static(qd.lang.impl.get_runtime().default_fp == qd.f32):
        theta = gu.qd_atan2_fast(d_chart[1], d_chart[0])
        phi = gu.qd_acos_fast(d_chart[2])
    else:
        theta = qd.atan2(d_chart[1], d_chart[0])
        phi = qd.acos(d_chart[2])
    i = gs.qd_int(qd.math.floor((theta + math.pi) / math.pi / 2 * support_res))
    if i >= support_res:
        i = i - support_res
    if i < 0:  # rounding at theta ~= -pi gives -1; Genesis 1.4.5 dropped this wrap
        i = i + support_res
    j = gs.qd_int(qd.math.clamp(qd.math.floor(phi / math.pi * support_res) - band_start, 0, band_rows - 1))
    return collider_info.support_field.support_cell_start[i_g] + (i_chart * support_res + i) * band_rows + j
