"""Genesis probe functions: scene build + controlled PD-hold sampling.

PD hold torque: in Genesis simulation, joint torques computed by the PD
controller to maintain target joint angles.

All Genesis-touching code lives here so that the analysis/reference/report layers
stay simulator-free and unit-testable in isolation. Probe functions return
plain-data samples (see :class:`ufactory.dynamics.report.GenesisDynamicsSample`)
rather than GS tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ufactory.dynamics.report import (
    POS_ERR_TOL,
    SATURATION_MARGIN,
    SETTLE_STEPS,
    SIM_DT,
    SIM_SUBSTEPS,
    VEL_TOL,
    GenesisDynamicsSample,
    _effort_limits,
)
from ufactory.dynamics.poses import _XARM6_RUNTIME
from ufactory.simulation import make_rigid_options
from ufactory.simulation.compat import require_genesis_runtime

if TYPE_CHECKING:
    from ufactory.robots.runtime import RobotRuntimeProfile


def _to_np(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


def resolve_entity_name(entity, requested_name: str, kind: str) -> str:
    available = {item.name for item in entity.joints} if kind == "joint" else {item.name for item in entity.links}
    if requested_name in available:
        return requested_name
    fallback = requested_name.split("/")[-1]
    if fallback in available:
        return fallback
    raise KeyError(f"{kind.capitalize()} name not found: {requested_name}. Available: {sorted(available)}")


def build_genesis_scene(
    urdf_path: str,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    show_viewer: bool = False,
    sim_dt: float = SIM_DT,
    sim_substeps: int = SIM_SUBSTEPS,
    backend: str = "gpu",
    zero_joint_frictionloss: bool = False,
    default_armature: float | None = 0.1,
):
    """Create a minimal fixed-base Genesis scene for dynamics checks.

    ``default_armature`` is forwarded to ``gs.morphs.URDF`` and matches Genesis's
    own default (0.1 kg*m^2 reflected rotor inertia applied uniformly to every
    joint). Override to ``None`` (disable) or a smaller value to diagnose
    simulation-side torque anomalies on light distal joints -- see
    ``uf_dynamics.md`` section 4.8 (Lite6 wrist anomaly) -- without touching the
    production default used by the CLI entry points.
    """
    gs = require_genesis_runtime()

    runtime = runtime_profile or _XARM6_RUNTIME
    gs_backend = gs.gpu if backend == "gpu" else gs.cpu
    gs.init(backend=gs_backend)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            refresh_rate=60,
        ),
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps),
        rigid_options=make_rigid_options(gs),
        show_viewer=show_viewer,
    )
    # Infinite ground plane primitive; avoids a cwd-relative plane.urdf file
    # dependency and keeps the dynamics scene headless/portable.
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(Path(urdf_path).resolve()),
            pos=(0, 0, 0),
            fixed=True,
            default_armature=default_armature,
        )
    )
    scene.build()

    available_joints = {j.name: j for j in robot.joints}
    dof_idx = [
        available_joints[resolve_entity_name(robot, name, "joint")].dofs_idx_local[0]
        for name in runtime.arm.joint_names
    ]
    ee_link = robot.get_link(resolve_entity_name(robot, runtime.arm.ee_link, "link"))
    if zero_joint_frictionloss:
        robot.set_dofs_frictionloss(np.zeros(len(dof_idx), dtype=np.float32), dof_idx)
    return scene, robot, ee_link, dof_idx


def read_joint_frictionloss(robot, dof_idx: Sequence[int]) -> np.ndarray:
    """Read per-joint frictionloss (Nm) applied on the robot DOFs."""
    return _to_np(robot.get_dofs_frictionloss(dof_idx)).flatten().astype(np.float64)


def capture_self_contacts(robot, *, min_force_n: float = 1e-6) -> list[dict[str, Any]]:
    """Return active self-contact pairs (link indices + contact force magnitude, N).

    Diagnostic helper only (not part of L1/L2/L3 validation layers). Used to check whether a
    coarse/overlapping collision mesh is injecting a spurious contact force that
    inflates ``pd_hold_tau`` at a specific configuration -- e.g. the Lite6 wrist
    (J4-J5-J6) anomaly documented in ``uf_dynamics.md`` sections 4.7/4.8. Returns
    an empty list if the backend does not support ``get_contacts`` or no
    self-contact is currently active.
    """
    try:
        contacts = robot.get_contacts(with_entity=robot, exclude_self_contact=False)
    except Exception:
        return []
    link_a = _to_np(contacts.get("link_a", np.zeros(0)))
    link_b = _to_np(contacts.get("link_b", np.zeros(0)))
    force_a = _to_np(contacts.get("force_a", np.zeros(0)))
    n = int(link_a.shape[0]) if link_a.ndim else 0
    out: list[dict[str, Any]] = []
    for i in range(n):
        force_vec = force_a[i] if force_a.ndim > 1 else np.asarray([force_a[i] if force_a.ndim else force_a])
        force_mag = float(np.linalg.norm(np.asarray(force_vec, dtype=np.float64)))
        if force_mag < min_force_n:
            continue
        out.append(
            {
                "link_a": int(link_a[i]),
                "link_b": int(link_b[i]),
                "force_n": force_mag,
            }
        )
    return out


def set_pd_gains(
    robot,
    dof_idx: Sequence[int],
    runtime_profile: RobotRuntimeProfile | None = None,
) -> None:
    """Apply profile-specific arm PD gains and force limits."""
    runtime = runtime_profile or _XARM6_RUNTIME
    robot.set_dofs_kp(np.asarray(runtime.arm.kp, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.asarray(runtime.arm.kv, dtype=np.float32), dof_idx)
    robot.set_dofs_force_range(
        np.asarray(runtime.arm.force_lower, dtype=np.float32),
        np.asarray(runtime.arm.force_upper, dtype=np.float32),
        dof_idx,
    )


def genesis_pd_hold_torque_at_q(
    robot,
    scene,
    dof_idx: Sequence[int],
    target_q: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    settle_steps: int = SETTLE_STEPS,
    pos_tol: float = POS_ERR_TOL,
    vel_tol: float = VEL_TOL,
    effort_limits: np.ndarray | None = None,
    snap_home_first: bool = False,
    snap_steps: int = 16,
    capture_contacts: bool = False,
) -> GenesisDynamicsSample:
    """Hold ``target_q`` with Genesis PD and return explicitly named physics quantities.

    PD hold torque: in Genesis simulation, joint torques computed by the PD
    controller to maintain target joint angles (``get_dofs_control_force``).

    ``capture_contacts=True`` additionally records active self-contact pairs at the
    settled state (see :func:`capture_self_contacts`); off by default since it costs
    an extra backend call and is only needed for diagnosing simulation-side torque
    anomalies (not part of L1/L2/L3 validation layers).
    """
    runtime = runtime_profile or _XARM6_RUNTIME
    set_pd_gains(robot, dof_idx, runtime)
    target = np.asarray(target_q, dtype=np.float32)
    if snap_home_first:
        home = np.zeros(len(dof_idx), dtype=np.float32)
        robot.set_dofs_position(home, dof_idx)
        robot.control_dofs_position(home, dof_idx)
        for _ in range(snap_steps):
            scene.step()
    robot.control_dofs_position(target, dof_idx)
    for _ in range(settle_steps):
        scene.step()

    qpos = _to_np(robot.get_dofs_position(dof_idx)).flatten().astype(np.float64)
    qvel = _to_np(robot.get_dofs_velocity(dof_idx)).flatten().astype(np.float64)
    pd_hold_tau = _to_np(robot.get_dofs_control_force(dof_idx)).flatten().astype(np.float64)
    actual_dof_force = _to_np(robot.get_dofs_force(dof_idx)).flatten().astype(np.float64)
    mass_matrix = _to_np(robot.get_mass_mat()).astype(np.float64)
    if mass_matrix.ndim == 3:
        mass_matrix = mass_matrix[0]

    armature = None
    frictionloss = None
    try:
        armature = _to_np(robot.get_dofs_armature(dof_idx)).flatten().astype(np.float64)
    except Exception:
        pass
    try:
        frictionloss = _to_np(robot.get_dofs_frictionloss(dof_idx)).flatten().astype(np.float64)
    except Exception:
        pass

    pos_err = float(np.abs(qpos - target).max())
    vel_mag = float(np.abs(qvel).max())
    settled = pos_err <= pos_tol and vel_mag <= vel_tol
    limits = effort_limits if effort_limits is not None else _effort_limits(runtime)
    saturated = bool(np.any(np.abs(pd_hold_tau) >= np.asarray(limits, dtype=np.float64) * SATURATION_MARGIN))
    self_contacts = capture_self_contacts(robot) if capture_contacts else None
    return GenesisDynamicsSample(
        q_actual=qpos,
        qvel=qvel,
        pd_hold_tau=pd_hold_tau,
        actual_dof_force=actual_dof_force,
        mass_matrix=mass_matrix,
        settled=settled,
        saturated=saturated,
        pos_err=pos_err,
        vel_mag=vel_mag,
        armature=armature,
        joint_frictionloss=frictionloss,
        self_contacts=self_contacts,
    )


def genesis_ee_z_mm_at_q(robot, scene, ee_link, dof_idx, q: np.ndarray, settle_steps: int = 5) -> float:
    _, _, z_mm = genesis_ee_xyz_mm_at_q(robot, scene, ee_link, dof_idx, q, settle_steps=settle_steps)
    return z_mm


def genesis_ee_xyz_mm_at_q(
    robot,
    scene,
    ee_link,
    dof_idx,
    q: np.ndarray,
    settle_steps: int = 5,
) -> tuple[float, float, float]:
    """EE (x, y, z) in mm by setting joint positions and reading link6 world position."""
    robot.set_dofs_position(np.asarray(q, dtype=np.float32), dof_idx)
    for _ in range(settle_steps):
        scene.step()
    pos = ee_link.get_pos()
    if pos.dim() == 1:
        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
    else:
        x, y, z = pos[0, 0].item(), pos[0, 1].item(), pos[0, 2].item()
    return float(x * 1000.0), float(y * 1000.0), float(z * 1000.0)


def compute_ee_z_table_from_sim(
    robot,
    scene,
    ee_link,
    dof_idx,
    configs: Sequence[tuple[str, np.ndarray]],
) -> dict[str, float]:
    table: dict[str, float] = {}
    for name, q in configs:
        table[name] = genesis_ee_z_mm_at_q(robot, scene, ee_link, dof_idx, q)
    return table


def check_genesis_path_z(
    robot,
    scene,
    ee_link,
    dof_idx,
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    z_min_mm: float,
    steps: int = 25,
) -> list[str]:
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    reasons: list[str] = []
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        z = genesis_ee_z_mm_at_q(robot, scene, ee_link, dof_idx, q, settle_steps=1)
        if z < z_min_mm:
            reasons.append(f"Genesis path step {i}: EE z {z:.2f} mm < z_min {z_min_mm:.2f} mm")
            break
    return reasons
