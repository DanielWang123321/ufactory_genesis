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
    import genesis as gs

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

    Diagnostic helper only (not part of L1/L2/L3 gating). Used to check whether a
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


def format_self_contacts(robot, contacts: Sequence[dict[str, Any]]) -> str:
    """Render ``capture_self_contacts`` output with link names for logging."""
    if not contacts:
        return "none"
    names = {link.idx: link.name.split("/")[-1] for link in robot.links}
    parts = []
    for c in contacts:
        a = names.get(c["link_a"], str(c["link_a"]))
        b = names.get(c["link_b"], str(c["link_b"]))
        parts.append(f"{a}<->{b}:{c['force_n']:.3f}N")
    return ", ".join(parts)


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
    anomalies (not part of L1/L2/L3 gating).
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


def genesis_gravity_torque_at_q(
    robot,
    scene,
    dof_idx: Sequence[int],
    target_q: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    settle_steps: int = SETTLE_STEPS,
    pos_tol: float = POS_ERR_TOL,
    vel_tol: float = VEL_TOL,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Backward-compatible alias returning Genesis PD hold torque.

    The name is intentionally deprecated: the returned torque is controller
    output from ``get_dofs_control_force()``, not a direct inverse-dynamics
    gravity vector.
    """
    sample = genesis_pd_hold_torque_at_q(
        robot,
        scene,
        dof_idx,
        target_q,
        runtime_profile=runtime_profile,
        settle_steps=settle_steps,
        pos_tol=pos_tol,
        vel_tol=vel_tol,
    )
    return sample.q_actual, sample.pd_hold_tau, sample.settled


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


def compute_ee_xyz_table_from_sim(
    robot,
    scene,
    ee_link,
    dof_idx,
    configs: Sequence[tuple[str, np.ndarray]],
) -> dict[str, tuple[float, float, float]]:
    table: dict[str, tuple[float, float, float]] = {}
    for name, q in configs:
        table[name] = genesis_ee_xyz_mm_at_q(robot, scene, ee_link, dof_idx, q)
    return table


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


# ---------------------------------------------------------------------------
# Physics plausibility checks (de-hardcoded; gains/effort from robot_params,
# masses from robot.get_mass / link.get_mass). Each returns (passed, lines).
# ---------------------------------------------------------------------------


def _home_q(runtime: RobotRuntimeProfile) -> np.ndarray:
    return np.asarray(runtime.arm.home_qpos, dtype=np.float32)


def _scalar(v) -> float:
    if hasattr(v, "item"):
        return float(v.item())
    return float(np.asarray(v).reshape(-1)[0])


def _mass_float(v) -> float:
    arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    return float(arr.reshape(-1)[0]) if arr.size == 1 else float(arr.sum())


def test_mass_parameters(robot, dof_idx, runtime_profile: RobotRuntimeProfile) -> tuple[bool, list[str]]:
    """Test 5: model-parameter readback. No hardcoded mass table; uses the
    internal consistency between per-link and total mass reported by Genesis."""
    lines: list[str] = ["--- Test 5: Model Parameter Readback ---"]
    passed = True
    total_mass = _mass_float(robot.get_mass())
    per_link = []
    for link in robot.links:
        m = _mass_float(link.get_mass())
        if m > 0:
            per_link.append((link.name.split("/")[-1], m))
    em01 = sum(m for _, m in per_link)
    cons_err = abs(total_mass - em01) / max(em01, 1e-9) if em01 else 1.0
    lines.append(f"  Total mass (get_mass): {total_mass:.4f} kg; sum per link: {em01:.4f} kg; consistency err {cons_err*100:.2f}%")
    if cons_err > 0.01:
        lines.append(f"  [FAIL] Total/per-link mass inconsistency {cons_err*100:.2f}% > 1%")
        passed = False
    for name, m in per_link:
        lines.append(f"    {name}: {m:.4f} kg")
    damping = _to_np(robot.get_dofs_damping(dof_idx)).flatten()
    frictionloss = _to_np(robot.get_dofs_frictionloss(dof_idx)).flatten()
    lines.append(f"  Joint damping: {np.round(damping, 4).tolist()}  frictionloss: {np.round(frictionloss, 4).tolist()}")
    if passed:
        lines.append("[PASS] Model parameters internally consistent")
    else:
        lines.append("[FAIL] Model parameter inconsistency")
    return passed, lines


def test_gravity_freefall(robot, scene, dof_idx, runtime_profile: RobotRuntimeProfile) -> tuple[bool, list[str]]:
    """Test 6: with controller disabled (kp=kv=0, zero force) the arm must
    collapse under gravity. Gains/info come from runtime_profile, not literals."""
    lines: list[str] = ["--- Test 6: Gravity Free-Fall Response ---"]
    home = _home_q(runtime_profile)
    n = len(dof_idx)
    robot.set_dofs_position(home, dof_idx)
    set_pd_gains(robot, dof_idx, runtime_profile)
    robot.control_dofs_position(home, dof_idx)
    for _ in range(200):
        scene.step()
    q_initial = _to_np(robot.get_dofs_position(dof_idx)).flatten().copy()
    robot.set_dofs_kp(np.zeros(n, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.zeros(n, dtype=np.float32), dof_idx)
    robot.control_dofs_force(np.zeros(n, dtype=np.float32), dof_idx)
    for _ in range(300):
        scene.step()
    q_final = _to_np(robot.get_dofs_position(dof_idx)).flatten()
    max_delta = float(np.abs(q_final - q_initial).max())
    lines.append(f"  Max displacement: {max_delta:.4f} rad ({np.degrees(max_delta):.2f} deg)")
    passed = max_delta > 0.1
    lines.append("[PASS] Arm collapses under gravity" if passed else "[FAIL] Arm did not move significantly")
    return passed, lines


def _link_potential_energy(robot, g: float = 9.81) -> float:
    pe = 0.0
    for link in robot.links:
        m = _mass_float(link.get_mass())
        if m <= 0:
            continue
        pos = link.get_pos()
        z = pos[2].item() if pos.dim() == 1 else pos[0, 2].item()
        pe += m * g * z
    return pe


def _link_kinetic_energy(robot, dof_idx) -> float:
    M = _to_np(robot.get_mass_mat())
    if M.ndim == 3:
        M = M[0]
    qdot = _to_np(robot.get_dofs_velocity(dof_idx)).flatten()
    return 0.5 * float(qdot @ M @ qdot)


def test_energy_dissipation(robot, scene, dof_idx, runtime_profile: RobotRuntimeProfile) -> tuple[bool, list[str]]:
    """Test 9: released damped swing must not gain energy. PE uses per-link
    mass * CoM height from Genesis (no hardcoded URDF_LINK_MASSES)."""
    lines: list[str] = ["--- Test 9: Energy Dissipation (Damped Free Swing) ---"]
    set_pd_gains(robot, dof_idx, runtime_profile)
    n = len(dof_idx)
    init_q = np.zeros(n, dtype=np.float32)
    init_q[1] = -0.8
    init_q[2] = 0.3
    init_q[4] = 0.5
    robot.set_dofs_position(init_q, dof_idx)
    robot.control_dofs_position(init_q, dof_idx)
    for _ in range(300):
        scene.step()
    robot.set_dofs_kp(np.zeros(n, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.zeros(n, dtype=np.float32), dof_idx)
    robot.control_dofs_force(np.zeros(n, dtype=np.float32), dof_idx)
    n_steps = 500
    energies = np.zeros(n_steps)
    for t in range(n_steps):
        scene.step()
        energies[t] = _link_kinetic_energy(robot, dof_idx) + _link_potential_energy(robot)
    total_dissipated = float(energies[0] - energies[-1])
    max_increase = float(np.diff(energies).max()) if n_steps > 1 else 0.0
    lines.append(f"  Initial total E={energies[0]:.4f} J  Final={energies[-1]:.4f} J")
    lines.append(f"  Total dissipated: {total_dissipated:.4f} J  Max single-step increase: {max_increase:.6f} J")
    tolerance = max(0.01 * abs(energies[0]), 0.01)
    passed = max_increase <= tolerance and total_dissipated >= -tolerance
    lines.append("[PASS] Energy dissipated within tolerance" if passed else "[FAIL] Energy conservation/dissipation violated")
    return passed, lines


def test_mass_matrix_plausibility(robot, scene, dof_idx, runtime_profile: RobotRuntimeProfile) -> tuple[bool, list[str]]:
    """Test 10: mass matrix symmetry, positive-definiteness, finite diagonal."""
    lines: list[str] = ["--- Test 10: Mass Matrix Plausibility ---"]
    set_pd_gains(robot, dof_idx, runtime_profile)
    n = len(dof_idx)
    test_q = np.zeros(n, dtype=np.float32)
    test_q[1] = -0.5
    test_q[4] = 0.5
    robot.set_dofs_position(test_q, dof_idx)
    robot.control_dofs_position(test_q, dof_idx)
    for _ in range(200):
        scene.step()
    M = _to_np(robot.get_mass_mat())
    if M.ndim == 3:
        M = M[0]
    passed = True
    asym = float(np.abs(M - M.T).max())
    lines.append(f"  Asymmetry |M-M^T|={asym:.8f}")
    if asym > 1e-5:
        lines.append(f"  [FAIL] Mass matrix not symmetric")
        passed = False
    eig = np.linalg.eigvalsh(M)
    lines.append(f"  Eigenvalues: [{', '.join(f'{v:.6f}' for v in eig)}]")
    if float(eig.min()) <= 0:
        lines.append(f"  [FAIL] Not positive definite (min eig={eig.min():.8f})")
        passed = False
    diag = np.diag(M)
    if np.any(diag <= 0):
        lines.append(f"  [FAIL] Non-positive diagonal: {diag.round(6).tolist()}")
        passed = False
    lines.append("[PASS] Mass matrix physically plausible" if passed else "[FAIL] Mass matrix check failed")
    return passed, lines


def test_pd_step_response(robot, scene, dof_idx, runtime_profile: RobotRuntimeProfile) -> tuple[bool, list[str]]:
    """Test 8: PD step response uses profile gains/effort (no hardcoded PD_KP/KV)."""
    lines: list[str] = ["--- Test 8: PD Step Response Quality ---"]
    set_pd_gains(robot, dof_idx, runtime_profile)
    n = len(dof_idx)
    home = np.zeros(n, dtype=np.float32)
    robot.set_dofs_position(home, dof_idx)
    robot.control_dofs_position(home, dof_idx)
    for _ in range(300):
        scene.step()
    targets = [
        ("small step", np.array([0.3, -0.2, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)),
        ("large step", np.array([1.0, -0.8, -0.15, 0.5, -0.3, 0.2], dtype=np.float32)),
        ("return home", np.zeros(n, dtype=np.float32)),
    ]
    passed = True
    n_steps = 500
    sim_dt = 0.01
    for name, target in targets:
        q_start = _to_np(robot.get_dofs_position(dof_idx)).flatten().copy()
        robot.control_dofs_position(target, dof_idx)
        traj = np.zeros((n_steps, n))
        for t in range(n_steps):
            scene.step()
            traj[t] = _to_np(robot.get_dofs_position(dof_idx)).flatten()
        lines.append(f"  [{name}] target={np.round(target, 3).tolist()}")
        for j in range(n):
            ss_err = float(np.abs(traj[-50:, j] - target[j]).mean())
            step = abs(float(target[j] - q_start[j]))
            if step > 0.01 and target[j] > q_start[j]:
                ov = max(0.0, float(traj[:, j].max()) - target[j])
            elif step > 0.01:
                ov = max(0.0, target[j] - float(traj[:, j].min()))
            else:
                ov = 0.0
            ov_pct = ov / step * 100 if step > 0.01 else 0.0
            if step > 0.01:
                thr = 0.05 * step
                settled = np.abs(traj[:, j] - target[j]) < thr
                if settled.all():
                    settle_t = 0.0
                elif settled.any():
                    settle_t = (np.where(~settled)[0][-1] + 1) * sim_dt
                else:
                    settle_t = n_steps * sim_dt
            else:
                settle_t = 0.0
            status = "[OK]"
            if ss_err > 0.05:
                status = "[FAIL]"
                passed = False
            lines.append(f"    J{j+1}: ss_err={ss_err:.4f}rad overshoot={ov_pct:.1f}% settle={settle_t:.2f}s {status}")
    lines.append("[PASS] PD step response acceptable" if passed else "[FAIL] PD step response has issues")
    return passed, lines


def test_gravity_compensation_torques(robot, scene, dof_idx, runtime_profile: RobotRuntimeProfile) -> tuple[bool, list[str]]:
    """Test 7: gravity-compensation torques. Effort limits from runtime_profile
    (no hardcoded URDF_JOINT_EFFORT)."""
    lines: list[str] = ["--- Test 7: Static Torque / Gravity Compensation ---"]
    set_pd_gains(robot, dof_idx, runtime_profile)
    n = len(dof_idx)
    home = np.zeros(n, dtype=np.float32)
    robot.set_dofs_position(home, dof_idx)
    robot.control_dofs_position(home, dof_idx)
    for _ in range(300):
        scene.step()
    configs = [
        ("home (upright)", np.zeros(n, dtype=np.float32)),
        ("arm extended", np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("arm sideways", np.array([1.57, -0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)),
    ]
    effort = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    passed = True
    for name, target in configs:
        robot.control_dofs_position(target, dof_idx)
        for _ in range(500):
            scene.step()
        cf = _to_np(robot.get_dofs_control_force(dof_idx)).flatten()
        qpos = _to_np(robot.get_dofs_position(dof_idx)).flatten()
        qvel = _to_np(robot.get_dofs_velocity(dof_idx)).flatten()
        pos_err = float(np.abs(qpos - target).max())
        vel_mag = float(np.abs(qvel).max())
        max_ctrl = float(np.abs(cf).max())
        lines.append(f"  [{name}] pos_err={pos_err:.4f} vel={vel_mag:.5f} max|ctrl|={max_ctrl:.3f} Nm")
        if pos_err > 0.05:
            lines.append(f"    [FAIL] Position error {pos_err:.4f} > 0.05 rad")
            passed = False
        if max_ctrl < 0.5:
            lines.append(f"    [FAIL] Control force too small ({max_ctrl:.4f} Nm) - no gravity compensation")
            passed = False
        for i in range(min(len(cf), len(effort))):
            if abs(float(cf[i])) > float(effort[i]) * 1.05:
                lines.append(f"    [FAIL] Joint {i+1} force {cf[i]:.2f} exceeds limit {effort[i]:.2f} Nm")
                passed = False
        if name != "home (upright)" and max_ctrl < 3.0:
            lines.append(f"    [FAIL] Max control force {max_ctrl:.2f} Nm too small for non-upright config")
            passed = False
    lines.append("[PASS] Gravity compensation torques plausible" if passed else "[FAIL] Gravity compensation torque check failed")
    return passed, lines