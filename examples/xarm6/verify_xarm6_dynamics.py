"""
xArm 6 Dynamics Verification Script for Genesis Simulation.
Tests: model parameter readback, gravity, static torques, PD step response,
       energy dissipation, mass matrix plausibility.

Continues numbering from verify_xarm6.py (Tests 1-4) as Tests 5-10.

Usage:
    source ~/envs/py312/bin/activate
    python examples/xarm6/verify_xarm6_dynamics.py              # headless (URDF default)
    python examples/xarm6/verify_xarm6_dynamics.py -v            # with viewer
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.paths import xarm6_urdf

XARM6_URDF_PATH = xarm6_urdf()

JOINT_NAMES = (
    "joint1", "joint2", "joint3",
    "joint4", "joint5", "joint6",
)
EE_LINK_NAME = "link6"

# URDF ground-truth values
URDF_LINK_MASSES = {
    "link_base": 2.7,
    "link1": 2.3814,
    "link2": 2.2675,
    "link3": 1.875,
    "link4": 1.3192,
    "link5": 1.33854,
    "link6": 0.17,
}
URDF_TOTAL_MASS = sum(URDF_LINK_MASSES.values())  # ~12.051 kg
URDF_JOINT_DAMPING = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
URDF_JOINT_EFFORT = [50.0, 50.0, 32.0, 32.0, 32.0, 20.0]

PD_KP = [3000, 3000, 2000, 2000, 1000, 1000]
PD_KV = [300, 300, 200, 200, 100, 100]
FORCE_LOWER = [-50, -50, -32, -32, -32, -20]
FORCE_UPPER = [50, 50, 32, 32, 32, 20]

SIM_DT = 0.01


# ---------------------------------------------------------------------------
# Utilities (same as verify_xarm6.py)
# ---------------------------------------------------------------------------
def resolve_entity_name(entity, requested_name: str, kind: str) -> str:
    available = {item.name for item in entity.joints} if kind == "joint" else {item.name for item in entity.links}
    if requested_name in available:
        return requested_name
    fallback = requested_name.split("/")[-1]
    if fallback in available:
        return fallback
    raise KeyError(f"{kind.capitalize()} name not found: {requested_name}. Available: {sorted(available)}")


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------
def build_scene(args):
    """Create scene, load robot, return (xarm6, ee_link, dof_idx, scene)."""
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            refresh_rate=60,
        ),
        sim_options=gs.options.SimOptions(dt=SIM_DT),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    robot_model = Path(args.robot_model).resolve()
    xarm6 = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_model), pos=(0, 0, 0), fixed=True,
        ),
    )
    assert xarm6.n_dofs == 6, f"Expected 6 DOFs, got {xarm6.n_dofs}"

    scene.build()

    # Resolve joint indices
    available_joints = {j.name: j for j in xarm6.joints}
    dof_idx = []
    for name in JOINT_NAMES:
        resolved = resolve_entity_name(xarm6, name, "joint")
        dof_idx.append(available_joints[resolved].dofs_idx_local[0])

    ee_link = xarm6.get_link(resolve_entity_name(xarm6, EE_LINK_NAME, "link"))

    print(f"Loaded robot model: {robot_model}")
    print(f"DOFs: {xarm6.n_dofs}, Links: {xarm6.n_links}")
    return xarm6, ee_link, dof_idx, scene


def set_pd_gains(xarm6, dof_idx):
    """Apply standard PD gains and force limits."""
    xarm6.set_dofs_kp(np.array(PD_KP, dtype=np.float32), dof_idx)
    xarm6.set_dofs_kv(np.array(PD_KV, dtype=np.float32), dof_idx)
    xarm6.set_dofs_force_range(
        np.array(FORCE_LOWER, dtype=np.float32),
        np.array(FORCE_UPPER, dtype=np.float32),
        dof_idx,
    )


# ---------------------------------------------------------------------------
# Test 5: Model Parameter Readback
# ---------------------------------------------------------------------------
def test_model_parameters(xarm6, dof_idx):
    print("\n--- Test 5: Model Parameter Readback ---")
    passed = True

    # 5a: Total mass
    total_mass = xarm6.get_mass()
    if hasattr(total_mass, "item"):
        total_mass = total_mass.item()
    mass_err = abs(total_mass - URDF_TOTAL_MASS) / URDF_TOTAL_MASS
    print(f"  Total mass: {total_mass:.4f} kg (expected: {URDF_TOTAL_MASS:.4f} kg, err: {mass_err*100:.2f}%)")
    if mass_err > 0.01:
        print(f"  [FAIL] Total mass error {mass_err*100:.2f}% > 1%")
        passed = False

    # 5b: Per-link masses
    print("  Per-link masses:")
    for link in xarm6.links:
        link_name = link.name.split("/")[-1]
        if link_name in URDF_LINK_MASSES:
            expected = URDF_LINK_MASSES[link_name]
            actual = link.get_mass()
            if hasattr(actual, "item"):
                actual = actual.item()
            err = abs(actual - expected) / expected if expected > 0 else 0
            status = "[OK]" if err <= 0.01 else "[FAIL]"
            print(f"    {link_name}: {actual:.4f} kg (expected {expected:.4f}, err {err*100:.1f}%) {status}")
            if err > 0.01:
                passed = False

    # 5c: Joint damping
    damping = xarm6.get_dofs_damping(dof_idx).cpu().numpy().flatten()
    print(f"  Joint damping: {damping}")
    for i, (act, exp) in enumerate(zip(damping, URDF_JOINT_DAMPING)):
        if abs(act - exp) > 0.01:
            print(f"  [FAIL] Joint {i+1} damping: {act} vs expected {exp}")
            passed = False

    # 5d: Friction loss (informational)
    friction = xarm6.get_dofs_frictionloss(dof_idx).cpu().numpy().flatten()
    print(f"  Joint friction loss: {friction}")

    if passed:
        print("[PASS] Model parameters match URDF definitions")
    else:
        print("[FAIL] Model parameter mismatch detected")
    return passed


# ---------------------------------------------------------------------------
# Test 6: Gravity Free-Fall Response
# ---------------------------------------------------------------------------
def test_gravity_freefall(xarm6, dof_idx, scene):
    print("\n--- Test 6: Gravity Free-Fall Response ---")

    # Reset to home
    home = np.zeros(6, dtype=np.float32)
    xarm6.set_dofs_position(home, dof_idx)
    set_pd_gains(xarm6, dof_idx)
    xarm6.control_dofs_position(home, dof_idx)
    for _ in range(200):
        scene.step()

    # Record settled position
    q_initial = xarm6.get_dofs_position(dof_idx).cpu().numpy().flatten().copy()

    # Disable controller: kp=kv=0, zero-force mode
    xarm6.set_dofs_kp(np.zeros(6, dtype=np.float32), dof_idx)
    xarm6.set_dofs_kv(np.zeros(6, dtype=np.float32), dof_idx)
    xarm6.control_dofs_force(np.zeros(6, dtype=np.float32), dof_idx)

    # Simulate 3 seconds
    for _ in range(300):
        scene.step()

    q_final = xarm6.get_dofs_position(dof_idx).cpu().numpy().flatten()
    delta_q = np.abs(q_final - q_initial)
    max_delta = delta_q.max()

    print(f"  Initial qpos: [{', '.join(f'{v:.4f}' for v in q_initial)}]")
    print(f"  Final qpos:   [{', '.join(f'{v:.4f}' for v in q_final)}]")
    print(f"  Displacement: [{', '.join(f'{v:.4f}' for v in delta_q)}]")
    print(f"  Max displacement: {max_delta:.4f} rad ({np.degrees(max_delta):.2f} deg)")

    passed = max_delta > 0.1
    if passed:
        print("[PASS] Arm collapses under gravity as expected")
    else:
        print("[FAIL] Arm did not move significantly under gravity")
    return passed


# ---------------------------------------------------------------------------
# Test 7: Static Torque / Gravity Compensation
# ---------------------------------------------------------------------------
def test_gravity_compensation_torques(xarm6, dof_idx, scene):
    print("\n--- Test 7: Static Torque / Gravity Compensation ---")

    set_pd_gains(xarm6, dof_idx)
    passed = True

    # Reset to home first (teleport OK here to establish a known state)
    home = np.zeros(6, dtype=np.float32)
    xarm6.set_dofs_position(home, dof_idx)
    xarm6.control_dofs_position(home, dof_idx)
    for _ in range(300):
        scene.step()

    test_configs = [
        ("home (upright)", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("arm extended",   np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("arm sideways",   np.array([1.57, -0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)),
    ]

    for name, target_q in test_configs:
        # Do NOT teleport: let PD controller drive the arm to the target.
        # At steady state, PD force = gravity torque (since the arm settles
        # with a small offset where kp * error = gravity_load).
        xarm6.control_dofs_position(target_q, dof_idx)
        for _ in range(500):
            scene.step()

        control_force = xarm6.get_dofs_control_force(dof_idx).cpu().numpy().flatten()
        internal_force = xarm6.get_dofs_force(dof_idx).cpu().numpy().flatten()
        qpos = xarm6.get_dofs_position(dof_idx).cpu().numpy().flatten()
        qvel = xarm6.get_dofs_velocity(dof_idx).cpu().numpy().flatten()

        pos_error = np.abs(qpos - target_q).max()
        vel_mag = np.abs(qvel).max()

        print(f"\n  [{name}]")
        print(f"    Target qpos:    [{', '.join(f'{v:8.4f}' for v in target_q)}]")
        print(f"    Actual qpos:    [{', '.join(f'{v:8.4f}' for v in qpos)}]")
        print(f"    Position error: {pos_error:.6f} rad")
        print(f"    Velocity:       max |v|={vel_mag:.6f} rad/s")
        print(f"    Control force:  [{', '.join(f'{v:8.3f}' for v in control_force)}] Nm")
        print(f"    Internal force: [{', '.join(f'{v:8.3f}' for v in internal_force)}] Nm")

        # a) Settled check
        if pos_error > 0.05:
            print(f"    [FAIL] Position error {pos_error:.4f} > 0.05 rad")
            passed = False

        if vel_mag > 0.01:
            print(f"    [WARN] Residual velocity: {vel_mag:.6f} rad/s")

        # b) All configs need gravity compensation (arm has mass above joints)
        max_ctrl = np.abs(control_force).max()
        print(f"    Max |ctrl force|: {max_ctrl:.3f} Nm")
        if max_ctrl < 0.5:
            print(f"    [FAIL] Control force too small ({max_ctrl:.4f} Nm) - no gravity compensation")
            passed = False

        # c) Forces within effort limits
        for i, (cf, limit) in enumerate(zip(control_force, URDF_JOINT_EFFORT)):
            if abs(cf) > limit * 1.05:
                print(f"    [FAIL] Joint {i+1} force {cf:.2f} Nm exceeds limit {limit} Nm")
                passed = False

        # d) For non-home configs, at least one joint should have significant
        #    gravity compensation (>3 Nm for a ~12 kg arm)
        if name != "home (upright)" and max_ctrl < 3.0:
            print(f"    [FAIL] Max control force {max_ctrl:.2f} Nm too small for non-upright config")
            passed = False

    if passed:
        print("\n[PASS] Gravity compensation torques are physically plausible")
    else:
        print("\n[FAIL] Gravity compensation torque check failed")
    return passed


# ---------------------------------------------------------------------------
# Test 8: PD Step Response Quality
# ---------------------------------------------------------------------------
def test_pd_step_response(xarm6, dof_idx, scene):
    print("\n--- Test 8: PD Step Response Quality ---")

    set_pd_gains(xarm6, dof_idx)

    # Settle at home first
    home = np.zeros(6, dtype=np.float32)
    xarm6.set_dofs_position(home, dof_idx)
    xarm6.control_dofs_position(home, dof_idx)
    for _ in range(300):
        scene.step()

    step_targets = [
        ("small step",  np.array([0.3, -0.2, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)),
        ("large step",  np.array([1.0, -0.8, -0.15, 0.5, -0.3, 0.2], dtype=np.float32)),
        ("return home", np.zeros(6, dtype=np.float32)),
    ]

    passed = True
    n_steps = 500

    for name, target in step_targets:
        # Record initial position
        q_start = xarm6.get_dofs_position(dof_idx).cpu().numpy().flatten().copy()

        xarm6.control_dofs_position(target, dof_idx)

        trajectory = np.zeros((n_steps, 6))
        for t in range(n_steps):
            scene.step()
            trajectory[t] = xarm6.get_dofs_position(dof_idx).cpu().numpy().flatten()

        print(f"\n  [{name}] target=[{', '.join(f'{v:.3f}' for v in target)}]")

        for j in range(6):
            traj_j = trajectory[:, j]
            target_j = target[j]
            start_j = q_start[j]

            # Steady-state error (last 50 steps)
            ss_error = np.abs(traj_j[-50:] - target_j).mean()

            # Overshoot
            step_size = abs(target_j - start_j)
            if step_size > 0.01:
                if target_j > start_j:
                    overshoot_raw = traj_j.max() - target_j
                else:
                    overshoot_raw = target_j - traj_j.min()
                overshoot_pct = max(0, overshoot_raw / step_size * 100)
            else:
                overshoot_pct = 0.0

            # Settling time (5% band)
            if step_size > 0.01:
                threshold = 0.05 * step_size
                settled_mask = np.abs(traj_j - target_j) < threshold
                if settled_mask.all():
                    settling_time = 0.0
                elif settled_mask.any():
                    last_unsettled = np.where(~settled_mask)[0][-1]
                    settling_time = (last_unsettled + 1) * SIM_DT
                else:
                    settling_time = n_steps * SIM_DT
            else:
                settling_time = 0.0

            status = "[OK]"
            if ss_error > 0.05:
                status = "[FAIL]"
                passed = False
            elif overshoot_pct > 30:
                status = "[WARN]"
            elif settling_time > 3.0:
                status = "[WARN]"

            print(f"    J{j+1}: ss_err={ss_error:.4f}rad  overshoot={overshoot_pct:.1f}%  settle={settling_time:.2f}s  {status}")

    if passed:
        print("\n[PASS] PD step response quality acceptable")
    else:
        print("\n[FAIL] PD step response has issues")
    return passed


# ---------------------------------------------------------------------------
# Test 9: Energy Dissipation (Damped Free Swing)
# ---------------------------------------------------------------------------
def compute_kinetic_energy(xarm6, dof_idx):
    """Compute KE = 0.5 * qdot^T * M * qdot."""
    M = xarm6.get_mass_mat()
    qdot = xarm6.get_dofs_velocity().cpu().numpy().flatten()
    M_np = M.cpu().numpy()
    if M_np.ndim == 3:
        M_np = M_np[0]
    return 0.5 * qdot @ M_np @ qdot


def compute_potential_energy(xarm6):
    """Compute PE = sum(m_i * g * z_i) using link CoM heights."""
    g = 9.81
    pe = 0.0
    for link in xarm6.links:
        link_name = link.name.split("/")[-1]
        if link_name in URDF_LINK_MASSES:
            mass = URDF_LINK_MASSES[link_name]
            pos = link.get_pos()
            z = pos[2].item() if pos.dim() == 1 else pos[0, 2].item()
            pe += mass * g * z
    return pe


def test_energy_dissipation(xarm6, dof_idx, scene):
    print("\n--- Test 9: Energy Dissipation (Damped Free Swing) ---")

    # Start from a non-equilibrium pose with PD control to stabilize
    set_pd_gains(xarm6, dof_idx)
    init_qpos = np.array([0.0, -0.8, 0.3, 0.0, 0.5, 0.0], dtype=np.float32)
    xarm6.set_dofs_position(init_qpos, dof_idx)
    xarm6.control_dofs_position(init_qpos, dof_idx)
    for _ in range(300):
        scene.step()

    # Release: disable controller
    xarm6.set_dofs_kp(np.zeros(6, dtype=np.float32), dof_idx)
    xarm6.set_dofs_kv(np.zeros(6, dtype=np.float32), dof_idx)
    xarm6.control_dofs_force(np.zeros(6, dtype=np.float32), dof_idx)

    n_steps = 500
    energies = np.zeros(n_steps)
    ke_arr = np.zeros(n_steps)
    pe_arr = np.zeros(n_steps)

    for t in range(n_steps):
        scene.step()
        ke = compute_kinetic_energy(xarm6, dof_idx)
        pe = compute_potential_energy(xarm6)
        ke_arr[t] = ke
        pe_arr[t] = pe
        energies[t] = ke + pe

    print(f"  Initial: KE={ke_arr[0]:.4f} J, PE={pe_arr[0]:.4f} J, Total={energies[0]:.4f} J")
    print(f"  Final:   KE={ke_arr[-1]:.4f} J, PE={pe_arr[-1]:.4f} J, Total={energies[-1]:.4f} J")

    total_dissipated = energies[0] - energies[-1]
    energy_increases = np.diff(energies)
    max_increase = energy_increases.max()

    print(f"  Total dissipated: {total_dissipated:.4f} J")
    print(f"  Max single-step energy increase: {max_increase:.6f} J")

    tolerance = max(0.01 * abs(energies[0]), 0.01)
    passed = True

    if max_increase > tolerance:
        print(f"  [FAIL] Energy increased by {max_increase:.6f} J (tolerance: {tolerance:.6f} J)")
        passed = False

    if total_dissipated < -tolerance:
        print(f"  [FAIL] Net energy increased ({total_dissipated:.4f} J)")
        passed = False

    if passed:
        print("[PASS] Energy is monotonically dissipated (within tolerance)")
    else:
        print("[FAIL] Energy conservation/dissipation violated")
    return passed


# ---------------------------------------------------------------------------
# Test 10: Mass Matrix Plausibility
# ---------------------------------------------------------------------------
def test_mass_matrix(xarm6, dof_idx, scene):
    print("\n--- Test 10: Mass Matrix Plausibility ---")

    set_pd_gains(xarm6, dof_idx)
    test_qpos = np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
    xarm6.set_dofs_position(test_qpos, dof_idx)
    xarm6.control_dofs_position(test_qpos, dof_idx)
    for _ in range(200):
        scene.step()

    M = xarm6.get_mass_mat()
    M_np = M.cpu().numpy()
    if M_np.ndim == 3:
        M_np = M_np[0]

    print(f"  Mass matrix shape: {M_np.shape}")
    print(f"  Mass matrix:\n{np.array2string(M_np, precision=6, suppress_small=True)}")

    passed = True

    # a) Symmetry
    asym = np.abs(M_np - M_np.T).max()
    print(f"  Asymmetry (max |M - M^T|): {asym:.8f}")
    if asym > 1e-5:
        print(f"  [FAIL] Mass matrix not symmetric")
        passed = False

    # b) Positive definite
    eigenvalues = np.linalg.eigvalsh(M_np)
    print(f"  Eigenvalues: [{', '.join(f'{v:.6f}' for v in eigenvalues)}]")
    if eigenvalues.min() <= 0:
        print(f"  [FAIL] Mass matrix not positive definite (min eigenvalue={eigenvalues.min():.8f})")
        passed = False

    # c) Diagonal elements
    diag = np.diag(M_np)
    print(f"  Diagonal: [{', '.join(f'{v:.6f}' for v in diag)}]")
    for i, d in enumerate(diag):
        if d <= 0:
            print(f"  [FAIL] M[{i},{i}] = {d:.6f} <= 0")
            passed = False
        elif d > 100:
            print(f"  [WARN] M[{i},{i}] = {d:.6f} seems too large")

    # d) Monotonicity (soft check)
    if diag[0] < diag[-1]:
        print(f"  [WARN] M[0,0]={diag[0]:.4f} < M[5,5]={diag[-1]:.4f} -- unexpected for serial arm")

    if passed:
        print("[PASS] Mass matrix is physically plausible")
    else:
        print("[FAIL] Mass matrix check failed")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="xArm 6 Dynamics Verification")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument(
        "--robot-model", type=str, default=None,
        help="Robot URDF model path. Default: xarm6_1305.urdf",
    )
    args = parser.parse_args()

    if args.robot_model is None:
        args.robot_model = XARM6_URDF_PATH

    xarm6, ee_link, dof_idx, scene = build_scene(args)

    results = {}
    results["Test 5:  Model Parameters"]         = test_model_parameters(xarm6, dof_idx)
    results["Test 6:  Gravity Freefall"]          = test_gravity_freefall(xarm6, dof_idx, scene)
    results["Test 7:  Gravity Comp Torques"]      = test_gravity_compensation_torques(xarm6, dof_idx, scene)
    results["Test 8:  PD Step Response"]          = test_pd_step_response(xarm6, dof_idx, scene)
    results["Test 9:  Energy Dissipation"]        = test_energy_dissipation(xarm6, dof_idx, scene)
    results["Test 10: Mass Matrix"]               = test_mass_matrix(xarm6, dof_idx, scene)

    # Summary
    print("\n" + "=" * 60)
    print("DYNAMICS VALIDATION SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\nAll dynamics tests PASSED!")
    else:
        print("\nSome dynamics tests FAILED!")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
