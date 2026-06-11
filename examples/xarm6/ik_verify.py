"""
xArm 6 IK Verification: Genesis IK accuracy check

Verification flow (per test config):
  1. Pick reference joint angles q_ref.
  2. SDK FK(q_ref) → target TCP (ground truth).
  3. q_ref is the ground-truth joint solution (by construction).
  4. Genesis IK from two non-trivial initial positions:
       - IK-near: q_ref + random perturbation (±0.3 rad)
       - IK-far:  HOME (all zeros)
  5. Metrics:
       Primary (PASS/FAIL): |Genesis_FK(q_ik) − target TCP|
       Informational: q_ik vs q_ref, IK solver residual

Usage:
    source ~/envs/py312/bin/activate
    python examples/xarm6/ik_verify.py --ip 192.168.1.60
    python examples/xarm6/ik_verify.py --ip 192.168.1.60 --urdf path/to/custom.urdf
    python examples/xarm6/ik_verify.py --ip 192.168.1.60 -v
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.paths import xarm6_urdf

DEFAULT_URDF = xarm6_urdf("xarm6_xarm6_kinematics_calib1_calib.urdf")

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
EE_LINK_NAME = "link6"

HOME_Q = np.zeros(6)

# Pass criteria (applied to Genesis FK of IK result vs target TCP)
PASS_POS_MM = 1.0    # mm
PASS_RPY_DEG = 0.5   # degrees

# Perturbation magnitude for IK-near test (radians)
NEAR_PERTURB_RAD = 0.3

# Joint limits from URDF (rad) — used to generate random configs within valid range
JOINT_LIMITS = np.array([
    [-6.2832,  6.2832],   # joint1
    [-2.0590,  2.0944],   # joint2
    [-3.9270,  0.19198],  # joint3
    [-6.2832,  6.2832],   # joint4
    [-1.69297, 3.14159],  # joint5
    [-6.2832,  6.2832],   # joint6
])

# Practical range for random test generation (avoid extreme joint angles)
# Use ~60% of full range centered on zero, clamped to actual limits
JOINT_RANGE_FOR_RANDOM = np.column_stack([
    np.maximum(JOINT_LIMITS[:, 0], np.array([-2.0, -1.5, -2.0, -2.0, -1.5, -3.0])),
    np.minimum(JOINT_LIMITS[:, 1], np.array([ 2.0,  1.5,  0.15, 2.0,  1.5,  3.0])),
])

# Hand-crafted reference configs (kept for reproducibility)
MANUAL_CONFIGS = [
    ("home",     np.array([ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ])),
    ("config_A", np.array([ 0.5,  -0.3,   0.0,   0.0,   0.3,   0.0 ])),
    ("config_B", np.array([ 0.0,  -0.5,  -0.1,   0.5,   0.5,   0.0 ])),
    ("config_C", np.array([-0.3,   0.2,  -0.15,  0.3,  -0.2,   0.1 ])),
    ("config_D", np.array([ 0.8,  -0.6,  -0.15,  0.8,  -0.3,   1.0 ])),
    ("config_E", np.array([-1.0,   0.5,  -0.5,  -1.0,   1.0,  -1.5 ])),
    ("config_F", np.array([ 0.0,  -1.0,  -0.3,   0.0,   1.5,   0.0 ])),
    ("config_G", np.array([ 1.5,  -0.2,  -0.1,   0.8,  -0.8,   2.0 ])),
    ("config_H", np.array([-1.5,   0.8,  -0.8,  -1.5,   1.2,  -2.0 ])),
    ("config_I", np.array([ 0.3,  -0.6,  -0.2,   0.6,   0.8,   0.5 ])),
]

NUM_RANDOM_CONFIGS = 40  # + 10 manual = 50 total


def generate_random_configs(rng, n):
    """Generate n random joint configs within practical joint limits."""
    configs = []
    for i in range(n):
        q = np.array([
            rng.uniform(JOINT_RANGE_FOR_RANDOM[j, 0], JOINT_RANGE_FOR_RANDOM[j, 1])
            for j in range(6)
        ])
        configs.append((f"rand_{i:02d}", q))
    return configs

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_to_rpy(quat):
    """Quaternion (w,x,y,z) → (roll, pitch, yaw) rad."""
    w, x, y, z = (float(v) for v in quat)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def rpy_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) rad → quaternion (w,x,y,z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) * 180.0 / math.pi


# ---------------------------------------------------------------------------
# Genesis helpers
# ---------------------------------------------------------------------------

def build_genesis_robot(urdf_path: str, show_viewer: bool):
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            requires_jac_and_IK=True,
        )
    )
    scene.build()
    return scene, robot


def genesis_fk(robot, qpos_np: np.ndarray, ee_link_idx: int):
    """Pure-math FK → (pos_m, quat_wxyz)."""
    qpos_t = torch.tensor(qpos_np, dtype=torch.float32, device=gs.device)
    lp, lq = robot.forward_kinematics(qpos=qpos_t)
    if lp.ndim == 2:
        return lp[ee_link_idx].cpu().numpy(), lq[ee_link_idx].cpu().numpy()
    return lp[0, ee_link_idx].cpu().numpy(), lq[0, ee_link_idx].cpu().numpy()


def run_genesis_ik(robot, ee_link, target_pos_m_t, target_quat_t, init_qpos_np,
                   max_solver_iters=20, damping=0.01):
    """Run Genesis IK with return_error=True. Returns (q_ik_np, pos_err, rot_err) or None on failure."""
    init_qpos_t = torch.tensor(init_qpos_np, dtype=torch.float32, device=gs.device)
    result = robot.inverse_kinematics(
        link=ee_link,
        pos=target_pos_m_t,
        quat=target_quat_t,
        init_qpos=init_qpos_t,
        return_error=True,
        max_solver_iters=max_solver_iters,
        damping=damping,
        pos_tol=1e-4,
        rot_tol=1e-3,
    )
    if result is None:
        return None
    # return_error=True returns (qpos, error)
    qpos, error = result
    q_ik = qpos.cpu().numpy().flatten()[:6]
    err = error.cpu().numpy().flatten()
    pos_err = float(np.linalg.norm(err[:3]))   # meters
    rot_err = float(np.linalg.norm(err[3:6]))   # radians
    return q_ik, pos_err, rot_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="xArm 6 IK Verification")
    parser.add_argument("--ip", required=True, help="xArm IP (simulation mode, for FK only)")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for perturbation")
    parser.add_argument("--max-iters", type=int, default=20,
                        help="IK solver max iterations per sample (default: 20)")
    parser.add_argument("--damping", type=float, default=0.01,
                        help="IK solver damping coefficient (default: 0.01)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build test config list: manual + random
    test_configs = list(MANUAL_CONFIGS) + generate_random_configs(rng, NUM_RANDOM_CONFIGS)

    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    print("=" * 80)
    print("xArm 6 IK Verification")
    print("=" * 80)
    print(f"URDF : {urdf_path}")
    print(f"SDK  : {args.ip}  [FK only, simulation mode]")
    print(f"Tests: {len(test_configs)} configs ({len(MANUAL_CONFIGS)} manual + {NUM_RANDOM_CONFIGS} random) × 2 init = {len(test_configs)*2} tests")
    print(f"Pass : FK(q_ik) pos_err < {PASS_POS_MM} mm, rpy_err < {PASS_RPY_DEG} deg")
    print(f"IK   : max_iters={args.max_iters}, damping={args.damping}, pos_tol=1e-4, rot_tol=1e-3")
    print()

    # ---- Genesis ----
    scene, robot = build_genesis_robot(str(urdf_path), args.vis)

    joint_map = {j.name: j for j in robot.joints}
    missing = [n for n in JOINT_NAMES if n not in joint_map]
    if missing:
        raise RuntimeError(f"Joints missing: {missing}")

    link_map = {lk.name: lk for lk in robot.links}
    ee_link = link_map[EE_LINK_NAME]
    ee_link_idx = int(ee_link.idx_local)
    print(f"EE link: '{EE_LINK_NAME}' idx_local={ee_link_idx}")
    print(f"n_dofs={robot.n_dofs}, n_links={robot.n_links}")
    print()

    # ---- SDK (FK only) ----
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip, is_radian=True)
    time.sleep(0.5)
    assert arm.connected, f"Cannot connect to {args.ip}"
    arm.set_simulation_robot(on_off=True)
    print(f"SDK connected  firmware={arm.version}  simulation_mode=ON  (FK only)")
    print()

    # ---- GPU warmup (non-trivial solve to trigger all CUDA kernel paths) ----
    _warmup_pos = torch.zeros(1, 3, dtype=torch.float32, device=gs.device)
    _warmup_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=gs.device)
    _warmup_init = np.array([0.5, -0.3, -0.1, 0.3, 0.2, 0.5])  # far from target to force real iterations
    run_genesis_ik(robot, ee_link, _warmup_pos, _warmup_quat, _warmup_init,
                   max_solver_iters=args.max_iters, damping=args.damping)
    print("GPU warmup done.\n")

    # ---- IK comparison loop ----
    results = []       # (name, pos_err_mm, max_rpy_err, max_joint_diff, solver_pos_mm, solver_rot_err, ik_time_ms, passed)
    detail_lines = []

    for name, q_ref in test_configs:

        # 1. SDK FK(q_ref) → target TCP (ground truth)
        code, sdk_pose = arm.get_forward_kinematics(
            angles=q_ref.tolist(), input_is_radian=True, return_is_radian=True,
        )
        if code != 0:
            print(f"[{name}] SDK FK FAILED code={code}, skip")
            continue
        target_pos_mm = np.array(sdk_pose[:3])
        target_rpy_rad = np.array(sdk_pose[3:6])

        # Convert target to Genesis format
        target_pos_m_t = torch.tensor(
            target_pos_mm / 1000.0, dtype=torch.float32, device=gs.device
        ).unsqueeze(0)
        w, x, y, z = rpy_to_quat(target_rpy_rad[0], target_rpy_rad[1], target_rpy_rad[2])
        target_quat_t = torch.tensor(
            [w, x, y, z], dtype=torch.float32, device=gs.device
        ).unsqueeze(0)

        # 2. Two IK tests per config
        init_cases = [
            ("near", q_ref + rng.uniform(-NEAR_PERTURB_RAD, NEAR_PERTURB_RAD, size=6)),
            ("far",  HOME_Q.copy()),
        ]

        for init_label, init_qpos in init_cases:
            test_name = f"{name}/{init_label}"

            t0 = time.perf_counter()
            ik_out = run_genesis_ik(robot, ee_link, target_pos_m_t, target_quat_t, init_qpos,
                                   max_solver_iters=args.max_iters, damping=args.damping)
            ik_time_ms = (time.perf_counter() - t0) * 1000.0
            if ik_out is None:
                print(f"[{test_name}] Genesis IK returned None")
                results.append((test_name, None, None, None, None, None, None, False))
                continue

            q_ik, solver_pos_err, solver_rot_err = ik_out

            # Verify: Genesis FK(q_ik) vs target TCP
            gs_fk_pos_m, gs_fk_quat = genesis_fk(robot, q_ik, ee_link_idx)
            gs_fk_pos_mm = gs_fk_pos_m * 1000.0
            gs_fk_rpy = np.array(quat_to_rpy(gs_fk_quat))

            pos_err_mm = float(np.linalg.norm(gs_fk_pos_mm - target_pos_mm))
            rpy_err_deg = np.array([
                angle_diff_deg(a, b) for a, b in zip(gs_fk_rpy, target_rpy_rad)
            ])
            max_rpy_err = float(rpy_err_deg.max())

            # Joint angle comparison vs q_ref (informational — may differ due to multiple solutions)
            joint_diff_deg = np.array([
                angle_diff_deg(a, b) for a, b in zip(q_ik, q_ref)
            ])
            max_joint_diff = float(joint_diff_deg.max())

            passed = (pos_err_mm <= PASS_POS_MM) and (max_rpy_err <= PASS_RPY_DEG)
            results.append((test_name, pos_err_mm, max_rpy_err, max_joint_diff,
                            solver_pos_err * 1000.0, solver_rot_err, ik_time_ms, passed))

            # Detail lines
            q_ref_deg = np.degrees(q_ref)
            q_ik_deg = np.degrees(q_ik)
            init_deg = np.degrees(init_qpos)
            target_rpy_deg = np.degrees(target_rpy_rad)
            gs_fk_rpy_deg = np.degrees(gs_fk_rpy)
            pos_delta_mm = gs_fk_pos_mm - target_pos_mm

            detail_lines.append(f"\n[{test_name}]  init={init_label}")
            detail_lines.append(
                f"  q_ref (deg)     : [{', '.join(f'{v:7.2f}' for v in q_ref_deg)}]"
            )
            detail_lines.append(
                f"  init_qpos (deg) : [{', '.join(f'{v:7.2f}' for v in init_deg)}]"
            )
            detail_lines.append(
                f"  Target TCP (mm) : [{target_pos_mm[0]:9.3f}, {target_pos_mm[1]:9.3f}, {target_pos_mm[2]:9.3f}]"
                f"  rpy(deg): [{target_rpy_deg[0]:7.3f}, {target_rpy_deg[1]:7.3f}, {target_rpy_deg[2]:7.3f}]"
            )
            detail_lines.append(
                f"  IK result (deg) : [{', '.join(f'{v:7.2f}' for v in q_ik_deg)}]"
            )
            detail_lines.append(
                f"  Joint diff (deg): [{', '.join(f'{v:7.3f}' for v in joint_diff_deg)}]  max={max_joint_diff:.3f}"
            )
            detail_lines.append(
                f"  GS FK(IK) (mm)  : [{gs_fk_pos_mm[0]:9.3f}, {gs_fk_pos_mm[1]:9.3f}, {gs_fk_pos_mm[2]:9.3f}]"
                f"  rpy: [{gs_fk_rpy_deg[0]:7.3f}, {gs_fk_rpy_deg[1]:7.3f}, {gs_fk_rpy_deg[2]:7.3f}]"
            )
            detail_lines.append(
                f"  FK err to target: pos={pos_err_mm:.3f} mm  Δ=[{pos_delta_mm[0]:+.3f}, {pos_delta_mm[1]:+.3f}, {pos_delta_mm[2]:+.3f}]"
                f"  rpy_max={max_rpy_err:.3f} deg"
            )
            detail_lines.append(
                f"  Solver residual : pos={solver_pos_err*1000:.3f} mm  rot={math.degrees(solver_rot_err):.3f} deg"
                f"  time={ik_time_ms:.1f} ms"
            )

    # ---- Summary table ----
    print()
    print(f"{'Test':<20} {'FK pos(mm)':>10} {'FK rpy(°)':>10} {'Joint Δ(°)':>10} {'Solv pos(mm)':>12} {'Solv rot(°)':>11} {'Time(ms)':>9}  Status")
    print("-" * 105)
    for r in results:
        test_name = r[0]
        if r[1] is None:
            print(f"{test_name:<20} {'FAILED':>10} {'':>10} {'':>10} {'':>12} {'':>11} {'':>9}  [FAIL]")
        else:
            status = "PASS" if r[7] else "FAIL"
            print(f"{test_name:<20} {r[1]:>10.3f} {r[2]:>10.3f} {r[3]:>10.3f} {r[4]:>12.3f} {math.degrees(r[5]) if r[5] else 0:>11.3f} {r[6]:>9.1f}  [{status}]")

    valid = [r for r in results if r[1] is not None]
    if valid:
        max_pos = max(r[1] for r in valid)
        max_rpy = max(r[2] for r in valid)
        max_jnt = max(r[3] for r in valid)
        avg_time = sum(r[6] for r in valid) / len(valid)
        max_time = max(r[6] for r in valid)
        all_passed = all(r[7] for r in results)
        n_pass = sum(1 for r in results if r[7])
        print()
        print(f"Max FK pos err   : {max_pos:.3f} mm   (threshold: {PASS_POS_MM} mm)")
        print(f"Max FK rpy err   : {max_rpy:.3f} deg  (threshold: {PASS_RPY_DEG} deg)")
        print(f"Max joint diff   : {max_jnt:.3f} deg  (vs q_ref, informational)")
        print(f"IK time          : avg={avg_time:.1f} ms  max={max_time:.1f} ms")
        print(f"Passed           : {n_pass}/{len(results)}")
        print(f"Overall          : {'PASS' if all_passed else 'FAIL'}")

    # ---- Detailed output ----
    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    for line in detail_lines:
        print(line)

    arm.set_simulation_robot(on_off=False)
    arm.disconnect()
    print("\nSDK disconnected.")


if __name__ == "__main__":
    main()
