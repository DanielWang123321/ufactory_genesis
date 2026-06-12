"""
xArm 6 FK Verification: Genesis (URDF) vs. xarm-python-sdk

According to Notes:
  - SDK get_forward_kinematics() in simulation mode is accurate (includes kinematic
    compensation) and serves as ground truth.
  - Genesis FK must match SDK FK to validate the URDF model.

Key:
  - No real robot movement. SDK runs in simulation mode only.
  - Genesis uses forward_kinematics() (pure math, no sim stepping).
  - Default base URDF: xarm6_1305.urdf (XI1305). Apply per-robot calibration via
    --kinematics-suffix or --kinematics-yaml (extract with scripts/gen_kinematics_params.py).

Usage:
    source ~/envs/py312/bin/activate
    python scripts/gen_kinematics_params.py 192.168.1.60 xi1305
    python examples/xarm6/fk_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305
    python examples/xarm6/fk_verify.py --ip 192.168.1.60 -v
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.kinematics import (
    get_robot_sn,
    log_kinematics_sn_status,
    prepare_robot_model_for_verification,
    validate_kinematics_calibration_request,
)
from ufactory.paths import xarm6_1305_urdf

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
EE_LINK_NAME = "link6"  # xArm TCP = flange = link6 (no tool)

# Pass/fail thresholds
PASS_POS_MM = 1.0   # mm
PASS_RPY_DEG = 0.5  # degrees

# Comprehensive test configurations (rad), covering a range of poses
TEST_CONFIGS = [
    ("home",     np.array([ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ])),
    ("config_A", np.array([ 0.5,  -0.3,   0.0,   0.0,   0.3,   0.0 ])),
    ("config_B", np.array([ 0.0,  -0.5,  -0.1,   0.5,   0.5,   0.0 ])),
    ("config_C", np.array([-0.3,   0.2,  -0.15,  0.3,  -0.2,   0.1 ])),
    ("config_D", np.array([ 1.0,  -0.8,   0.0,   1.0,  -0.5,   1.5 ])),
    ("config_E", np.array([-1.0,   0.5,  -0.5,  -1.0,   1.0,  -1.5 ])),
    ("config_F", np.array([ 0.0,  -1.0,  -0.3,   0.0,   1.5,   0.0 ])),
    ("config_G", np.array([ 1.5,  -0.2,  -0.1,   0.8,  -0.8,   2.0 ])),
    ("config_H", np.array([-1.5,   0.8,  -0.8,  -1.5,   1.2,  -2.0 ])),
    ("config_I", np.array([ 0.3,  -0.6,  -0.2,   0.6,   0.8,   0.5 ])),
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_to_rpy(quat):
    """Quaternion (w, x, y, z) → roll-pitch-yaw (rad)."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed-magnitude difference between two angles, in degrees."""
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) * 180.0 / math.pi


# ---------------------------------------------------------------------------
# Genesis setup
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
            requires_jac_and_IK=True,  # forward_kinematics() needs IK internals
        )
    )
    scene.build()
    return scene, robot


def genesis_fk(robot, qpos_np: np.ndarray, ee_link_idx: int):
    """Pure-math FK via Genesis forward_kinematics (no simulation stepping)."""
    qpos_t = torch.tensor(qpos_np, dtype=torch.float32, device=gs.device)
    links_pos, links_quat = robot.forward_kinematics(qpos=qpos_t)
    if links_pos.ndim == 2:
        pos = links_pos[ee_link_idx].cpu().numpy()
        quat = links_quat[ee_link_idx].cpu().numpy()
    elif links_pos.ndim == 3:
        pos = links_pos[0, ee_link_idx].cpu().numpy()
        quat = links_quat[0, ee_link_idx].cpu().numpy()
    else:
        raise RuntimeError(f"Unexpected forward_kinematics shape: {tuple(links_pos.shape)}")
    return pos, quat  # pos in meters, quat as (w, x, y, z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="xArm 6 FK Verification: Genesis vs. SDK")
    parser.add_argument(
        "--ip", type=str, required=True,
        help="xArm IP address (e.g., 192.168.1.60). SDK runs in simulation mode — no motion.",
    )
    parser.add_argument(
        "--robot-model", type=str, default=None,
        help="Base URDF path. Default: xarm6_1305.urdf",
    )
    parser.add_argument(
        "--urdf", type=str, default=None,
        help="Alias for --robot-model (deprecated).",
    )
    parser.add_argument(
        "--kinematics-suffix",
        type=str,
        default=None,
        help="Kinematics YAML suffix, e.g. xi1305 from xarm6_kinematics_xi1305.yaml",
    )
    parser.add_argument(
        "--kinematics-yaml",
        type=str,
        default=None,
        help="Explicit path to kinematics YAML for URDF offset patching.",
    )
    parser.add_argument(
        "--kinematics-yaml-dir",
        type=str,
        default=None,
        help="Directory to search for kinematics YAML when only suffix is provided.",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Enable Genesis viewer")
    args = parser.parse_args()

    robot_model_arg = args.robot_model or args.urdf

    urdf_path_str, kinematics_yaml_path = prepare_robot_model_for_verification(
        robot_model_arg,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=xarm6_1305_urdf(),
    )
    urdf_path = Path(urdf_path_str).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    print("=" * 80)
    print("xArm 6 FK Verification: Genesis vs. xarm-python-sdk")
    print("=" * 80)
    print(f"URDF : {urdf_path}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")
    print(f"Robot: {args.ip}  [simulation mode, no motion]")
    print(f"Pass : pos < {PASS_POS_MM} mm,  max_rpy < {PASS_RPY_DEG} deg")
    print()

    # --- Genesis ---
    scene, robot = build_genesis_robot(str(urdf_path), args.vis)

    joint_map = {j.name: j for j in robot.joints}
    missing = [n for n in JOINT_NAMES if n not in joint_map]
    if missing:
        raise RuntimeError(f"Joints not found in URDF: {missing}. "
                           f"Available: {sorted(joint_map.keys())}")
    dof_idx = [joint_map[n].dofs_idx_local[0] for n in JOINT_NAMES]

    link_map = {lk.name: lk for lk in robot.links}
    if EE_LINK_NAME not in link_map:
        raise RuntimeError(f"EE link '{EE_LINK_NAME}' not found. "
                           f"Available: {sorted(link_map.keys())}")
    ee_link = link_map[EE_LINK_NAME]
    ee_link_idx = int(ee_link.idx_local)
    print(f"EE link: '{EE_LINK_NAME}'  local_idx={ee_link_idx}")
    print(f"Robot: n_dofs={robot.n_dofs}, n_links={robot.n_links}")
    print()

    # --- SDK (simulation mode) ---
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip, is_radian=True)
    time.sleep(0.5)
    assert arm.connected, f"Failed to connect to xArm at {args.ip}"
    arm.set_simulation_robot(on_off=True)
    arm.clean_error()
    arm.clean_warn()
    time.sleep(0.5)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    if arm.error_code != 0:
        arm.clean_error()
        time.sleep(0.3)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
    time.sleep(0.3)
    print(f"SDK connected  firmware={arm.version}  simulation_mode=ON")
    print(f"error_code={arm.error_code}  state={arm.state}")
    print(f"tcp_offset   : {list(arm.tcp_offset)}")
    print(f"world_offset : {list(arm.world_offset)}")
    sn = get_robot_sn(arm)
    validate_kinematics_calibration_request(
        sn, "xarm6",
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
    )
    log_kinematics_sn_status(
        sn, "xarm6",
        kinematics_yaml=kinematics_yaml_path,
        kinematics_suffix=args.kinematics_suffix,
    )
    print()

    # --- FK comparison loop ---
    print(f"{'Config':<12} {'Pos err (mm)':>14} {'RPY err max (°)':>16}  Status")
    print("-" * 52)

    results = []
    detail_lines = []

    for name, q in TEST_CONFIGS:
        # Genesis FK (pure math)
        gs_pos_m, gs_quat = genesis_fk(robot, q, ee_link_idx)
        gs_pos_mm = gs_pos_m * 1000.0
        gs_rpy = np.array(quat_to_rpy(gs_quat))

        # SDK FK (ground truth, includes kinematic compensation)
        code, sdk_pose = arm.get_forward_kinematics(
            angles=q.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        if code != 0:
            print(f"{name:<12}: SDK FK FAILED (code={code})")
            continue

        sdk_pos_mm = np.array(sdk_pose[:3])
        sdk_rpy = np.array(sdk_pose[3:6])

        # Errors
        pos_delta_mm = gs_pos_mm - sdk_pos_mm
        pos_err_mm = float(np.linalg.norm(pos_delta_mm))
        rpy_diff_deg = np.array([angle_diff_deg(a, b) for a, b in zip(gs_rpy, sdk_rpy)])
        max_rpy_err_deg = float(rpy_diff_deg.max())

        passed = (pos_err_mm <= PASS_POS_MM) and (max_rpy_err_deg <= PASS_RPY_DEG)
        status = "PASS" if passed else "FAIL"
        results.append((name, pos_err_mm, max_rpy_err_deg, passed))

        # Brief table line
        print(f"{name:<12} {pos_err_mm:>14.3f} {max_rpy_err_deg:>16.3f}  [{status}]")

        # Detailed lines (printed after summary)
        detail_lines.append(
            f"\n[{name}]  q(deg)=[{', '.join(f'{np.degrees(v):7.2f}' for v in q)}]"
        )
        detail_lines.append(
            f"  Genesis pos(mm) : [{gs_pos_mm[0]:9.3f}, {gs_pos_mm[1]:9.3f}, {gs_pos_mm[2]:9.3f}]"
            f"  rpy(deg): [{np.degrees(gs_rpy[0]):7.3f}, {np.degrees(gs_rpy[1]):7.3f}, {np.degrees(gs_rpy[2]):7.3f}]"
        )
        detail_lines.append(
            f"  SDK     pos(mm) : [{sdk_pos_mm[0]:9.3f}, {sdk_pos_mm[1]:9.3f}, {sdk_pos_mm[2]:9.3f}]"
            f"  rpy(deg): [{np.degrees(sdk_rpy[0]):7.3f}, {np.degrees(sdk_rpy[1]):7.3f}, {np.degrees(sdk_rpy[2]):7.3f}]"
        )
        detail_lines.append(
            f"  Δpos(mm)        : [{pos_delta_mm[0]:+9.3f}, {pos_delta_mm[1]:+9.3f}, {pos_delta_mm[2]:+9.3f}]"
            f"  norm={pos_err_mm:.3f} mm"
        )
        detail_lines.append(
            f"  Δrpy(deg)       : [{rpy_diff_deg[0]:9.3f}, {rpy_diff_deg[1]:9.3f}, {rpy_diff_deg[2]:9.3f}]"
            f"  max={max_rpy_err_deg:.3f}°"
        )

    # --- Overall summary ---
    print()
    if results:
        max_pos = max(r[1] for r in results)
        max_rpy = max(r[2] for r in results)
        all_passed = all(r[3] for r in results)
        n_pass = sum(1 for r in results if r[3])
        n_total = len(results)
        print(f"Max position error : {max_pos:.3f} mm  (threshold: {PASS_POS_MM} mm)")
        print(f"Max RPY error      : {max_rpy:.3f} °   (threshold: {PASS_RPY_DEG} °)")
        print(f"Passed             : {n_pass}/{n_total}")
        print(f"Overall            : {'PASS' if all_passed else 'FAIL'}")
    else:
        print("No results collected.")

    # --- Detailed output ---
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
