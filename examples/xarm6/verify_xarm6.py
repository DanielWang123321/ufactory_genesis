"""
xArm 6 Verification Script for Genesis Simulation.
Tests: URDF loading, rendering, FK, IK, joint PD control.
Optional: Compare IK with real xArm 6 via xarm-python-sdk.

Usage:
    source ~/envs/py312/bin/activate
    python examples/xarm6/verify_xarm6.py -v                           # with viewer
    python examples/xarm6/verify_xarm6.py                              # headless
    python examples/xarm6/verify_xarm6.py --real-ip 192.168.1.60       # + real robot IK comparison
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import xml.etree.ElementTree as ET

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.paths import PROJECT_ROOT, xarm6_urdf

REPO_ROOT = PROJECT_ROOT
XARM6_URDF_PATH = xarm6_urdf()

# Joint/link names (URDF style, with fallback for namespaced names)
JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
EE_LINK_NAME = "link6"


def resolve_entity_name(entity, requested_name: str, kind: str) -> str:
    """Resolve either namespaced or raw link/joint names."""
    available = {item.name for item in entity.joints} if kind == "joint" else {item.name for item in entity.links}
    if requested_name in available:
        return requested_name
    fallback = requested_name.split("/")[-1]
    if fallback in available:
        return fallback
    raise KeyError(f"{kind.capitalize()} name not found: {requested_name}. Available: {sorted(available)}")


def quat_to_rpy(quat):
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw (rad)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(roll, pitch, yaw):
    """Convert roll-pitch-yaw (rad) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z


def normalize_angle_to_pi(angle: float) -> float:
    """Normalize an angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_deg(a: float, b: float) -> float:
    """Compute smallest wrapped angle difference in degrees."""
    return abs(normalize_angle_to_pi(a - b)) * 180.0 / math.pi


def load_kinematics_yaml(kinematics_yaml_path: str) -> Dict[str, Dict[str, float]]:
    """Load joint offsets from xArm kinematics YAML."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load kinematics YAML. Install with `pip install pyyaml`."
        ) from e

    yaml_path = Path(kinematics_yaml_path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Kinematics YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid kinematics YAML format: {yaml_path}")

    kinematics = data.get("kinematics", data)
    if not isinstance(kinematics, dict):
        raise ValueError(f"Invalid kinematics block in YAML: {yaml_path}")

    values = {}
    for i in range(1, 7):
        joint_key = f"joint{i}"
        cfg = kinematics.get(joint_key, {})
        if not isinstance(cfg, dict):
            cfg = {}
        values[joint_key] = {
            "x": float(cfg.get("x", 0.0)),
            "y": float(cfg.get("y", 0.0)),
            "z": float(cfg.get("z", 0.0)),
            "roll": float(cfg.get("roll", 0.0)),
            "pitch": float(cfg.get("pitch", 0.0)),
            "yaw": float(cfg.get("yaw", 0.0)),
        }
    return values


def find_kinematics_yaml(
    kinematics_suffix: str,
    kinematics_yaml_dir: Optional[str] = None,
) -> Path:
    """Find a kinematics yaml file from a suffix (e.g., SUFFIX -> *SUFFIX*.yaml)."""
    suffix = (kinematics_suffix or "").strip()
    if not suffix:
        raise ValueError("kinematics_suffix is empty")

    search_dirs = [Path.cwd()]
    if kinematics_yaml_dir:
        search_dirs.append(Path(kinematics_yaml_dir).expanduser())
    search_dirs.append(REPO_ROOT)

    patterns = (
        f"*kinematics_{suffix}.yaml",
        f"*{suffix}*.yaml",
    )
    for root in search_dirs:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"Cannot find kinematics YAML for suffix '{suffix}'. "
        "Please pass --kinematics-yaml explicitly."
    )


def build_calibrated_urdf(
    base_urdf_path: str,
    kinematics: Dict[str, Dict[str, float]],
    suffix: Optional[str] = None,
) -> str:
    """Generate a patched URDF with calibrated joint origins for testing."""
    base = Path(base_urdf_path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base URDF not found: {base}")

    safe_suffix = "calib"
    if suffix:
        safe_suffix = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in suffix) or "calib"

    output_path = base.with_name(f"{base.stem}_{safe_suffix}_calib.urdf")
    tree = ET.parse(str(base))
    root = tree.getroot()

    for i in range(1, 7):
        joint_name = f"joint{i}"
        target = None
        for joint in root.findall("joint"):
            if joint.get("name") == joint_name:
                target = joint
                break

        if target is None:
            continue

        cfg = kinematics.get(joint_name, {})
        x = float(cfg.get("x", 0.0))
        y = float(cfg.get("y", 0.0))
        z = float(cfg.get("z", 0.0))
        roll = float(cfg.get("roll", 0.0))
        pitch = float(cfg.get("pitch", 0.0))
        yaw = float(cfg.get("yaw", 0.0))

        origin = target.find("origin")
        if origin is None:
            origin = ET.Element("origin")
            target.insert(0, origin)
        origin.set("xyz", f"{x} {y} {z}")
        origin.set("rpy", f"{roll} {pitch} {yaw}")

    try:
        ET.indent(tree)  # Python 3.9+
    except AttributeError:
        pass
    tree.write(str(output_path), encoding="utf-8", xml_declaration=False)
    return str(output_path)


def prepare_robot_model_for_verification(
    robot_model: Optional[str],
    kinematics_yaml: Optional[str],
    kinematics_suffix: Optional[str],
    kinematics_yaml_dir: Optional[str] = None,
) -> str:
    """Resolve robot model and apply kinematic calibration if requested."""
    model_path = Path(robot_model).expanduser().resolve() if robot_model else Path(XARM6_URDF_PATH)

    if kinematics_yaml is None and kinematics_suffix is None:
        return str(model_path)

    if kinematics_yaml is not None:
        yaml_path = Path(kinematics_yaml).expanduser().resolve()
    else:
        yaml_path = find_kinematics_yaml(kinematics_suffix, kinematics_yaml_dir)

    return build_calibrated_urdf(
        str(model_path),
        load_kinematics_yaml(str(yaml_path)),
        suffix=kinematics_suffix or yaml_path.stem,
    )


def run_genesis_tests(args, robot_model_path):
    """Part A: Genesis URDF loading and basic verification."""
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.vis,
    )

    # Ground plane
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    # Load xArm 6 from URDF
    robot_model = Path(robot_model_path).resolve()
    xarm6 = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_model),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            requires_jac_and_IK=True,
        )
    )
    print(f"Loaded robot model: {robot_model}")
    assert xarm6.n_dofs in (6, 12), f"Expected 6 or 12 DOFs, got {xarm6.n_dofs}"

    scene.build()

    # Print robot info
    print("=" * 60)
    print("xArm 6 Robot Info")
    print("=" * 60)
    print(f"Number of DOFs: {xarm6.n_dofs}")
    print(f"Number of links: {xarm6.n_links}")
    print(f"Joint names: {[j.name for j in xarm6.joints]}")
    print(f"Link names: {[l.name for l in xarm6.links]}")

    print(f"\n[PASS] DOF count = {xarm6.n_dofs}")

    # Get joint indices
    available_joint_names = {j.name: j for j in xarm6.joints}
    dof_idx = []
    for name in JOINT_NAMES:
        target_name = resolve_entity_name(xarm6, name, "joint")
        dof_idx.append(available_joint_names[target_name].dofs_idx_local[0])

    # Set PD gains
    xarm6.set_dofs_kp(np.array([3000, 3000, 2000, 2000, 1000, 1000]), dof_idx)
    xarm6.set_dofs_kv(np.array([300, 300, 200, 200, 100, 100]), dof_idx)
    xarm6.set_dofs_force_range(
        np.array([-50, -50, -32, -32, -32, -20]),
        np.array([50, 50, 32, 32, 32, 20]),
        dof_idx,
    )

    # Test 1: Forward Kinematics
    print("\n--- Test 1: Forward Kinematics ---")
    test_qpos = np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    xarm6.set_dofs_position(test_qpos, dof_idx)
    for _ in range(50):
        scene.step()

    ee_link = xarm6.get_link(resolve_entity_name(xarm6, EE_LINK_NAME, "link"))
    ee_pos = ee_link.get_pos()
    ee_quat = ee_link.get_quat()
    print(f"  Test qpos: {test_qpos}")
    print(f"  EE position (m): {ee_pos}")
    print(f"  EE quaternion (w,x,y,z): {ee_quat}")

    ee_z = ee_pos[2].item() if ee_pos.dim() == 1 else ee_pos[0, 2].item()
    assert ee_z > 0.0, f"EE should be above ground, got z={ee_z}"
    print("[PASS] FK: EE is above ground")

    # Test 2: Inverse Kinematics
    print("\n--- Test 2: Inverse Kinematics ---")
    target_pos = ee_pos.clone()
    if target_pos.dim() == 1:
        target_pos = target_pos.unsqueeze(0)
    target_pos[0, 0] += 0.05

    target_quat = ee_quat.clone()
    if target_quat.dim() == 1:
        target_quat = target_quat.unsqueeze(0)

    ik_qpos = xarm6.inverse_kinematics(
        link=ee_link,
        pos=target_pos,
        quat=target_quat,
    )
    print(f"  Target EE pos: {target_pos}")
    print(f"  IK solution qpos: {ik_qpos}")
    assert ik_qpos is not None, "IK should return a solution"
    print("[PASS] IK: Solution found")

    # Test 3: PD Position Control
    print("\n--- Test 3: PD Position Control ---")
    target_positions = [
        np.array([0.5, -0.3, 0.0, 0.0, 0.3, 0.0]),
        np.array([-0.5, 0.3, -0.15, 0.5, -0.3, 0.2]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    for i, target in enumerate(target_positions):
        xarm6.control_dofs_position(target, dof_idx)
        for _ in range(200):
            scene.step()
        actual_qpos = xarm6.get_dofs_position(dof_idx)
        actual_np = actual_qpos.cpu().numpy().flatten()
        error = np.abs(actual_np - target).max()
        print(f"  Config {i}: max error={error:.4f} rad")
        assert error < 0.15, f"PD control error too large: {error}"
    print("[PASS] PD position control: all configs reached")

    # Test 4: Joint state readout
    print("\n--- Test 4: Joint State Readout ---")
    qpos = xarm6.get_dofs_position(dof_idx)
    qvel = xarm6.get_dofs_velocity(dof_idx)
    print(f"  Joint positions: {qpos}")
    print(f"  Joint velocities: {qvel}")
    print("[PASS] Joint state readout works")

    print("\n" + "=" * 60)
    print("Part A: All Genesis verification tests PASSED!")
    print("=" * 60)

    return xarm6, ee_link, dof_idx, scene


def genesis_fk(xarm6, qpos_np, ee_link_idx):
    """Pure math FK using Genesis forward_kinematics (no sim stepping).

    Note: for n_envs=0 (single env), forward_kinematics internally adds the
    batch dimension, so we pass a 1D tensor of shape (n_qs,).
    """
    qpos_t = torch.tensor(qpos_np, dtype=torch.float32, device=gs.device)
    links_pos, links_quat = xarm6.forward_kinematics(qpos=qpos_t)

    ee_link_idx = int(ee_link_idx)

    if links_pos.ndim == 2:
        if ee_link_idx < 0 or ee_link_idx >= links_pos.shape[0]:
            raise IndexError(
                f"EE link index out of range: idx={ee_link_idx}, "
                f"n_links={links_pos.shape[0]}, fk_shape={tuple(links_pos.shape)}"
            )
        pos = links_pos[ee_link_idx].cpu().numpy()    # (3,) meters
        quat = links_quat[ee_link_idx].cpu().numpy()  # (4,) w,x,y,z
    elif links_pos.ndim == 3:
        if ee_link_idx < 0 or ee_link_idx >= links_pos.shape[1]:
            raise IndexError(
                f"EE link index out of range: idx={ee_link_idx}, "
                f"n_links={links_pos.shape[1]}, fk_shape={tuple(links_pos.shape)}"
            )
        pos = links_pos[0, ee_link_idx].cpu().numpy()    # (3,) meters
        quat = links_quat[0, ee_link_idx].cpu().numpy()  # (4,) w,x,y,z
    else:
        raise RuntimeError(
            f"Unexpected forward_kinematics output shape for xarm6: {tuple(links_pos.shape)}"
        )

    return pos, quat


def run_real_robot_comparison(xarm6, ee_link, real_ip):
    """Part B: Compare Genesis FK/IK with real xArm 6 via xarm-python-sdk."""
    import time
    from xarm.wrapper import XArmAPI

    print("\n" + "=" * 60)
    print(f"Part B: Real Robot FK/IK Comparison (IP: {real_ip})")
    print("=" * 60)

    # Connect to real robot in simulation mode
    arm = XArmAPI(real_ip, is_radian=True)
    time.sleep(0.5)
    assert arm.connected, f"Failed to connect to xArm at {real_ip}"
    arm.set_simulation_robot(on_off=True)
    print(f"Connected to xArm (firmware: {arm.version}), simulation mode ON")

    # Find EE link index for forward_kinematics
    ee_link_idx = int(ee_link.idx_local)
    print(f"EE link: {EE_LINK_NAME} (index={ee_link_idx})")

    # Test joint configurations
    test_configs = [
        ("home",     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("config A", np.array([0.5, -0.3, 0.0, 0.0, 0.3, 0.0])),
        ("config B", np.array([0.0, -0.5, -0.1, 0.5, 0.5, 0.0])),
        ("config C", np.array([-0.3, 0.2, -0.15, 0.3, -0.2, 0.1])),
    ]

    # ========== FK Comparison ==========
    print("\n--- FK Comparison (pure math, no sim stepping) ---")
    print(f"{'Name':<12} {'Genesis pos (mm)':<35} {'SDK pos (mm)':<35} {'Pos diff (mm)'}")
    print("-" * 100)
    max_fk_pos_diff_mm = 0.0
    max_fk_rpy_diff_deg = 0.0

    for name, q in test_configs:
        # Genesis FK (pure math)
        gs_pos_m, gs_quat = genesis_fk(xarm6, q, ee_link_idx)
        gs_pos_mm = gs_pos_m * 1000.0
        gs_rpy = np.array(quat_to_rpy(gs_quat))
        gs_rpy_deg = gs_rpy * 180.0 / math.pi
        q_deg = q * 180.0 / math.pi

        # SDK FK
        code, sdk_pose = arm.get_forward_kinematics(
            angles=q.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        assert code == 0, f"SDK FK failed with code {code}"
        sdk_pos_mm = np.array(sdk_pose[:3])
        sdk_rpy = np.array(sdk_pose[3:6])
        sdk_rpy_deg = sdk_rpy * 180.0 / math.pi

        pos_diff = np.linalg.norm(gs_pos_mm - sdk_pos_mm)
        rpy_diff_deg = np.array([angle_diff_deg(a, b) for a, b in zip(gs_rpy, sdk_rpy)])
        pos_delta_mm = gs_pos_mm - sdk_pos_mm
        max_fk_pos_diff_mm = max(max_fk_pos_diff_mm, float(pos_diff))
        max_fk_rpy_diff_deg = max(max_fk_rpy_diff_deg, float(rpy_diff_deg.max()))

        print(f"[{name}] Send q (rad):      [{', '.join(f'{v:8.4f}' for v in q)}]")
        print(f"{'':12}Send q (deg):      [{', '.join(f'{v:8.2f}' for v in q_deg)}]")
        print(
            f"{'':12}Genesis FK pos (mm): [{gs_pos_mm[0]:8.2f}, {gs_pos_mm[1]:8.2f}, {gs_pos_mm[2]:8.2f}]"
        )
        print(
            f"{'':12}SDK FK pos (mm):    [{sdk_pos_mm[0]:8.2f}, {sdk_pos_mm[1]:8.2f}, {sdk_pos_mm[2]:8.2f}]"
        )
        print(
            f"{'':12}Pos delta (GS-SDK): [{pos_delta_mm[0]:8.2f}, {pos_delta_mm[1]:8.2f}, {pos_delta_mm[2]:8.2f}]  "
            f"norm={pos_diff:8.4f} mm"
        )
        print(
            f"{'':12}Genesis RPY (deg):  [{gs_rpy_deg[0]:7.2f}, {gs_rpy_deg[1]:7.2f}, {gs_rpy_deg[2]:7.2f}]"
        )
        print(
            f"{'':12}SDK FK RPY (deg):   [{sdk_rpy_deg[0]:7.2f}, {sdk_rpy_deg[1]:7.2f}, {sdk_rpy_deg[2]:7.2f}]"
        )
        print(
            f"{'':12}RPY delta (deg):    [{rpy_diff_deg[0]:7.2f}, {rpy_diff_deg[1]:7.2f}, {rpy_diff_deg[2]:7.2f}]  "
            f"max={rpy_diff_deg.max():.2f}°"
        )
        print()

    # ========== IK Comparison ==========
    print("\n--- IK Comparison ---")
    print("For each config: SDK FK → target TCP, then solve IK with both, compare joint angles\n")

    max_joint_diff_deg = 0.0
    max_gs_verify_err_mm = 0.0
    max_sdk_verify_err_mm = 0.0
    ik_fail_count = 0

    for name, q_ref in test_configs:
        q_ref_t = torch.tensor(q_ref, dtype=torch.float32, device=gs.device)
        q_ref_deg = q_ref * 180.0 / math.pi

        # Get target TCP from SDK FK
        code, sdk_pose = arm.get_forward_kinematics(
            angles=q_ref.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        assert code == 0, f"SDK FK failed"
        target_pos_mm = np.array(sdk_pose[:3])
        target_rpy = np.array(sdk_pose[3:6])
        target_rpy_deg = target_rpy * 180.0 / math.pi

        print(f"[{name}] Target ref q (rad): [{', '.join(f'{v:8.4f}' for v in q_ref)}]")
        print(f"{'':14}Target ref q (deg): [{', '.join(f'{v:8.2f}' for v in q_ref_deg)}]")
        print(f"[{name}] Target TCP: pos(mm)=[{target_pos_mm[0]:.2f}, {target_pos_mm[1]:.2f}, {target_pos_mm[2]:.2f}] "
              f"rpy(deg)=[{target_rpy_deg[0]:.2f}, {target_rpy_deg[1]:.2f}, {target_rpy_deg[2]:.2f}]")

        # SDK IK
        code, sdk_ik_angles = arm.get_inverse_kinematics(
            pose=sdk_pose,
            input_is_radian=True,
            return_is_radian=True,
            ref_angles=q_ref.tolist(),
        )
        if code != 0:
            print(f"  SDK IK: FAILED (code={code})")
            ik_fail_count += 1
            continue
        sdk_joints_rad = np.array(sdk_ik_angles[:6])

        # Genesis IK: convert target to Genesis format (m + quat)
        target_pos_m = torch.tensor(
            target_pos_mm / 1000.0, dtype=torch.float32, device=gs.device
        ).unsqueeze(0)
        w, x, y, z = rpy_to_quat(target_rpy[0], target_rpy[1], target_rpy[2])
        target_quat = torch.tensor(
            [w, x, y, z], dtype=torch.float32, device=gs.device
        ).unsqueeze(0)

        gs_ik_qpos = xarm6.inverse_kinematics(
            link=ee_link,
            pos=target_pos_m,
            quat=target_quat,
            init_qpos=q_ref_t,
        )
        if gs_ik_qpos is None:
            print("  Genesis IK: FAILED (no solution)")
            ik_fail_count += 1
            continue
        gs_joints_rad = gs_ik_qpos.cpu().numpy().flatten()[:6]
        gs_joints_deg = gs_joints_rad * 180.0 / math.pi

        # Joint angle comparison
        joint_diff_deg = np.array(
            [angle_diff_deg(a, b) for a, b in zip(gs_joints_rad, sdk_joints_rad)]
        )

        print(f"  SDK IK joints (deg):     [{', '.join(f'{a:8.2f}' for a in sdk_joints_rad * 180 / math.pi)}]")
        print(f"  Genesis IK joints (deg): [{', '.join(f'{a:8.2f}' for a in gs_joints_deg)}]")
        print(f"  Joint diff (deg):        [{', '.join(f'{d:8.2f}' for d in joint_diff_deg)}]  max={joint_diff_deg.max():.2f}°")
        print(f"  Joint diff L2 norm:      {np.linalg.norm(joint_diff_deg):.4f} deg")

        # Verify both IK solutions with FK
        # Genesis FK(Genesis IK result)
        gs_verify_pos, gs_verify_quat = genesis_fk(xarm6, gs_joints_rad, ee_link_idx)
        gs_verify_pos_mm = gs_verify_pos * 1000.0
        gs_verify_rpy = np.array(quat_to_rpy(gs_verify_quat))
        gs_verify_rpy_deg = gs_verify_rpy * 180.0 / math.pi
        gs_verify_err = np.linalg.norm(gs_verify_pos_mm - target_pos_mm)
        gs_verify_pos_delta_mm = gs_verify_pos_mm - target_pos_mm
        gs_verify_rpy_delta_deg = np.array(
            [angle_diff_deg(a, b) for a, b in zip(gs_verify_rpy, target_rpy)]
        )

        # SDK FK(SDK IK result)
        _, sdk_verify_pose = arm.get_forward_kinematics(
            angles=sdk_joints_rad.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        sdk_verify_pos_mm = np.array(sdk_verify_pose[:3])
        sdk_verify_rpy = np.array(sdk_verify_pose[3:6])
        sdk_verify_rpy_deg = sdk_verify_rpy * 180.0 / math.pi
        sdk_verify_err = np.linalg.norm(sdk_verify_pos_mm - target_pos_mm)
        sdk_verify_pos_delta_mm = sdk_verify_pos_mm - target_pos_mm
        sdk_verify_rpy_delta_deg = np.array(
            [angle_diff_deg(a, b) for a, b in zip(sdk_verify_rpy, target_rpy)]
        )

        print(
            f"  Genesis FK from IK pos(mm): [{gs_verify_pos_mm[0]:8.2f}, {gs_verify_pos_mm[1]:8.2f}, {gs_verify_pos_mm[2]:8.2f}]"
        )
        print(
            f"  SDK FK from IK pos(mm):    [{sdk_verify_pos_mm[0]:8.2f}, {sdk_verify_pos_mm[1]:8.2f}, {sdk_verify_pos_mm[2]:8.2f}]"
        )
        print(
            f"  Pos delta to target(mm):   [{gs_verify_pos_delta_mm[0]:8.2f}, {gs_verify_pos_delta_mm[1]:8.2f}, {gs_verify_pos_delta_mm[2]:8.2f}]  "
            f"GS err={gs_verify_err:.4f} mm"
        )
        print(
            f"  Pos delta to target(mm):   [{sdk_verify_pos_delta_mm[0]:8.2f}, {sdk_verify_pos_delta_mm[1]:8.2f}, {sdk_verify_pos_delta_mm[2]:8.2f}]  "
            f"SDK err={sdk_verify_err:.4f} mm"
        )
        print(
            f"  Genesis FK RPY(deg):       [{gs_verify_rpy_deg[0]:7.2f}, {gs_verify_rpy_deg[1]:7.2f}, {gs_verify_rpy_deg[2]:7.2f}]"
        )
        print(
            f"  SDK FK RPY(deg):          [{sdk_verify_rpy_deg[0]:7.2f}, {sdk_verify_rpy_deg[1]:7.2f}, {sdk_verify_rpy_deg[2]:7.2f}]"
        )
        print(
            f"  RPY delta to target(deg):  [{gs_verify_rpy_delta_deg[0]:7.2f}, {gs_verify_rpy_delta_deg[1]:7.2f}, {gs_verify_rpy_delta_deg[2]:7.2f}]"
        )
        print(
            f"  RPY delta to target(deg):  [{sdk_verify_rpy_delta_deg[0]:7.2f}, {sdk_verify_rpy_delta_deg[1]:7.2f}, {sdk_verify_rpy_delta_deg[2]:7.2f}]"
        )

        max_joint_diff_deg = max(max_joint_diff_deg, float(joint_diff_deg.max()))
        max_gs_verify_err_mm = max(max_gs_verify_err_mm, float(gs_verify_err))
        max_sdk_verify_err_mm = max(max_sdk_verify_err_mm, float(sdk_verify_err))

        print(f"  Genesis FK(Genesis IK) err: {gs_verify_err:.4f} mm")
        print(f"  SDK FK(SDK IK) err:         {sdk_verify_err:.4f} mm")
        print()

    print(f"IK_SUMMARY_MAX_JOINT_DIFF_DEG={max_joint_diff_deg:.4f}")
    print(f"IK_SUMMARY_MAX_GS_VERIFY_ERR_MM={max_gs_verify_err_mm:.4f}")
    print(f"IK_SUMMARY_MAX_SDK_VERIFY_ERR_MM={max_sdk_verify_err_mm:.4f}")
    print(f"IK_SUMMARY_FAIL_COUNT={ik_fail_count}")
    print(f"FK_SUMMARY_MAX_POS_MM={max_fk_pos_diff_mm:.4f}")
    print(f"FK_SUMMARY_MAX_RPY_DIFF_DEG={max_fk_rpy_diff_deg:.4f}")

    # Cleanup
    arm.set_simulation_robot(on_off=False)
    arm.disconnect()
    print("Real robot disconnected, simulation mode OFF")

    print("\n" + "=" * 60)
    print("Part B: Real Robot FK/IK Comparison DONE!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="xArm 6 Verification")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--robot-model", type=str, default=None,
                        help="Robot URDF model path. Default: xarm6.urdf")
    parser.add_argument(
        "--kinematics-suffix",
        type=str,
        default=None,
        help="Suffix for xArm kinematics YAML file, e.g. SUFFIX from xarm6_kinematics_SUFFIX.yaml",
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
        help="Directory used to auto-find kinematics yaml when only suffix is provided.",
    )
    parser.add_argument("--real-ip", type=str, default=None,
                        help="Real xArm IP for IK comparison (e.g., 192.168.1.60)")
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Skip real-robot IK comparison and only run pure Genesis tests.",
    )
    args = parser.parse_args()

    robot_model = prepare_robot_model_for_verification(
        args.robot_model,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
    )
    xarm6, ee_link, _, _ = run_genesis_tests(args, robot_model)

    if args.real_ip:
        if args.skip_ik:
            print("[INFO] --skip-ik enabled, skip real robot IK comparison.")
        else:
            run_real_robot_comparison(xarm6, ee_link, args.real_ip)


if __name__ == "__main__":
    main()
