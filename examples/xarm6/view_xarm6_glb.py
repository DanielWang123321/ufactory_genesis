"""
View xArm6 1305 GLB visual model in Genesis.

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    python examples/xarm6/view_xarm6_glb.py              # arm only
    python examples/xarm6/view_xarm6_glb.py --gripper-g2         # arm + Gripper G2 static preview
    python examples/xarm6/view_xarm6_glb.py --gripper-g2 --movable --gripper-demo
    python examples/xarm6/view_xarm6_glb.py --gripper-g2 --movable --pd --gripper-demo
    python examples/xarm6/view_xarm6_glb.py --diagnose   # headless alignment check
    python examples/xarm6/view_xarm6_glb.py --show-tcp   # DH TCP debug marker
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf, xarm6_urdf

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_ROOT))

from _robot_viewer import (
    _apply_kinematic_hold,
    _disable_robot_pd,
    _kinematic_step,
    add_tcp_marker,
    resolve_robot_link,
    start_deferred_viewer,
    update_tcp_marker,
)

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
EE_LINK_NAME = "link6"
PD_KP = [3000, 3000, 2000, 2000, 1000, 1000]
PD_KV = [300, 300, 200, 200, 100, 100]
FORCE_LOWER = [-50, -50, -32, -32, -32, -20]
FORCE_UPPER = [50, 50, 32, 32, 32, 20]
ARM_LINKS = ("link_base", "link1", "link2", "link3", "link4", "link5", "link6")
HOME_QPOS = np.zeros(6)
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.85
GRIPPER_HOLD_STEPS = 200
ALL_GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
def _to_numpy3(pos) -> np.ndarray:
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    return np.asarray(pos).reshape(-1)[:3]


def _link_world_positions(robot) -> dict[str, list[float]]:
    available = {link.name.split("/")[-1]: link for link in robot.links}
    out = {}
    for name in ARM_LINKS:
        if name not in available:
            continue
        pos = available[name].get_pos()
        if hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
        arr = np.asarray(pos).reshape(-1)[:3]
        out[name] = [float(x) for x in arr]
    return out


def _setup_arm_pd(robot, dof_idx: list[int]) -> None:
    n = len(dof_idx)
    robot.set_dofs_kp(np.array(PD_KP[:n]), dof_idx)
    robot.set_dofs_kv(np.array(PD_KV[:n]), dof_idx)
    robot.set_dofs_force_range(
        np.array(FORCE_LOWER[:n]),
        np.array(FORCE_UPPER[:n]),
        dof_idx,
    )


def _hold_home(robot, dof_idx: list[int]) -> None:
    robot.set_dofs_position(HOME_QPOS[: len(dof_idx)], dof_idx)
    robot.control_dofs_position(HOME_QPOS[: len(dof_idx)], dof_idx)


def _setup_gripper_pd(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int]) -> None:
    active_dofs = all_gripper_dof_idx or gripper_dof_idx
    n_grip = len(active_dofs)
    robot.set_dofs_kp(np.full(n_grip, 30.0), active_dofs)
    robot.set_dofs_kv(np.full(n_grip, 6.0), active_dofs)
    robot.set_dofs_force_range(np.full(n_grip, -50.0), np.full(n_grip, 50.0), active_dofs)
    robot.set_dofs_damping(np.full(n_grip, 0.05), active_dofs)
    robot.set_dofs_frictionloss(np.zeros(n_grip), active_dofs)


def _set_gripper_pose(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int], value: float) -> None:
    active_dofs = all_gripper_dof_idx or gripper_dof_idx
    target = np.full(len(active_dofs), value)
    robot.set_dofs_position(target, active_dofs)
    robot.control_dofs_position(target, active_dofs)


def _control_gripper_pose(robot, gripper_dof_idx: list[int], all_gripper_dof_idx: list[int], value: float) -> None:
    active_dofs = all_gripper_dof_idx or gripper_dof_idx
    target = np.full(len(active_dofs), value)
    robot.set_dofs_position(target, active_dofs, zero_velocity=False)
    robot.control_dofs_position(target, active_dofs)


def _gripper_demo_target(step: int) -> float:
    phase = (step // GRIPPER_HOLD_STEPS) % 2
    return GRIPPER_CLOSE if phase else GRIPPER_OPEN


def _ensure_fk_scratch(robot) -> None:
    if hasattr(robot, "_IK_qpos_orig"):
        return
    if robot.n_qs == 0:
        return

    # Genesis forward_kinematics() reuses this IK scratch field but
    # creates it lazily only from the IK path. Allocate just the field FK needs.
    import quadrants as qd

    robot._IK_qpos_orig = qd.field(dtype=gs.qd_float, shape=(robot.n_qs, robot._solver._B))


def _fk_link6_pos(robot, ee_link, qpos_np: np.ndarray) -> np.ndarray:
    _ensure_fk_scratch(robot)
    qpos_t = torch.tensor(qpos_np, dtype=torch.float32, device=gs.device)
    links_pos, _ = robot.forward_kinematics(qpos=qpos_t)
    idx = int(ee_link.idx_local)
    if links_pos.ndim == 2:
        return links_pos[idx].cpu().numpy()
    return links_pos[0, idx].cpu().numpy()


def run_diagnose() -> None:
    """Headless: compare GLB vs STL URDF link poses and print visual surface flags."""
    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu)
    stl_path = xarm6_urdf()
    glb_path = xarm6_1305_visual_glb_urdf(with_gripper_g2=False)

    def load_robot(urdf_path: str, use_glb: bool = False):
        scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
        morph = gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            requires_jac_and_IK=True,
        )
        if use_glb:
            robot = scene.add_entity(morph, surface=glb_view_surface())
        else:
            robot = scene.add_entity(morph)
        scene.build()
        return robot, scene, _link_world_positions(robot)

    _, _, stl_pos = load_robot(stl_path)
    robot, scene, glb_pos = load_robot(glb_path, use_glb=True)

    max_delta_mm = 0.0
    for link in set(stl_pos) & set(glb_pos):
        d = float(np.linalg.norm(np.array(stl_pos[link]) - np.array(glb_pos[link])) * 1000)
        max_delta_mm = max(max_delta_mm, d)

    print(f"max link pose delta STL vs GLB: {max_delta_mm:.3f} mm")
    ctx = robot.scene.visualizer.context
    pyrender_by_uid = {uid: node for uid, node in ctx.rigid_nodes.items()}
    for vg in robot.vgeoms:
        link_name = vg.link.name.split("/")[-1]
        meta = vg.metadata or {}
        mesh_path = str(meta.get("mesh_path", ""))
        if link_name not in ARM_LINKS or "visual_glb" not in mesh_path:
            continue
        surf = vg.surface
        mt = getattr(surf, "metallic_texture", None)
        rt = getattr(surf, "roughness_texture", None)
        metal_val = getattr(mt, "color", (None,))[0] if mt is not None else None
        rough_val = getattr(rt, "color", (None,))[0] if rt is not None else None
        trimesh_mat = getattr(getattr(vg.get_trimesh().visual, "material", None), "metallicFactor", None)
        py_mat = pyrender_by_uid.get(vg.uid)
        pr_metal = pr_rough = None
        if py_mat is not None:
            mat = py_mat.mesh.primitives[0].material
            pr_metal = getattr(mat, "metallicFactor", None)
            pr_rough = getattr(mat, "roughnessFactor", None)
        print(
            f"  {link_name} {meta.get('name')}: surface(m={metal_val}, r={rough_val}) "
            f"trimesh(m={trimesh_mat}) pyrender(m={pr_metal}, r={pr_rough})"
        )

    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in JOINT_NAMES if n in joint_map]
    if arm_dof_idx:
        robot.set_dofs_position(HOME_QPOS[: len(arm_dof_idx)], arm_dof_idx)
        for _ in range(5):
            scene.step()
    ee_link = resolve_robot_link(robot, EE_LINK_NAME)
    qpos = robot.get_dofs_position()
    if hasattr(qpos, "cpu"):
        qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos).reshape(-1)
    fk_pos = _fk_link6_pos(robot, ee_link, qpos)
    ee_pos = _to_numpy3(ee_link.get_pos())
    fk_delta_mm = float(np.linalg.norm(fk_pos - ee_pos) * 1000)
    print(f"link6 get_pos vs forward_kinematics: {fk_delta_mm:.4f} mm")


def main():
    parser = argparse.ArgumentParser(description="View xArm6 1305 GLB model")
    parser.add_argument("--gripper-g2", action="store_true", help="Include Gripper G2 visual URDF")
    parser.add_argument(
        "--movable",
        action="store_true",
        help="Use per-link G2 GLB visuals (required for gripper animation)",
    )
    parser.add_argument(
        "--gripper-demo",
        action="store_true",
        help="Cycle drive_joint open/close (requires --gripper-g2 --movable)",
    )
    parser.add_argument("--pd", action="store_true", help="Run simple joint PD motion demo")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--diagnose", action="store_true", help="Headless alignment diagnostic")
    parser.add_argument(
        "--show-tcp",
        action="store_true",
        help="Show red DH TCP debug marker on link6 flange (default: hidden)",
    )
    args = parser.parse_args()

    if args.movable and not args.gripper_g2:
        parser.error("--movable requires --gripper-g2")
    if args.gripper_demo and not args.movable:
        parser.error("--gripper-demo requires --movable")

    if args.diagnose:
        run_diagnose()
        return

    enable_glb_pbr_surfaces()
    urdf_path = xarm6_1305_visual_glb_urdf(with_gripper_g2=args.gripper_g2, movable=args.movable)
    print(f"Loading: {urdf_path}")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            refresh_rate=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            requires_jac_and_IK=True,
        ),
        surface=glb_view_surface(),
    )
    tcp_marker = None
    if args.show_tcp and not args.headless:
        tcp_marker = add_tcp_marker(scene)
    scene.build()

    print(f"DOFs: {robot.n_dofs}, Links: {robot.n_links}")

    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in JOINT_NAMES if n in joint_map]
    gripper_dof_idx: list[int] = []
    all_gripper_dof_idx: list[int] = []
    if "drive_joint" in joint_map:
        gripper_dof_idx = [joint_map["drive_joint"].dofs_idx_local[0]]
        all_gripper_dof_idx = [
            joint_map[n].dofs_idx_local[0] for n in ALL_GRIPPER_JOINTS if n in joint_map
        ]
    ee_link = resolve_robot_link(robot, EE_LINK_NAME)
    if tcp_marker is not None:
        print(f"TCP marker: {EE_LINK_NAME} (DH flange, no tool)")

    arm_kinematic_hold = not args.pd
    idle_gripper_kinematic_hold = not args.gripper_demo
    held_dof_idx: list[int] = []
    if arm_kinematic_hold:
        held_dof_idx.extend(arm_dof_idx)
    if idle_gripper_kinematic_hold:
        held_dof_idx.extend(all_gripper_dof_idx)
    if held_dof_idx:
        _disable_robot_pd(robot, sorted(set(held_dof_idx)))
    if arm_kinematic_hold or idle_gripper_kinematic_hold:
        _apply_kinematic_hold(
            robot,
            arm_dof_idx,
            HOME_QPOS,
            hold_arm=arm_kinematic_hold,
            hold_gripper=idle_gripper_kinematic_hold,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=[],
            all_bio_gripper_g2_dof_idx=[],
        )

    if args.pd and arm_dof_idx:
        _setup_arm_pd(robot, arm_dof_idx)
        _hold_home(robot, arm_dof_idx)
    if args.gripper_demo and gripper_dof_idx:
        _setup_gripper_pd(robot, gripper_dof_idx, all_gripper_dof_idx)
        _set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)
        print("Gripper demo: all gripper joints open/close cycle")
    warmup_steps = 3 if arm_kinematic_hold else 100
    for _ in range(warmup_steps):
        if args.pd and arm_dof_idx:
            robot.control_dofs_position(HOME_QPOS[: len(arm_dof_idx)], arm_dof_idx)
        if tcp_marker is not None:
            update_tcp_marker(tcp_marker, ee_link)
        _kinematic_step(
            scene,
            robot,
            arm_kinematic_hold=arm_kinematic_hold,
            idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
            arm_dof_idx=arm_dof_idx,
            home=HOME_QPOS,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=[],
            all_bio_gripper_g2_dof_idx=[],
        )

    if not args.headless:
        start_deferred_viewer(scene)

    if not args.pd and not args.gripper_demo:
        print("Viewer running (holding home pose). Close window or Ctrl+C to exit.")
        while True:
            if tcp_marker is not None:
                update_tcp_marker(tcp_marker, ee_link)
            _kinematic_step(
                scene,
                robot,
                arm_kinematic_hold=arm_kinematic_hold,
                idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
                arm_dof_idx=arm_dof_idx,
                home=HOME_QPOS,
                all_gripper_dof_idx=all_gripper_dof_idx,
                all_lite6_gripper_dof_idx=[],
                all_bio_gripper_g2_dof_idx=[],
            )
            time.sleep(0.01)
        return

    poses = [
        np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0]),
        np.array([0.5, -0.3, -0.1, 0.5, 0.3, 0.0]),
        np.array([-0.3, 0.2, -0.15, 0.3, -0.2, 0.1]),
        HOME_QPOS.copy(),
    ]
    if args.pd:
        print("PD motion demo (looping)...")
    elif args.gripper_demo:
        print("Gripper open/close demo (looping)...")
    step = 0
    pose_idx = 0
    hold_steps = 300
    last_gripper_phase = -1
    while True:
        if args.pd and step % hold_steps == 0:
            target = poses[pose_idx % len(poses)]
            if arm_dof_idx:
                robot.control_dofs_position(target[: len(arm_dof_idx)], arm_dof_idx)
            pose_idx += 1
            print(f"  Target pose {pose_idx % len(poses)}: {target.round(2)}")
        if args.gripper_demo and gripper_dof_idx:
            grip_phase = (step // GRIPPER_HOLD_STEPS) % 2
            if grip_phase != last_gripper_phase:
                label = "closed" if grip_phase else "open"
                print(f"  Gripper target: {label} ({GRIPPER_CLOSE if grip_phase else GRIPPER_OPEN})")
                last_gripper_phase = grip_phase
            _control_gripper_pose(
                robot,
                gripper_dof_idx,
                all_gripper_dof_idx,
                _gripper_demo_target(step),
            )
        if tcp_marker is not None:
            update_tcp_marker(tcp_marker, ee_link)
        _kinematic_step(
            scene,
            robot,
            arm_kinematic_hold=arm_kinematic_hold,
            idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
            arm_dof_idx=arm_dof_idx,
            home=HOME_QPOS,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=[],
            all_bio_gripper_g2_dof_idx=[],
        )
        step += 1
        time.sleep(0.01)


if __name__ == "__main__":
    main()
