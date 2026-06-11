"""
View xArm6 1305 GLB visual model in Genesis.

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    python examples/xarm6/view_xarm6_glb.py              # arm only
    python examples/xarm6/view_xarm6_glb.py --g2         # arm + Gripper G2
    python examples/xarm6/view_xarm6_glb.py --g2 --pd    # with simple joint motion demo
    python examples/xarm6/view_xarm6_glb.py --diagnose   # headless alignment check
    python examples/xarm6/view_xarm6_glb.py --no-show-tcp  # hide DH TCP marker
"""

import argparse
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf, xarm6_urdf

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
EE_LINK_NAME = "link6"
TCP_MARKER_RADIUS = 0.008
PD_KP = [3000, 3000, 2000, 2000, 1000, 1000]
PD_KV = [300, 300, 200, 200, 100, 100]
FORCE_LOWER = [-50, -50, -32, -32, -32, -20]
FORCE_UPPER = [50, 50, 32, 32, 32, 20]
ARM_LINKS = ("link_base", "link1", "link2", "link3", "link4", "link5", "link6")
HOME_QPOS = np.zeros(6)


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


def _resolve_link(robot, name: str):
    available = {link.name.split("/")[-1]: link for link in robot.links}
    if name not in available:
        raise KeyError(f"Link not found: {name}. Available: {sorted(available)}")
    return available[name]


def _add_tcp_marker(scene):
    """Red sphere at DH TCP (link6 flange); visual only, no collision."""
    return scene.add_entity(
        gs.morphs.Sphere(
            radius=TCP_MARKER_RADIUS,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
        ),
    )


def _update_tcp_marker(marker, ee_link) -> None:
    marker.set_pos(ee_link.get_pos())


def _fk_link6_pos(robot, ee_link, qpos_np: np.ndarray) -> np.ndarray:
    qpos_t = torch.tensor(qpos_np, dtype=torch.float32, device=gs.device)
    links_pos, _ = robot.forward_kinematics(qpos=qpos_t)
    idx = int(ee_link.idx_local)
    if links_pos.ndim == 2:
        return links_pos[idx].cpu().numpy()
    return links_pos[0, idx].cpu().numpy()


def _to_numpy3(pos) -> np.ndarray:
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    return np.asarray(pos).reshape(-1)[:3]


def run_diagnose() -> None:
    """Headless: compare GLB vs STL URDF link poses and print visual surface flags."""
    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu)
    stl_path = xarm6_urdf("xarm6_1305.urdf")
    glb_path = xarm6_1305_visual_glb_urdf(with_g2=False)

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
    ee_link = _resolve_link(robot, EE_LINK_NAME)
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
    parser.add_argument("--g2", action="store_true", help="Include Gripper G2 visual URDF")
    parser.add_argument("--pd", action="store_true", help="Run simple joint PD motion demo")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--diagnose", action="store_true", help="Headless alignment diagnostic")
    parser.add_argument(
        "--no-show-tcp",
        action="store_true",
        help="Hide red DH TCP marker (link6 flange, default: shown)",
    )
    args = parser.parse_args()

    if args.diagnose:
        run_diagnose()
        return

    enable_glb_pbr_surfaces()
    urdf_path = xarm6_1305_visual_glb_urdf(with_g2=args.g2)
    print(f"Loading: {urdf_path}")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=not args.headless,
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
    if not args.no_show_tcp:
        tcp_marker = _add_tcp_marker(scene)
    scene.build()

    print(f"DOFs: {robot.n_dofs}, Links: {robot.n_links}")

    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in JOINT_NAMES if n in joint_map]
    ee_link = _resolve_link(robot, EE_LINK_NAME)
    if tcp_marker is not None:
        print(f"TCP marker: {EE_LINK_NAME} (DH flange, no tool)")

    if arm_dof_idx:
        _setup_arm_pd(robot, arm_dof_idx)
        _hold_home(robot, arm_dof_idx)
        for _ in range(100):
            if tcp_marker is not None:
                _update_tcp_marker(tcp_marker, ee_link)
            scene.step()

    if not args.pd:
        print("Viewer running (holding home pose). Close window or Ctrl+C to exit.")
        while True:
            if arm_dof_idx:
                robot.control_dofs_position(HOME_QPOS[: len(arm_dof_idx)], arm_dof_idx)
            if tcp_marker is not None:
                _update_tcp_marker(tcp_marker, ee_link)
            scene.step()
            time.sleep(0.01)
        return

    poses = [
        np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0]),
        np.array([0.5, -0.3, -0.1, 0.5, 0.3, 0.0]),
        np.array([-0.3, 0.2, -0.15, 0.3, -0.2, 0.1]),
        HOME_QPOS.copy(),
    ]
    print("PD motion demo (looping)...")
    step = 0
    pose_idx = 0
    hold_steps = 300
    while True:
        if step % hold_steps == 0:
            target = poses[pose_idx % len(poses)]
            if arm_dof_idx:
                robot.control_dofs_position(target[: len(arm_dof_idx)], arm_dof_idx)
            pose_idx += 1
            print(f"  Target pose {pose_idx % len(poses)}: {target.round(2)}")
        if tcp_marker is not None:
            _update_tcp_marker(tcp_marker, ee_link)
        scene.step()
        step += 1
        time.sleep(0.01)


if __name__ == "__main__":
    main()
