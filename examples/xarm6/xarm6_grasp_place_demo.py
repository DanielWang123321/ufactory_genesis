"""
xArm 6 Grasp-Place - Scripted Demo Visualization.

Demonstrates a scripted (non-RL) pick-and-place sequence:
  1. Start from home position: EE at [300, 0, 300]mm, gripper pointing down, open
  2. Approach cube at (300, 0)mm on table
  3. Close gripper to grasp
  4. Lift to z=300mm above robot base
  5. Move to place position (x=300, y=300)mm
  6. Lower and place on table
  7. Open gripper and retreat
  8. Return to exact initial position and pose

All coordinates are in mm relative to robot base (converted to meters internally).
Robot base is at world (0, 0, table_height).

Usage:
    source ~/envs/py312/bin/activate
    python examples/xarm6/xarm6_grasp_place_demo.py
"""

import math

import torch

import _bootstrap  # noqa: F401
import genesis as gs
from genesis.utils.geom import xyz_to_quat
from ufactory.paths import xarm6_urdf

XARM6_GRIPPER_URDF = xarm6_urdf("xarm6_with_gripper.urdf")
TABLE_HEIGHT = 0.4        # meters
OBJ_SIZE = [0.04, 0.04, 0.04]

# Task parameters (mm → m)
OBJ_XY = [0.30, 0.00]    # cube spawn position (world frame)
PLACE_XY = [0.30, 0.30]  # place target position (world frame)
LIFT_Z = 0.30             # 300mm above robot base
HOME_Z = 0.30             # 300mm above robot base

GRIPPER_OPEN = 0.0        # drive_joint=0 → fingers open (84mm gap)
GRIPPER_CLOSE = 0.85      # drive_joint=0.85 → fingers closed (0mm gap)


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)

    # ═══ Build scene ═══
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=8),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=50,
            camera_pos=(1.2, -1.2, 1.1),
            camera_lookat=(0.3, 0.15, TABLE_HEIGHT + 0.1),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    # Ground
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    # Table
    scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.8, TABLE_HEIGHT),
            pos=(0.45, 0.0, TABLE_HEIGHT / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6)),
        ),
    )

    # Robot (base at table height)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=XARM6_GRIPPER_URDF,
            pos=(0.0, 0.0, TABLE_HEIGHT),
            fixed=True,
            requires_jac_and_IK=True,
        ),
    )

    # Cube to grasp (red)
    obj_half_z = OBJ_SIZE[2] / 2
    obj = scene.add_entity(
        gs.morphs.Box(
            size=tuple(OBJ_SIZE),
            pos=(OBJ_XY[0], OBJ_XY[1], TABLE_HEIGHT + obj_half_z),
            fixed=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1)),
        ),
    )

    # Place target marker (green sphere)
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.02,
            pos=(PLACE_XY[0], PLACE_XY[1], TABLE_HEIGHT + obj_half_z),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
        ),
    )

    scene.build(n_envs=1)

    # ═══ Robot setup ═══
    ik_link = robot.get_link("link6")
    left_finger = robot.get_link("left_finger")
    right_finger = robot.get_link("right_finger")

    arm_dof_idx = [robot.get_joint(f"joint{i+1}").dofs_idx_local[0] for i in range(6)]
    gripper_dof_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]

    # All gripper DOFs (drive + 5 mimic) — for damping/friction overrides
    all_gripper_joints = [
        "drive_joint",
        "left_finger_joint", "left_inner_knuckle_joint",
        "right_outer_knuckle_joint", "right_finger_joint", "right_inner_knuckle_joint",
    ]
    all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in all_gripper_joints]

    # PD gains — arm
    robot.set_dofs_kp(
        torch.tensor([3000, 3000, 2000, 2000, 1000, 1000], device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor([300, 300, 200, 200, 100, 100], device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor([-50, -50, -32, -32, -32, -20], device=gs.device, dtype=gs.tc_float),
        torch.tensor([50, 50, 32, 32, 32, 20], device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )

    # PD gains — gripper (drive_joint only; mimic joints follow via equality constraints)
    robot.set_dofs_kp(
        torch.tensor([2.0], device=gs.device, dtype=gs.tc_float), gripper_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor([3.0], device=gs.device, dtype=gs.tc_float), gripper_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor([-1.0], device=gs.device, dtype=gs.tc_float),
        torch.tensor([1.0], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
    )

    # Remove URDF joint friction/damping on ALL gripper DOFs so force_range=[-1,1] is sufficient
    n_grip = len(all_gripper_dof_idx)
    robot.set_dofs_damping(
        torch.full((n_grip,), 0.1, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )
    robot.set_dofs_frictionloss(
        torch.zeros(n_grip, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )

    # Gripper-down quaternion (roll=180°)
    down_quat = xyz_to_quat(
        torch.tensor([[math.pi, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
        rpy=True, degrees=False,
    )

    # ═══ Compute home pose via IK ═══
    home_link6_pos = torch.tensor(
        [[0.3, 0.0, TABLE_HEIGHT + HOME_Z]], device=gs.device, dtype=gs.tc_float,
    )
    home_qpos_result = robot.inverse_kinematics(
        link=ik_link, pos=home_link6_pos, quat=down_quat, dofs_idx_local=arm_dof_idx,
    )

    # Build full initial qpos and save it for later restoration
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    init_qpos[:, gripper_dof_idx[0]] = GRIPPER_OPEN
    home_qpos_saved = init_qpos.clone()  # save for end-of-demo restoration

    robot.set_qpos(init_qpos)
    scene.step()

    # ═══ Measure finger geometry ═══
    link6_pos = ik_link.get_pos()[0]
    fc_pos = ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]
    lf_pos = left_finger.get_pos()[0]
    rf_pos = right_finger.get_pos()[0]

    # link6-to-finger-center offset (in world frame, gripper pointing down)
    finger_z_offset = (link6_pos[2] - fc_pos[2]).item()
    finger_y_gap = (lf_pos[1] - rf_pos[1]).abs().item()

    print(f"Link6 pos:             [{link6_pos[0]:.4f}, {link6_pos[1]:.4f}, {link6_pos[2]:.4f}]")
    print(f"Finger center pos:     [{fc_pos[0]:.4f}, {fc_pos[1]:.4f}, {fc_pos[2]:.4f}]")
    print(f"Left finger pos:       [{lf_pos[0]:.4f}, {lf_pos[1]:.4f}, {lf_pos[2]:.4f}]")
    print(f"Right finger pos:      [{rf_pos[0]:.4f}, {rf_pos[1]:.4f}, {rf_pos[2]:.4f}]")
    print(f"Link6-to-FC Z offset:  {finger_z_offset:.4f} m")
    print(f"Finger Y gap (open):   {finger_y_gap * 1000:.1f} mm")

    # Finger geometry constants (measured / URDF-derived)
    FINGER_PAD_BELOW_FC = 0.061   # finger tip is 61mm below finger center (measured)
    FINGER_CLOSE_DESCENT = 0.015  # finger tip descends ~15mm during close (URDF linkage)
    GRASP_TABLE_CLEARANCE = 0.010 # 10mm clearance above table after closing

    # ═══ Helper functions ═══

    def finger_center():
        return ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]

    def print_state(label):
        fc = finger_center()
        grip_val = robot.get_dofs_position(gripper_dof_idx)[0].item()
        obj_pos = obj.get_pos()[0]
        print(
            f"  [{label:15s}] FC: [{fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f}]  "
            f"Grip: {grip_val:.3f}  "
            f"Obj: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]"
        )

    def move_to(target_link6_pos, gripper_val, steps=100, label=""):
        """Linearly interpolate link6 from current to target over N steps."""
        target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
        start_pos = ik_link.get_pos().clone()
        grip_t = torch.tensor([[gripper_val]], device=gs.device, dtype=gs.tc_float)

        for s in range(steps):
            alpha = (s + 1) / steps
            interp = start_pos + alpha * (target_t - start_pos)
            qpos = robot.inverse_kinematics(
                link=ik_link, pos=interp, quat=down_quat, dofs_idx_local=arm_dof_idx,
            )
            robot.control_dofs_position(qpos[:, arm_dof_idx], arm_dof_idx)
            robot.control_dofs_position(grip_t, gripper_dof_idx)
            scene.step()

        if label:
            print_state(label)

    def hold(target_link6_pos, gripper_val, steps=50, label=""):
        """Hold position using joint-level control (no IK drift)."""
        target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
        grip_t = torch.tensor([[gripper_val]], device=gs.device, dtype=gs.tc_float)
        # IK solved once; subsequent steps use joint-level control
        target_qpos = robot.inverse_kinematics(
            link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
        )
        for _ in range(steps):
            robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
            robot.control_dofs_position(grip_t, gripper_dof_idx)
            scene.step()

        if label:
            print_state(label)

    def grasp_close(target_link6_pos, steps=200, label=""):
        """Gradually close gripper at a fixed arm position (single IK, no drift)."""
        target_t = torch.tensor([target_link6_pos], device=gs.device, dtype=gs.tc_float)
        target_qpos = robot.inverse_kinematics(
            link=ik_link, pos=target_t, quat=down_quat, dofs_idx_local=arm_dof_idx,
        )
        for s in range(steps):
            alpha = (s + 1) / steps
            robot.control_dofs_position(target_qpos[:, arm_dof_idx], arm_dof_idx)
            grip_val = GRIPPER_OPEN + alpha * (GRIPPER_CLOSE - GRIPPER_OPEN)
            robot.control_dofs_position(
                torch.tensor([[grip_val]], device=gs.device, dtype=gs.tc_float),
                gripper_dof_idx,
            )
            scene.step()

        if label:
            print_state(label)

    def restore_home(steps=150):
        """Smoothly restore the exact initial qpos (position-level, not IK)."""
        start_qpos = robot.get_dofs_position().clone()
        target_qpos = home_qpos_saved.clone()

        for s in range(steps):
            alpha = (s + 1) / steps
            interp = start_qpos + alpha * (target_qpos - start_qpos)
            robot.control_dofs_position(interp[:, [*arm_dof_idx]], arm_dof_idx)
            robot.control_dofs_position(
                interp[:, gripper_dof_idx], gripper_dof_idx,
            )
            scene.step()

        # Settle at exact home qpos
        for _ in range(50):
            robot.control_dofs_position(target_qpos[:, [*arm_dof_idx]], arm_dof_idx)
            robot.control_dofs_position(
                target_qpos[:, gripper_dof_idx], gripper_dof_idx,
            )
            scene.step()

    # ═══ Compute key heights ═══
    # Grasp height: ensure closed finger tips stay above table with clearance.
    #   closed tip Z  = TABLE_HEIGHT + GRASP_TABLE_CLEARANCE
    #   open tip Z    = closed tip Z + FINGER_CLOSE_DESCENT
    #   FC Z          = open tip Z   + FINGER_PAD_BELOW_FC
    #   link6 Z       = FC Z         + finger_z_offset
    grasp_link6_z = (TABLE_HEIGHT + GRASP_TABLE_CLEARANCE
                     + FINGER_CLOSE_DESCENT
                     + FINGER_PAD_BELOW_FC
                     + finger_z_offset)

    # Pre-grasp: 10cm above the grasp point
    pre_grasp_link6_z = grasp_link6_z + 0.10

    # Lift height: link6 at table_height + LIFT_Z
    lift_link6_z = TABLE_HEIGHT + LIFT_Z

    home_pos = [0.3, 0.0, TABLE_HEIGHT + HOME_Z]

    print(f"\nComputed heights (link6 target Z):")
    print(f"  Grasp:      {grasp_link6_z:.4f} (table clearance {GRASP_TABLE_CLEARANCE*1000:.0f}mm)")
    print(f"  Pre-grasp:  {pre_grasp_link6_z:.4f}")
    print(f"  Lift:       {lift_link6_z:.4f}")

    # ═══ Scripted pick-and-place sequence ═══
    print("\n" + "=" * 60)
    print("  xArm6 Scripted Pick-and-Place Demo")
    print("=" * 60)

    # Phase 1: Settle at home
    print("\n[Phase 1] Home position")
    hold(home_pos, GRIPPER_OPEN, steps=30, label="Home")

    # Phase 2: Move above cube (high approach)
    print("\n[Phase 2] Move above cube")
    move_to(
        [OBJ_XY[0], OBJ_XY[1], pre_grasp_link6_z],
        GRIPPER_OPEN, steps=100, label="Pre-grasp",
    )

    # Phase 3: Descend slowly to grasp height
    print("\n[Phase 3] Descend to grasp height")
    move_to(
        [OBJ_XY[0], OBJ_XY[1], grasp_link6_z],
        GRIPPER_OPEN, steps=120, label="At cube",
    )

    # Phase 4: Settle briefly before closing (let physics stabilize)
    print("\n[Phase 4] Settle at grasp position")
    hold(
        [OBJ_XY[0], OBJ_XY[1], grasp_link6_z],
        GRIPPER_OPEN, steps=30, label="Settled",
    )

    # Phase 5: Close gripper to grasp
    print("\n[Phase 5] Close gripper (grasp)")
    grasp_close(
        [OBJ_XY[0], OBJ_XY[1], grasp_link6_z],
        steps=200, label="Grasped",
    )

    # Phase 6: Lift to z=300mm above base
    print("\n[Phase 6] Lift to z=300mm")
    move_to(
        [OBJ_XY[0], OBJ_XY[1], lift_link6_z],
        GRIPPER_CLOSE, steps=120, label="Lifted",
    )

    # Phase 7: Move to place position (300, 300)mm at lift height
    print("\n[Phase 7] Move to place position (x=300, y=300)")
    move_to(
        [PLACE_XY[0], PLACE_XY[1], lift_link6_z],
        GRIPPER_CLOSE, steps=150, label="Above target",
    )

    # Phase 8: Lower to table
    print("\n[Phase 8] Lower to table")
    move_to(
        [PLACE_XY[0], PLACE_XY[1], grasp_link6_z],
        GRIPPER_CLOSE, steps=100, label="At table",
    )

    # Phase 9: Open gripper (release)
    print("\n[Phase 9] Release object")
    hold(
        [PLACE_XY[0], PLACE_XY[1], grasp_link6_z],
        GRIPPER_OPEN, steps=150, label="Released",
    )

    # Phase 10: Retreat upward
    print("\n[Phase 10] Retreat upward")
    move_to(
        [PLACE_XY[0], PLACE_XY[1], lift_link6_z],
        GRIPPER_OPEN, steps=80, label="Retreated",
    )

    # Phase 11: Return to exact initial position and pose
    print("\n[Phase 11] Restore initial position and pose")
    restore_home(steps=150)
    print_state("Home restored")

    # ─── Final report ───
    final_obj_pos = obj.get_pos()[0]
    target_world = torch.tensor(
        [PLACE_XY[0], PLACE_XY[1], TABLE_HEIGHT + obj_half_z],
        device=gs.device, dtype=gs.tc_float,
    )
    dist_to_target = torch.norm(final_obj_pos - target_world).item()

    final_link6 = ik_link.get_pos()[0]
    home_link6_expected = torch.tensor(home_pos, device=gs.device, dtype=gs.tc_float)
    home_error = torch.norm(final_link6 - home_link6_expected).item()

    print("\n" + "=" * 60)
    print(f"  Object final pos:    [{final_obj_pos[0]:.3f}, {final_obj_pos[1]:.3f}, {final_obj_pos[2]:.3f}]")
    print(f"  Place target:        [{PLACE_XY[0]:.3f}, {PLACE_XY[1]:.3f}, {TABLE_HEIGHT + obj_half_z:.3f}]")
    print(f"  Place error:         {dist_to_target * 1000:.1f} mm  {'OK' if dist_to_target < 0.05 else 'MISS'}")
    print(f"  Link6 final pos:     [{final_link6[0]:.3f}, {final_link6[1]:.3f}, {final_link6[2]:.3f}]")
    print(f"  Home target:         [{home_pos[0]:.3f}, {home_pos[1]:.3f}, {home_pos[2]:.3f}]")
    print(f"  Home error:          {home_error * 1000:.1f} mm  {'OK' if home_error < 0.01 else 'DRIFT'}")
    print("=" * 60)

    # Keep viewer open
    print("\nDemo complete. Press Ctrl+C to exit viewer...")
    try:
        while True:
            hold(home_pos, GRIPPER_OPEN, steps=50)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
