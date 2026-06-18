"""UFACTORY robot utilities for Genesis simulation."""

from ufactory.robot_registry import (
    PROJECT_ROOT,
    ROBOT_PROFILES,
    RobotModelSpec,
    arm_link_names,
    get_profile_key_for_robot_name,
    get_robot_profile,
    glb_output_name,
    joint_names,
    link_glb_stl_pairs,
)

from ufactory.paths import (
    robot_assets,
    robot_urdf,
    robot_visual_glb_urdf,
    kinematics_user_dir,
    # Per-robot convenience helpers
    xarm6_urdf,
    xarm6_1305_urdf,
    xarm6_1305_visual_glb_urdf,
    xarm5_1305_urdf,
    xarm5_1305_visual_glb_urdf,
    xarm7_1305_urdf,
    xarm7_1305_visual_glb_urdf,
    lite6_urdf,
    lite6_visual_glb_urdf,
    lite6_with_gripper_urdf,
    lite6_with_vacuum_gripper_urdf,
    uf850_urdf,
    uf850_visual_glb_urdf,
    # Gripper helpers
    bio_gripper_g2_glb,
    bio_gripper_g2_movable_visual_urdf,
    gripper_g2_static_glb,
    gripper_g2_base_glb,
    gripper_g2_shared_glb,
    gripper_g2_movable_visual_urdf,
    lite6_gripper_movable_visual_urdf,
)

from ufactory.kinematics import (
    build_calibrated_urdf,
    find_kinematics_yaml,
    has_per_unit_kinematics_calibration,
    load_kinematics_yaml,
    log_kinematics_sn_status,
    parse_sn_model_code,
    prepare_robot_model_for_verification,
    robot_name_from_firmware,
    validate_kinematics_calibration_request,
)

from ufactory.glb_visual import (
    enable_glb_pbr_surfaces,
    glb_view_surface,
)
