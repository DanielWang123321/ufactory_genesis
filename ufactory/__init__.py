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
    robot_cli_choices,
)

from ufactory.paths import (
    robot_assets,
    robot_urdf,
    robot_visual_glb_urdf,
    kinematics_user_dir,
    # Per-robot convenience helpers
    xarm5_urdf,
    xarm6_urdf,
    xarm7_urdf,
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

from ufactory.robot_params import (
    ArmControlParams,
    DynamicsValidationParams,
    GripperControlParams,
    RobotRuntimeProfile,
    TaskProfile,
    get_robot_runtime_profile,
    robot_runtime_cli_choices,
)

from ufactory.kinematics_validation import (
    angle_diff_deg,
    cli_fk,
    cli_ik,
    quat_to_rpy,
    rpy_to_quat,
    validation_configs,
)

from ufactory.dynamics_validation import (
    ABS_ERR_LIMITS,
    DYNAMICS_EXTRA_CONFIGS,
    XARM6_DEFAULT_DYNAMICS_CONFIGS,
    XARM6_STRESS_DYNAMICS_CONFIGS,
    DynamicsRunConfig,
    DynamicsSample,
    GenesisDynamicsSample,
    L2_ERR_LIMIT,
    PinocchioReference,
    REL_ERR_LIMIT,
    URDF_JOINT_EFFORT,
    UrdfDynamicsIssue,
    ValidationStatus,
    SafePose,
    TorqueCompareResult,
    build_dynamics_sample,
    build_genesis_scene,
    check_genesis_path_z,
    check_joint_limit_path,
    classify_torque_result,
    compare_torques,
    compute_ee_z_table_from_sim,
    filter_safe_configs,
    format_torque_row,
    genesis_ee_z_mm_at_q,
    genesis_gravity_torque_at_q,
    genesis_pd_hold_torque_at_q,
    load_reference_backend,
    merge_test_configs,
    parse_joint_limits,
    read_report_records,
    set_pd_gains,
    validate_urdf_dynamics,
    write_csv_report,
    write_jsonl_report,
    dynamics_default_configs,
    xarm6_default_dynamics_configs,
)

from ufactory.xarm_control import (
    MODE_CART_VEL,
    MODE_JOINT_VEL,
    MODE_POSITION,
    MODE_SERVO,
    STATE_MOTION,
    STATE_STOP,
    assert_motion_ready,
    format_arm_status,
    prepare_arm_for_motion,
)

from ufactory.glb_visual import (
    enable_glb_pbr_surfaces,
    glb_view_surface,
)

from ufactory.bio_gripper_g2 import BioGripperG2
