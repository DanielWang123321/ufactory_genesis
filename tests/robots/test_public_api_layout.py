"""Public import layout checks for the v0.2 modular package structure."""

from __future__ import annotations

import importlib

import pytest


CANONICAL_MODULES = (
    "ufactory",
    "ufactory.robots",
    "ufactory.robots.paths",
    "ufactory.robots.runtime",
    "ufactory.kinematics",
    "ufactory.kinematics.calibration",
    "ufactory.kinematics.validation",
    "ufactory.kinematics.genesis",
    "ufactory.dynamics",
    "ufactory.hardware",
    "ufactory.hardware.xarm",
    "ufactory.hardware.session",
    "ufactory.hardware.observe",
    "ufactory.grippers",
    "ufactory.grippers.g2",
    "ufactory.grippers.bio_g2",
    "ufactory.trajectory",
    "ufactory.manipulation",
    "ufactory.visualization",
    "ufactory.deploy",
)

REMOVED_MODULES = (
    "ufactory.paths",
    "ufactory.robot_registry",
    "ufactory.robot_params",
    "ufactory.kinematics_validation",
    "ufactory.real_robot_session",
    "ufactory.xarm_control",
    "ufactory.gripper_g2",
    "ufactory.bio_gripper_g2",
    "ufactory.glb_visual",
    "ufactory.dynamics_validation",
    "ufactory.dynamics_static_analysis",
    "ufactory.dynamics_verify",
)


@pytest.mark.parametrize("module_name", CANONICAL_MODULES)
def test_canonical_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_legacy_modules_do_not_import(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_root_namespace_is_core_robot_api_only() -> None:
    import ufactory

    for name in (
        "DynamicsRunConfig",
        "BioGripperG2",
        "enable_glb_pbr_surfaces",
        "MODE_POSITION",
        "build_calibrated_urdf",
    ):
        assert not hasattr(ufactory, name)

    for name in (
        "ROBOT_PROFILES",
        "RobotModelSpec",
        "get_robot_profile",
        "RepositoryAssetStore",
        "robot_urdf",
        "robot_visual_glb_urdf",
    ):
        assert hasattr(ufactory, name)

    for name in ("RobotRuntimeProfile", "get_robot_runtime_profile"):
        assert not hasattr(ufactory, name)


def test_v025_trajectory_api_does_not_export_unsafe_legacy_replay() -> None:
    import ufactory.trajectory as trajectory

    assert set(trajectory.__all__) == {
        "plan_mixed_waypoints",
        "preflight_program",
        "execute_sim",
        "execute_real",
    }
    for name in ("replay_real", "replay_sim", "RealExecutorConfig", "SafetyGuard"):
        assert not hasattr(trajectory, name)


def test_v025_dynamics_api_is_report_and_validation_only() -> None:
    import ufactory.dynamics as dynamics

    assert "DynamicsValidationService" in dynamics.__all__
    for name in ("cli_hardware_check", "build_genesis_scene", "test_pd_step_response"):
        assert not hasattr(dynamics, name)
