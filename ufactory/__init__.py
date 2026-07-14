"""Minimal robot catalog and source-asset access API."""

from ufactory.config.assets import AssetLayoutError, RepositoryAssetStore
from ufactory.robots.registry import (
    PROJECT_ROOT,
    ROBOT_PROFILES,
    RobotModelSpec,
    get_robot_profile,
    robot_cli_choices,
)
from ufactory.robots.paths import robot_assets, robot_urdf, robot_visual_glb_urdf

__all__ = [
    "PROJECT_ROOT",
    "ROBOT_PROFILES",
    "RobotModelSpec",
    "AssetLayoutError",
    "RepositoryAssetStore",
    "get_robot_profile",
    "robot_assets",
    "robot_cli_choices",
    "robot_urdf",
    "robot_visual_glb_urdf",
]
