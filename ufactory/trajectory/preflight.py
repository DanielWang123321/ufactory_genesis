"""Application service joining runtime config, assets, and safety preflight/approval."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from ufactory.config import RepositoryAssetStore, ResolvedRuntimeConfig
from ufactory.safety import CollisionBackend, KinematicsBackend, PreflightReport, SafetyGate
from ufactory.safety.gate import sha256_file
from ufactory.trajectory.segments import Program


def load_joint_position_limits(
    urdf_path: str | Path, joint_names: tuple[str, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load exact lower/upper limits for every configured joint."""

    try:
        root = ET.parse(str(urdf_path)).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot parse URDF joint limits: {exc}") from exc
    joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_names:
        joint = joints.get(name)
        limit = None if joint is None else joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"URDF is missing finite position limits for {name}")
        try:
            lo, hi = float(limit.get("lower", "")), float(limit.get("upper", ""))
        except ValueError as exc:
            raise ValueError(f"URDF has invalid position limits for {name}") from exc
        lower.append(lo)
        upper.append(hi)
    return tuple(lower), tuple(upper)


def create_safety_gate(
    config: ResolvedRuntimeConfig,
    *,
    kinematics: KinematicsBackend | None,
    collision: CollisionBackend | None,
    calibration_sha256: str,
    scene_sha256: str,
    urdf_path: str | Path | None = None,
) -> SafetyGate:
    store = RepositoryAssetStore.discover()
    path = (
        Path(urdf_path) if urdf_path is not None else store.require(Path(config.robot.assets_dir) / config.robot.urdf)
    )
    lower, upper = load_joint_position_limits(path, config.robot.joint_names)
    return SafetyGate(
        config,
        joint_lower_rad=lower,
        joint_upper_rad=upper,
        kinematics=kinematics,
        collision=collision,
        urdf_sha256=sha256_file(path),
        calibration_sha256=calibration_sha256,
        scene_sha256=scene_sha256,
    )


def preflight_program(
    program: Program,
    config: ResolvedRuntimeConfig,
    *,
    executor: str,
    kinematics: KinematicsBackend | None,
    collision: CollisionBackend | None,
    calibration_sha256: str,
    scene_sha256: str,
    urdf_path: str | Path | None = None,
) -> PreflightReport:
    """Return the full report; issue approval separately through the same SafetyGate."""

    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=collision,
        calibration_sha256=calibration_sha256,
        scene_sha256=scene_sha256,
        urdf_path=urdf_path,
    )
    return gate.preflight(program, executor=executor)
