"""Kinematic calibration helpers for xArm URDF patching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from ufactory.robots.paths import PROJECT_ROOT, kinematics_user_dir, xarm6_urdf
from ufactory.robots.registry import get_robot_profile

DEFAULT_XARM6_URDF = xarm6_urdf()

# SN positions 3-6 (1-based) = four-digit model code, e.g. XI130506... -> 1305
XARM_KINEMATICS_MIN_SN_MODEL_CODE = 1304  # xarm5/6/7: code < 1304 => no compensation
LITE6_KINEMATICS_MIN_SN_MODEL_CODE = 1006  # lite6: code < 1006 => no compensation
# UF850: all units have per-unit kinematics compensation in firmware

DEFAULT_KINEMATICS_SUFFIX_ENV = "XARM_KINEMATICS_SUFFIX"

_CALIBRATION_FIELDS = frozenset(("x", "y", "z", "roll", "pitch", "yaw"))


@dataclass(frozen=True)
class KinematicsCalibration:
    """Strictly validated, robot-bound kinematic calibration."""

    schema_version: int
    robot_key: str
    serial_number: str
    position_unit: str
    angle_unit: str
    joints: dict[str, dict[str, float]]
    source_path: Path
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_joint_mapping(
    joints: Mapping[str, object],
    *,
    joint_names: tuple[str, ...],
    source: str,
) -> dict[str, dict[str, float]]:
    expected = set(joint_names)
    actual = set(map(str, joints))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{source}: calibration joint set mismatch; missing={missing}, extra={extra}")
    result: dict[str, dict[str, float]] = {}
    for joint_name in joint_names:
        raw = joints[joint_name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: {joint_name} must be a mapping")
        fields = set(map(str, raw))
        if fields != _CALIBRATION_FIELDS:
            raise ValueError(
                f"{source}: {joint_name} fields must be exactly {sorted(_CALIBRATION_FIELDS)}, got {sorted(fields)}"
            )
        values = {name: float(raw[name]) for name in _CALIBRATION_FIELDS}
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{source}: {joint_name}.{name} must be finite")
            bound = 2.0 if name in {"x", "y", "z"} else 2.0 * math.pi
            if abs(value) > bound:
                unit = "m" if name in {"x", "y", "z"} else "rad"
                raise ValueError(f"{source}: {joint_name}.{name}={value} {unit} exceeds reasonable ±{bound}")
        result[joint_name] = values
    return result


def load_kinematics_calibration(
    kinematics_yaml_path: str,
    *,
    robot_key: str,
    serial_number: str | None = None,
    joint_names: tuple[str, ...] | None = None,
) -> KinematicsCalibration:
    """Load schema-v1 calibration with exact robot, serial, units and joints."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load kinematics YAML.") from e

    profile = get_robot_profile(robot_key)
    expected_joints = joint_names or tuple(f"joint{i}" for i in range(1, profile.dof + 1))
    yaml_path = Path(kinematics_yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Kinematics YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid kinematics YAML root: {yaml_path}")
    expected_root = {"schema_version", "robot_key", "serial_number", "units", "joints"}
    if set(map(str, data)) != expected_root:
        actual_root = set(map(str, data))
        if actual_root == {"kinematics"}:
            raise ValueError(
                f"{yaml_path}: legacy calibration schema with root 'kinematics' is not accepted; "
                "regenerate this robot-bound file with "
                "'python scripts/gen_kinematics_params.py <robot-ip>' so it includes "
                "schema_version, robot_key, full serial_number, units and joints"
            )
        raise ValueError(f"{yaml_path}: root fields must be exactly {sorted(expected_root)}, got {sorted(actual_root)}")
    if data["schema_version"] != 1:
        raise ValueError(f"{yaml_path}: schema_version must be 1")
    if str(data["robot_key"]) != profile.key:
        raise ValueError(f"{yaml_path}: robot_key {data['robot_key']!r} does not match {profile.key!r}")
    file_serial = str(data["serial_number"]).strip()
    if file_serial != "NOMINAL" and len(file_serial) < 8:
        raise ValueError(f"{yaml_path}: serial_number must be complete, not a short suffix")
    if serial_number is not None and file_serial != str(serial_number).strip():
        raise ValueError(f"{yaml_path}: serial number does not match connected robot")
    units = data["units"]
    if not isinstance(units, Mapping) or set(map(str, units)) != {"position", "angle"}:
        raise ValueError(f"{yaml_path}: units must contain exactly position and angle")
    if units["position"] != "m" or units["angle"] != "rad":
        raise ValueError(f"{yaml_path}: only position=m and angle=rad are accepted")
    joints = data["joints"]
    if not isinstance(joints, Mapping):
        raise ValueError(f"{yaml_path}: joints must be a mapping")
    validated = _validate_joint_mapping(joints, joint_names=expected_joints, source=str(yaml_path))
    return KinematicsCalibration(
        schema_version=1,
        robot_key=profile.key,
        serial_number=file_serial,
        position_unit="m",
        angle_unit="rad",
        joints=validated,
        source_path=yaml_path,
        sha256=_sha256(yaml_path),
    )


def load_kinematics_yaml(
    kinematics_yaml_path: str,
    joint_count: int | None = None,
) -> dict[str, dict[str, float]]:
    """Load strict schema-v1 joint offsets (deprecated narrow convenience API).

    The robot is inferred only from the exact ``<robot>_...`` filename prefix;
    ambiguous filenames are rejected.  New code should use
    :func:`load_kinematics_calibration` to also retain identity and hashes.
    """
    path = Path(kinematics_yaml_path).expanduser().resolve()
    candidates = [name for name in ("xarm5", "xarm6", "xarm7", "uf850", "lite6") if path.name.startswith(f"{name}_")]
    if len(candidates) != 1:
        raise ValueError(f"cannot infer exact robot from calibration filename: {path.name}")
    profile = get_robot_profile(candidates[0])
    if joint_count is not None and int(joint_count) != profile.dof:
        raise ValueError(f"joint_count {joint_count} does not match {profile.key} DOF {profile.dof}")
    return load_kinematics_calibration(str(path), robot_key=profile.key).joints


def find_kinematics_yaml(
    kinematics_suffix: str,
    kinematics_yaml_dir: str | None = None,
    robot_name: str = "xarm6",
) -> Path:
    """Find a kinematics yaml file from a suffix (e.g., XXXXXX -> xarm6_kinematics_XXXXXX.yaml)."""
    suffix = (kinematics_suffix or "").strip()
    if not suffix:
        raise ValueError("kinematics_suffix is empty")

    profile = get_robot_profile(robot_name)
    prefix = profile.kinematics_prefix

    search_dirs = []
    if kinematics_yaml_dir:
        search_dirs.append(Path(kinematics_yaml_dir).expanduser())
    search_dirs.append(kinematics_user_dir(profile.robot_name))
    exact_name = f"{prefix}_kinematics_{suffix}.yaml"
    for root in search_dirs:
        if not root.exists():
            continue
        candidate = root / exact_name
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find kinematics YAML for {robot_name} suffix '{suffix}'. "
        f"Run: python scripts/gen_kinematics_params.py <robot_ip> {suffix}"
    )


def user_kinematics_yaml_exists(
    robot_name: str,
    kinematics_suffix: str,
    *,
    kinematics_yaml_dir: str | None = None,
) -> bool:
    """Return True when a user kinematics YAML for ``kinematics_suffix`` is on disk."""
    try:
        find_kinematics_yaml(kinematics_suffix, kinematics_yaml_dir, robot_name=robot_name)
    except (FileNotFoundError, ValueError):
        return False
    return True


def sn_matching_user_kinematics_yaml_exists(
    sn: str,
    robot_name: str,
    *,
    kinematics_suffix: str | None = None,
    kinematics_yaml_dir: str | None = None,
) -> bool:
    """True when suffix matches SN last-6 and the corresponding user YAML exists."""
    try:
        sn_suffix = kinematics_suffix_from_sn(sn)
    except ValueError:
        return False
    suffix = (kinematics_suffix or sn_suffix).strip()
    if suffix != sn_suffix:
        return False
    return user_kinematics_yaml_exists(robot_name, suffix, kinematics_yaml_dir=kinematics_yaml_dir)


def build_calibrated_urdf(
    base_urdf_path: str,
    kinematics: KinematicsCalibration | dict[str, dict[str, float]],
    suffix: str | None = None,
    joint_count: int | None = None,
    output_dir: str | None = None,
) -> str:
    """Generate a patched URDF with calibrated joint origins."""
    base = Path(base_urdf_path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base URDF not found: {base}")

    safe_suffix = "calib"
    if suffix:
        safe_suffix = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in suffix) or "calib"

    joint_names = tuple(
        f"joint{i}"
        for i in range(
            1,
            (joint_count or len(kinematics.joints if isinstance(kinematics, KinematicsCalibration) else kinematics))
            + 1,
        )
    )
    values = (
        kinematics.joints
        if isinstance(kinematics, KinematicsCalibration)
        else _validate_joint_mapping(kinematics, joint_names=joint_names, source="kinematics argument")
    )
    calibration_sha256 = (
        kinematics.sha256
        if isinstance(kinematics, KinematicsCalibration)
        else hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()
    )
    digest_src = {
        "base": str(base),
        "base_sha256": _sha256(base),
        "calibration_sha256": calibration_sha256,
        "kinematics": values,
        "joint_count": joint_count,
    }
    digest = hashlib.sha256(json.dumps(digest_src, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    if output_dir is None:
        out_dir = PROJECT_ROOT / ".cache" / "ufactory" / "urdf"
    else:
        out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{base.stem}_{safe_suffix}_{digest}_calib.urdf"
    tree = ET.parse(str(base))
    root = tree.getroot()

    # Cached URDFs live outside the asset directory, so relative mesh paths must
    # remain anchored to the original URDF location.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename or filename.startswith(("package://", "file://")):
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh.set("filename", str((base.parent / mesh_path).resolve()))

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    n = joint_count or len(values)
    for i in range(1, n + 1):
        joint_name = f"joint{i}"
        target = joints.get(joint_name)
        if target is None:
            raise ValueError(f"base URDF is missing calibrated joint {joint_name}")

        cfg = values[joint_name]
        x = cfg["x"]
        y = cfg["y"]
        z = cfg["z"]
        roll = cfg["roll"]
        pitch = cfg["pitch"]
        yaw = cfg["yaw"]

        origin = target.find("origin")
        if origin is None:
            origin = ET.Element("origin")
            target.insert(0, origin)
        origin.set("xyz", f"{x} {y} {z}")
        origin.set("rpy", f"{roll} {pitch} {yaw}")

    try:
        ET.indent(tree)
    except AttributeError:
        pass
    tree.write(str(output_path), encoding="utf-8", xml_declaration=False)
    output_sha256 = _sha256(output_path)
    manifest = {
        "schema_version": 1,
        "base_urdf": str(base),
        "base_urdf_sha256": _sha256(base),
        "calibration_sha256": calibration_sha256,
        "output_urdf": str(output_path),
        "output_urdf_sha256": output_sha256,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(output_path)


def prepare_robot_model_for_verification(
    robot_model: str | None,
    kinematics_yaml: str | None,
    kinematics_suffix: str | None,
    kinematics_yaml_dir: str | None = None,
    default_base_urdf: str | None = None,
    robot_name: str = "xarm6",
    joint_count: int | None = None,
    output_dir: str | None = None,
    serial_number: str | None = None,
) -> tuple[str, str | None]:
    """Resolve robot model and apply kinematic calibration if requested.

    Returns (urdf_path, kinematics_yaml_path_or_none).
    """
    profile = get_robot_profile(robot_name)
    dof = joint_count or profile.dof
    base_default = default_base_urdf or str(profile.assets_dir / profile.default_urdf)
    model_path = Path(robot_model).expanduser().resolve() if robot_model else Path(base_default)

    if kinematics_yaml is None and kinematics_suffix is None:
        return str(model_path), None

    if kinematics_yaml is not None:
        yaml_path = Path(kinematics_yaml).expanduser().resolve()
    else:
        yaml_path = find_kinematics_yaml(kinematics_suffix, kinematics_yaml_dir, robot_name=profile.robot_name)

    calibration = load_kinematics_calibration(
        str(yaml_path),
        robot_key=profile.key,
        serial_number=serial_number,
        joint_names=tuple(f"joint{i}" for i in range(1, dof + 1)),
    )
    calibrated = build_calibrated_urdf(
        str(model_path),
        calibration,
        suffix=kinematics_suffix or yaml_path.stem,
        joint_count=dof,
        output_dir=output_dir,
    )
    return calibrated, str(yaml_path)


def parse_sn_model_code(sn: str) -> int | None:
    """Parse the 4-digit model code from SN positions 3-6 (1-based).

    Example: ``XI130506XXXXXX`` -> ``1305``.
    """
    digits = (sn or "").strip().upper()[2:6]
    return int(digits) if len(digits) == 4 and digits.isdigit() else None


def robot_name_from_firmware(robot_dof: int, robot_type: int) -> str:
    """Map control-box firmware identifiers to kinematics robot_name."""
    if robot_dof == 6 and robot_type == 12:
        return "uf850"
    if robot_dof == 6 and robot_type == 9:
        return "lite6"
    return f"xarm{robot_dof}"


def has_per_unit_kinematics_calibration(sn: str, robot_name: str) -> bool:
    """Return whether firmware may provide per-unit kinematic compensation.

    - xArm 5/6/7: SN model code < 1304 => definitely **no** compensation.
    - Lite6: SN model code < 1006 => definitely **no** compensation.
    - UF850: all models have compensation.
    - Unparseable SN: returns True (cannot rule out compensation).
    """
    if robot_name == "uf850":
        return True

    model_code = parse_sn_model_code(sn)
    if model_code is None:
        return True

    if robot_name == "lite6":
        return model_code >= LITE6_KINEMATICS_MIN_SN_MODEL_CODE
    if robot_name in ("xarm5", "xarm6", "xarm7"):
        return model_code >= XARM_KINEMATICS_MIN_SN_MODEL_CODE
    return True


def get_robot_sn(arm) -> str:
    """Read robot SN from an XArmAPI instance."""
    code, sn = arm.get_robot_sn()
    if code == 0 and sn:
        return str(sn).strip()
    fallback = getattr(arm, "sn", None)
    return str(fallback).strip() if fallback else ""


def kinematics_suffix_from_sn(sn: str) -> str:
    """Return default kinematics YAML suffix: last 6 characters of SN (case preserved)."""
    normalized = (sn or "").strip()
    if len(normalized) < 6:
        raise ValueError(f"SN too short for kinematics suffix: {sn!r}")
    return normalized[-6:]


def resolve_kinematics_suffix(
    *,
    kinematics_suffix: str | None = None,
    kinematics_yaml: str | None = None,
    sn: str | None = None,
    robot_name: str | None = None,
    env_suffix: str | None = None,
    kinematics_yaml_dir: str | None = None,
) -> str | None:
    """Resolve kinematics suffix from CLI, env, or SN (when eligible).

    When the SN rule says no factory compensation, still auto-select the SN
    suffix if a matching user YAML already exists on disk (post-factory POE).
    """
    if kinematics_yaml is not None:
        explicit = (kinematics_suffix or "").strip() or None
        return explicit

    explicit = (kinematics_suffix or "").strip() or None
    if explicit:
        return explicit

    env_value = (env_suffix if env_suffix is not None else os.environ.get(DEFAULT_KINEMATICS_SUFFIX_ENV) or "").strip()
    if env_value:
        return env_value

    if not sn or not robot_name:
        return None

    if has_per_unit_kinematics_calibration(sn, robot_name):
        return kinematics_suffix_from_sn(sn)

    if sn_matching_user_kinematics_yaml_exists(
        sn,
        robot_name,
        kinematics_yaml_dir=kinematics_yaml_dir,
    ):
        return kinematics_suffix_from_sn(sn)
    return None


def fetch_robot_sn_from_ip(ip: str) -> str:
    """Connect briefly to the control box and read the robot SN."""
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    try:
        connect = getattr(arm, "connect", None)
        if connect is not None:
            connect()
        if not arm.connected:
            raise RuntimeError(f"cannot connect to {ip}")
        sn = get_robot_sn(arm)
        if not sn:
            raise RuntimeError(f"empty SN from {ip}")
        return sn
    finally:
        disconnect = getattr(arm, "disconnect", None)
        if disconnect is not None:
            disconnect()


def resolve_kinematics_suffix_from_ip(
    ip: str,
    robot_name: str,
    *,
    kinematics_suffix: str | None = None,
    kinematics_yaml: str | None = None,
    env_suffix: str | None = None,
    kinematics_yaml_dir: str | None = None,
) -> tuple[str | None, str]:
    """Resolve suffix using robot IP; returns ``(suffix, sn)``."""
    sn = fetch_robot_sn_from_ip(ip)
    suffix = resolve_kinematics_suffix(
        kinematics_suffix=kinematics_suffix,
        kinematics_yaml=kinematics_yaml,
        sn=sn,
        robot_name=robot_name,
        env_suffix=env_suffix,
        kinematics_yaml_dir=kinematics_yaml_dir,
    )
    return suffix, sn


def validate_kinematics_calibration_request(
    sn: str,
    robot_name: str,
    *,
    kinematics_yaml: str | None = None,
    kinematics_suffix: str | None = None,
    allow_sn_override: bool = False,
    kinematics_yaml_dir: str | None = None,
) -> None:
    """Raise ValueError if calibration files are requested but SN rules them out."""
    wants_calib = kinematics_yaml is not None or kinematics_suffix is not None
    if not wants_calib or has_per_unit_kinematics_calibration(sn, robot_name):
        return

    model_code = parse_sn_model_code(sn)
    code_str = str(model_code) if model_code is not None else "????"
    if robot_name == "lite6":
        rule = f"Lite6 SN model code {code_str} < {LITE6_KINEMATICS_MIN_SN_MODEL_CODE}"
    else:
        rule = f"xArm SN model code {code_str} < {XARM_KINEMATICS_MIN_SN_MODEL_CODE}"

    if allow_sn_override:
        print(
            f"[WARN] {rule}: SN rule overridden; using requested kinematics YAML/suffix anyway "
            "(e.g. POE calibration written to firmware after factory)."
        )
        return

    if kinematics_suffix and sn_matching_user_kinematics_yaml_exists(
        sn,
        robot_name,
        kinematics_suffix=kinematics_suffix,
        kinematics_yaml_dir=kinematics_yaml_dir,
    ):
        yaml_path = find_kinematics_yaml(
            kinematics_suffix,
            kinematics_yaml_dir,
            robot_name=robot_name,
        )
        print(
            f"[WARN] {rule}: SN rule does not expect factory compensation, but found user YAML "
            f"{yaml_path}; loading it anyway (e.g. POE / after-sales calibration)."
        )
        return

    raise ValueError(
        f"{rule}: this unit has no per-unit kinematics compensation in firmware. "
        "Do not pass --kinematics-suffix/--kinematics-yaml; use the nominal URDF only. "
        "Pass --force / --force-kinematics to override after verifying exported YAML."
    )


def log_kinematics_sn_status(
    sn: str,
    robot_name: str,
    *,
    kinematics_yaml: str | None = None,
    kinematics_suffix: str | None = None,
    allow_sn_override: bool = False,
    kinematics_yaml_dir: str | None = None,
) -> None:
    """Print SN / calibration eligibility and warn on likely misconfiguration."""
    model_code = parse_sn_model_code(sn)
    has_calib = has_per_unit_kinematics_calibration(sn, robot_name)
    wants_calib = kinematics_yaml is not None or kinematics_suffix is not None

    print(f"robot_sn       : {sn or '(unknown)'}")
    if model_code is not None:
        print(f"sn_model_code  : {model_code} (SN positions 3-6)")

    if robot_name == "uf850":
        print("kinematics     : UF850 — all units have per-unit calibration")
    elif not has_calib:
        print(f"kinematics     : no per-unit calibration expected for this SN (model code {model_code})")
        if wants_calib:
            validate_kinematics_calibration_request(
                sn,
                robot_name,
                kinematics_yaml=kinematics_yaml,
                kinematics_suffix=kinematics_suffix,
                allow_sn_override=allow_sn_override,
                kinematics_yaml_dir=kinematics_yaml_dir,
            )
    else:
        print(f"kinematics     : per-unit calibration may be required (model code {model_code})")
        if not wants_calib:
            print(
                "[WARN] No --kinematics-suffix/--kinematics-yaml: URDF may not match "
                "firmware calibration. Run: python scripts/gen_kinematics_params.py <ip>"
            )
