"""Kinematic calibration helpers for xArm URDF patching."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

from ufactory.paths import PROJECT_ROOT, kinematics_user_dir, xarm6_urdf
from ufactory.robot_registry import ROBOT_PROFILES, RobotModelSpec, get_robot_profile

DEFAULT_XARM6_URDF = xarm6_urdf()

# SN positions 3-6 (1-based) = four-digit model code, e.g. XI130506... -> 1305
XARM_KINEMATICS_MIN_SN_MODEL_CODE = 1304  # xarm5/6/7: code < 1304 => no compensation
LITE6_KINEMATICS_MIN_SN_MODEL_CODE = 1006  # lite6: code < 1006 => no compensation
# UF850: all units have per-unit kinematics compensation in firmware


def load_kinematics_yaml(
  kinematics_yaml_path: str,
  joint_count: int | None = None,
) -> Dict[str, Dict[str, float]]:
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

  n = joint_count or max(
    (int(k.replace("joint", "")) for k in kinematics if str(k).startswith("joint")),
    default=6,
  )
  values = {}
  for i in range(1, n + 1):
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
  robot_name: str = "xarm6",
) -> Path:
  """Find a kinematics yaml file from a suffix (e.g., xi1305 -> xarm6_kinematics_xi1305.yaml)."""
  suffix = (kinematics_suffix or "").strip()
  if not suffix:
    raise ValueError("kinematics_suffix is empty")

  profile_key = _profile_key_for_robot_name(robot_name)
  profile = get_robot_profile(profile_key)
  prefix = profile.kinematics_prefix

  search_dirs = [Path.cwd(), kinematics_user_dir(profile.robot_name)]
  if kinematics_yaml_dir:
    search_dirs.append(Path(kinematics_yaml_dir).expanduser())
  search_dirs.append(PROJECT_ROOT)

  patterns = (
    f"{prefix}_kinematics_{suffix}.yaml",
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
    f"Cannot find kinematics YAML for {robot_name} suffix '{suffix}'. "
    f"Run: python scripts/gen_kinematics_params.py <robot_ip> {suffix}"
  )


def build_calibrated_urdf(
  base_urdf_path: str,
  kinematics: Dict[str, Dict[str, float]],
  suffix: Optional[str] = None,
  joint_count: int | None = None,
) -> str:
  """Generate a patched URDF with calibrated joint origins."""
  base = Path(base_urdf_path).expanduser().resolve()
  if not base.exists():
    raise FileNotFoundError(f"Base URDF not found: {base}")

  safe_suffix = "calib"
  if suffix:
    safe_suffix = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in suffix) or "calib"

  output_path = base.with_name(f"{base.stem}_{safe_suffix}_calib.urdf")
  tree = ET.parse(str(base))
  root = tree.getroot()

  n = joint_count or len(kinematics)
  for i in range(1, n + 1):
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
    ET.indent(tree)
  except AttributeError:
    pass
  tree.write(str(output_path), encoding="utf-8", xml_declaration=False)
  return str(output_path)


def prepare_robot_model_for_verification(
  robot_model: Optional[str],
  kinematics_yaml: Optional[str],
  kinematics_suffix: Optional[str],
  kinematics_yaml_dir: Optional[str] = None,
  default_base_urdf: Optional[str] = None,
  robot_name: str = "xarm6",
  joint_count: int | None = None,
) -> tuple[str, Optional[str]]:
  """Resolve robot model and apply kinematic calibration if requested.

  Returns (urdf_path, kinematics_yaml_path_or_none).
  """
  profile_key = _profile_key_for_robot_name(robot_name)
  profile = get_robot_profile(profile_key)
  dof = joint_count or profile.dof
  base_default = default_base_urdf or str(profile.assets_dir / profile.default_urdf)
  model_path = Path(robot_model).expanduser().resolve() if robot_model else Path(base_default)

  if kinematics_yaml is None and kinematics_suffix is None:
    return str(model_path), None

  if kinematics_yaml is not None:
    yaml_path = Path(kinematics_yaml).expanduser().resolve()
  else:
    yaml_path = find_kinematics_yaml(kinematics_suffix, kinematics_yaml_dir, robot_name=profile.robot_name)

  calibrated = build_calibrated_urdf(
    str(model_path),
    load_kinematics_yaml(str(yaml_path), joint_count=dof),
    suffix=kinematics_suffix or yaml_path.stem,
    joint_count=dof,
  )
  return calibrated, str(yaml_path)


def _profile_key_for_robot_name(robot_name: str) -> str:
  if robot_name in ROBOT_PROFILES:
    return robot_name
  for key, profile in ROBOT_PROFILES.items():
    if profile.robot_name == robot_name:
      return key
  raise KeyError(f"Unknown robot name: {robot_name}")


def parse_sn_model_code(sn: str) -> Optional[int]:
  """Parse the 4-digit model code from SN positions 3-6 (1-based).

  Example: ``XI130506D43A0A`` -> ``1305``.
  """
  cleaned = (sn or "").strip().upper()
  if len(cleaned) < 6:
    return None
  digits = cleaned[2:6]
  if not digits.isdigit():
    return None
  return int(digits)


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


def validate_kinematics_calibration_request(
  sn: str,
  robot_name: str,
  *,
  kinematics_yaml: Optional[str] = None,
  kinematics_suffix: Optional[str] = None,
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
  raise ValueError(
    f"{rule}: this unit has no per-unit kinematics compensation in firmware. "
    "Do not pass --kinematics-suffix/--kinematics-yaml; use the nominal URDF only."
  )


def log_kinematics_sn_status(
  sn: str,
  robot_name: str,
  *,
  kinematics_yaml: Optional[str] = None,
  kinematics_suffix: Optional[str] = None,
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
    print(
      f"kinematics     : no per-unit calibration expected for this SN "
      f"(model code {model_code})"
    )
    if wants_calib:
      validate_kinematics_calibration_request(
        sn, robot_name,
        kinematics_yaml=kinematics_yaml,
        kinematics_suffix=kinematics_suffix,
      )
  else:
    print(
      f"kinematics     : per-unit calibration may be required "
      f"(model code {model_code})"
    )
    if not wants_calib:
      print(
        "[WARN] No --kinematics-suffix/--kinematics-yaml: URDF may not match "
        "firmware calibration. Run: python scripts/gen_kinematics_params.py <ip> <suffix>"
      )
