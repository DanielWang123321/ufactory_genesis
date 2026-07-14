from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ufactory.kinematics import load_kinematics_calibration


def _data() -> dict:
    return {
        "schema_version": 1,
        "robot_key": "xarm6_1305",
        "serial_number": "XI130506XXXXXX",
        "units": {"position": "m", "angle": "rad"},
        "joints": {f"joint{i}": {name: 0.0 for name in ("x", "y", "z", "roll", "pitch", "yaw")} for i in range(1, 7)},
    }


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_strict_calibration_binds_robot_serial_units_and_hash(tmp_path: Path):
    calibration = load_kinematics_calibration(
        str(_write(tmp_path / "xarm6_kinematics_XXXXXX.yaml", _data())),
        robot_key="xarm6",
        serial_number="XI130506XXXXXX",
    )
    assert calibration.robot_key == "xarm6_1305"
    assert len(calibration.joints) == 6
    assert len(calibration.sha256) == 64


@pytest.mark.parametrize("mutation", ["missing", "extra", "field", "serial", "unit", "nan"])
def test_malformed_calibration_is_rejected(tmp_path: Path, mutation: str):
    data = _data()
    if mutation == "missing":
        del data["joints"]["joint6"]
    elif mutation == "extra":
        data["joints"]["joint7"] = data["joints"]["joint1"].copy()
    elif mutation == "field":
        del data["joints"]["joint1"]["yaw"]
    elif mutation == "serial":
        data["serial_number"] = "XXXXXX"
    elif mutation == "unit":
        data["units"]["position"] = "mm"
    else:
        data["joints"]["joint1"]["x"] = float("nan")
    path = _write(tmp_path / "xarm6_kinematics_XXXXXX.yaml", data)
    with pytest.raises(ValueError):
        load_kinematics_calibration(str(path), robot_key="xarm6")


def test_serial_mismatch_is_rejected(tmp_path: Path):
    path = _write(tmp_path / "xarm6_kinematics_XXXXXX.yaml", _data())
    with pytest.raises(ValueError, match="serial number"):
        load_kinematics_calibration(str(path), robot_key="xarm6", serial_number="XI130506YYYYYY")


def test_legacy_calibration_error_explains_safe_regeneration(tmp_path: Path):
    legacy = {"kinematics": _data()["joints"]}
    path = _write(tmp_path / "xarm6_kinematics_legacy.yaml", legacy)

    with pytest.raises(ValueError, match=r"legacy calibration schema.*gen_kinematics_params\.py"):
        load_kinematics_calibration(
            str(path),
            robot_key="xarm6",
            serial_number="XI130506XXXXXX",
        )
