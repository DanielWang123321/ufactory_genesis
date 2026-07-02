"""Unit tests for SN-based kinematics calibration eligibility."""

from unittest.mock import patch

import pytest

from ufactory.kinematics import (
  has_per_unit_kinematics_calibration,
  kinematics_suffix_from_sn,
  parse_sn_model_code,
  resolve_kinematics_suffix,
  validate_kinematics_calibration_request,
)


def test_parse_sn_model_code():
  assert parse_sn_model_code("XI130506D43A0A") == 1305
  assert parse_sn_model_code("XF130312C23B1E") == 1303
  assert parse_sn_model_code("ab") is None
  assert parse_sn_model_code("LI100307251B0E") == 1003


def test_xarm_calibration_threshold():
  assert not has_per_unit_kinematics_calibration("XF130312C23B1E", "xarm6")
  assert has_per_unit_kinematics_calibration("XI130412C23B1E", "xarm6")
  assert has_per_unit_kinematics_calibration("XI130506D43A0A", "xarm6")
  assert not has_per_unit_kinematics_calibration("XF130312C23B1E", "xarm5")


def test_lite6_calibration_threshold():
  assert not has_per_unit_kinematics_calibration("XX100512345678", "lite6")
  assert has_per_unit_kinematics_calibration("XX100612345678", "lite6")
  assert not has_per_unit_kinematics_calibration("LI100307251B0E", "lite6")


def test_uf850_always_has_calibration():
  assert has_per_unit_kinematics_calibration("", "uf850")
  assert has_per_unit_kinematics_calibration("XF130012345678", "uf850")


def test_validate_kinematics_rejects_old_lite6_sn_without_override():
  with pytest.raises(ValueError, match="1003 < 1006"):
    validate_kinematics_calibration_request(
      "LI100307251B0E",
      "lite6",
      kinematics_suffix="lite6_192_168_1_170",
    )


def test_validate_kinematics_allows_old_lite6_sn_with_override():
  validate_kinematics_calibration_request(
    "LI100307251B0E",
    "lite6",
    kinematics_suffix="lite6_192_168_1_170",
    allow_sn_override=True,
  )


def test_kinematics_suffix_from_sn():
  assert kinematics_suffix_from_sn("XI130506D43A0A") == "D43A0A"
  assert kinematics_suffix_from_sn("LI100307251B0E") == "251B0E"
  with pytest.raises(ValueError, match="too short"):
    kinematics_suffix_from_sn("AB12")


def test_resolve_kinematics_suffix_priority():
  sn = "XI130506D43A0A"
  assert resolve_kinematics_suffix(
    kinematics_suffix="custom",
    sn=sn,
    robot_name="xarm6",
  ) == "custom"
  assert resolve_kinematics_suffix(
    kinematics_suffix=None,
    sn=sn,
    robot_name="xarm6",
    env_suffix="from_env",
  ) == "from_env"
  assert resolve_kinematics_suffix(
    kinematics_suffix=None,
    sn=sn,
    robot_name="xarm6",
  ) == "D43A0A"
  assert resolve_kinematics_suffix(
    kinematics_yaml="/tmp/calib.yaml",
    kinematics_suffix=None,
    sn=sn,
    robot_name="xarm6",
  ) is None


def test_resolve_kinematics_suffix_skips_old_lite6_sn():
  assert resolve_kinematics_suffix(
    kinematics_suffix=None,
    sn="LI100307251B0E",
    robot_name="lite6",
  ) is None


@patch("ufactory.kinematics.fetch_robot_sn_from_ip", return_value="XI130506D43A0A")
def test_resolve_kinematics_suffix_from_ip(mock_fetch):
  from ufactory.kinematics import resolve_kinematics_suffix_from_ip

  suffix, sn = resolve_kinematics_suffix_from_ip("192.168.1.65", "xarm6")
  assert sn == "XI130506D43A0A"
  assert suffix == "D43A0A"
  mock_fetch.assert_called_once_with("192.168.1.65")
