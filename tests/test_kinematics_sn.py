"""Unit tests for SN-based kinematics calibration eligibility."""

from ufactory.kinematics import (
  has_per_unit_kinematics_calibration,
  parse_sn_model_code,
)


def test_parse_sn_model_code():
  assert parse_sn_model_code("XI130506D43A0A") == 1305
  assert parse_sn_model_code("XF130312C23B1E") == 1303
  assert parse_sn_model_code("ab") is None


def test_xarm_calibration_threshold():
  assert not has_per_unit_kinematics_calibration("XF130312C23B1E", "xarm6")
  assert has_per_unit_kinematics_calibration("XI130412C23B1E", "xarm6")
  assert has_per_unit_kinematics_calibration("XI130506D43A0A", "xarm6")
  assert not has_per_unit_kinematics_calibration("XF130312C23B1E", "xarm5")


def test_lite6_calibration_threshold():
  assert not has_per_unit_kinematics_calibration("XX100512345678", "lite6")
  assert has_per_unit_kinematics_calibration("XX100612345678", "lite6")


def test_uf850_always_has_calibration():
  assert has_per_unit_kinematics_calibration("", "uf850")
  assert has_per_unit_kinematics_calibration("XF130012345678", "uf850")
