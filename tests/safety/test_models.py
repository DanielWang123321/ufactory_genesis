from __future__ import annotations

import math

import pytest

import ufactory.safety as safety
import ufactory.safety.clock as clock_module
from ufactory.safety import (
    CollisionExemption,
    FaultClass,
    PreflightCheck,
    PreflightReport,
    SafetyPolicy,
    SafetyViolation,
    ViolationType,
)


def test_domain_models_validate_and_serialize():
    violation = SafetyViolation(ViolationType.TIMING, "run", "late", fault_class=FaultClass.PAUSE, actual=1, limit=2)
    check = PreflightCheck("timing", False, 1, 0.1, {"late": True})
    report = PreflightReport(1, False, "p", "robot", "servo_j", "c", "u", "k", "s", (check,), (violation,))
    assert report.to_dict()["violations"][0]["fault_class"] == "pause"
    with pytest.raises(ValueError, match="finite"):
        SafetyViolation(ViolationType.TIMING, "run", "bad", actual=math.inf)
    with pytest.raises(ValueError, match="metrics"):
        PreflightCheck("bad", True, -1)
    with pytest.raises(ValueError, match="does not match"):
        PreflightReport(1, True, "p", "r", "e", "c", "u", "k", "s", (check,), ())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": 2},
        {"workspace_lower_m": (0.0, 0.0)},
        {"workspace_lower_m": (0.0, 0.0, 0.0), "workspace_upper_m": (0.0, 1.0, 1.0)},
        {"z_min_m": 2.0},
        {"joint_limit_margin_rad": -1.0},
        {"minor_lateness_ratio": 2.0},
        {"minor_lateness_limit_per_s": 0},
        {"max_ik_jump_rad": math.nan},
    ],
)
def test_safety_policy_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SafetyPolicy(**kwargs)


def test_collision_exemption_is_pair_order_independent_and_exact():
    exemption = CollisionExemption("a", "b", "grasp", "known", "u", "0.2.6")
    assert exemption.matches("b", "a", "grasp", "u")
    assert not exemption.matches("a", "b", "place", "u")


def test_system_clock_uses_monotonic_ns_and_never_negative_sleep(monkeypatch):
    sleeps: list[float] = []
    values = iter((100, 100, 200, 200))
    monkeypatch.setattr(clock_module.time, "monotonic_ns", lambda: next(values))
    monkeypatch.setattr(clock_module.time, "sleep", sleeps.append)
    clock = clock_module.SystemClock()
    assert clock.monotonic_ns() == 100
    clock.wait_until_ns(200)
    clock.wait_until_ns(50)
    assert sleeps == [pytest.approx(1e-7)]


def test_safety_package_unknown_attribute_is_rejected():
    with pytest.raises(AttributeError):
        safety.__getattr__("does_not_exist")
