"""Layered static dynamics analysis (L2/L3) + torque comparison for validation.

Owns:
* the L2a/L2b/L3a/L3b static layer machinery (``StaticLayerResult`` /
  ``StaticPoseAnalysis``) with the refactored L2b oracle
  ``pd_hold_tau vs pin_G(q_actual)``
* armature-aligned mass-matrix comparison (:func:`analyze_mass_matrices`)
* URDF static dynamics checks (:func:`validate_urdf_dynamics`)
* torque compare / classify / ``build_dynamics_sample`` glue that wires a
  Genesis sample to the layered analysis

Depends on ``report`` (dataclasses + runtime/limits helpers) and ``poses``
(runtime singletons / SafePose); it has no runtime dependency back into probe/cli.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.dynamics.report import (
    ABS_ERR_LIMITS,
    L2_ERR_LIMIT,
    REL_ERR_LIMIT,
    GenesisDynamicsSample,
    DynamicsSample,
    TorqueCompareResult,
    UrdfDynamicsIssue,
    ValidationStatus,
    _abs_err_limits,
    _effort_limits,
    _runtime_dof,
    array_or_nan,
)
from ufactory.dynamics.poses import (
    JOINT_NAMES,
    SafePose,
    _XARM6_RUNTIME,
)

if TYPE_CHECKING:
    from ufactory.robot_params import RobotRuntimeProfile

LAYER_L2A = "L2a"
LAYER_L2B = "L2b"
LAYER_L3A = "L3a"
LAYER_L3B = "L3b"
STATIC_LAYERS = (LAYER_L2A, LAYER_L2B, LAYER_L3A, LAYER_L3B)


@dataclass(frozen=True)
class StaticLayerResult:
    layer: str
    passed: bool
    severity: str  # pass | warn | fail
    l2_err: float
    abs_err: np.ndarray
    signed_err: np.ndarray
    notes: str = ""


@dataclass
class StaticPoseAnalysis:
    pose: str
    layers: list[StaticLayerResult] = field(default_factory=list)
    clamp_slack_est: np.ndarray | None = None
    armature: np.ndarray | None = None
    reference_mass_matrix: np.ndarray | None = None
    mass_rel_fro: float | None = None
    warnings: list[str] = field(default_factory=list)


def format_torque_row(values) -> str:
    return "[" + ", ".join(f"{float(v):8.3f}" for v in values) + "]"


def parse_strict_static_layers(value: str | None) -> frozenset[str]:
    if not value or value.strip().lower() in {"all", "*"}:
        return frozenset(STATIC_LAYERS)
    layers: set[str] = set()
    alias = {
        "l2a": LAYER_L2A,
        "l2b": LAYER_L2B,
        "l3a": LAYER_L3A,
        "l3b": LAYER_L3B,
    }
    for token in value.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t == "l2":
            layers.update({LAYER_L2A, LAYER_L2B})
        elif t == "l3":
            layers.update({LAYER_L3A, LAYER_L3B})
        elif t in alias:
            layers.add(alias[t])
    return frozenset(layers or STATIC_LAYERS)


def estimate_pd_residual(
    q_target: Sequence[float],
    q_actual: Sequence[float],
    qvel: Sequence[float],
    *,
    kp: Sequence[float],
    kv: Sequence[float],
) -> np.ndarray:
    """Estimate PD feedback torque: Kp*(q* - q) + Kv*(0 - qdot)."""
    target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    actual = np.asarray(q_actual, dtype=np.float64).reshape(-1)
    vel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    kp_arr = np.asarray(kp, dtype=np.float64).reshape(-1)
    kv_arr = np.asarray(kv, dtype=np.float64).reshape(-1)
    n = min(len(target), len(actual), len(vel), len(kp_arr), len(kv_arr))
    return kp_arr[:n] * (target[:n] - actual[:n]) - kv_arr[:n] * vel[:n]


def _layer_result(
    layer: str,
    signed_err: np.ndarray,
    *,
    abs_limits: np.ndarray,
    l2_limit: float,
    rel_limit: float,
    effort_limits: np.ndarray,
    warn_scale: float = 1.0,
    fail_scale: float = 1.0,
    notes: str = "",
) -> StaticLayerResult:
    abs_err = np.abs(signed_err)
    rel_err = abs_err / np.maximum(effort_limits[: len(abs_err)], 1e-9)
    l2_err = float(np.linalg.norm(abs_err))
    warn_abs = abs_limits * warn_scale
    fail_abs = abs_limits * fail_scale
    passed = bool((abs_err <= warn_abs).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit * warn_scale)
    if passed:
        severity = "pass"
    elif bool((abs_err <= fail_abs).all() and l2_err <= l2_limit * fail_scale):
        severity = "warn"
    else:
        severity = "fail"
    return StaticLayerResult(
        layer=layer,
        passed=passed,
        severity=severity,
        l2_err=l2_err,
        abs_err=abs_err,
        signed_err=np.asarray(signed_err, dtype=np.float64),
        notes=notes,
    )


def analyze_genesis_internal(
    sample: Any,
    *,
    runtime_profile: Any,
) -> StaticLayerResult:
    dyn = runtime_profile.dynamics
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    signed = sample.pd_hold_tau - sample.actual_dof_force
    return _layer_result(
        LAYER_L2A,
        signed,
        abs_limits=np.asarray(dyn.genesis_internal_abs_limits, dtype=np.float64),
        l2_limit=dyn.genesis_internal_l2_limit,
        rel_limit=dyn.rel_err_limit,
        effort_limits=limits,
        warn_scale=1.0,
        fail_scale=1.5,
        notes="pd_hold_tau vs actual_dof_force",
    )


def analyze_pd_vs_pinocchio(
    pd_hold_tau: np.ndarray,
    q_actual: Sequence[float],
    *,
    reference: Any,
    runtime_profile: Any,
) -> StaticLayerResult:
    """L2b: Genesis PD hold torque vs Pinocchio gravity oracle G(q_actual).

    The gravity oracle is ``pin_G(q_actual)`` evaluated on the independent
    reference backend. ``pd_hold_tau`` is the Genesis controller output at the
    settled configuration, which at steady state carries the gravity-compensating
    torque plus joint friction/damping. This compares those two quantities instead
    of the previous ``gravity_est = pd_hold_tau - pd_residual`` term, which is
    identically the force-range clamp slack (~0 when unsaturated) and is not a
    gravity estimate.
    """
    dyn = runtime_profile.dynamics
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    pin_gravity = np.asarray(reference.gravity(np.asarray(q_actual, dtype=np.float64)), dtype=np.float64)
    signed = np.asarray(pd_hold_tau, dtype=np.float64) - pin_gravity
    return _layer_result(
        LAYER_L2B,
        signed,
        abs_limits=np.asarray(dyn.pd_vs_pin_abs_limits, dtype=np.float64),
        l2_limit=dyn.pd_vs_pin_l2_limit,
        rel_limit=dyn.rel_err_limit,
        effort_limits=limits,
        warn_scale=1.0,
        fail_scale=1.5,
        notes="pd_hold_tau vs pin_G",
    )


def analyze_mass_matrices(
    mass_genesis: np.ndarray,
    mass_pin: np.ndarray,
    *,
    runtime_profile: Any,
    armature: Sequence[float] | np.ndarray | None = None,
) -> tuple[StaticLayerResult, float]:
    m_gen = np.asarray(mass_genesis, dtype=np.float64)
    m_pin = np.asarray(mass_pin, dtype=np.float64)
    if m_gen.shape != m_pin.shape:
        n = min(m_gen.shape[0], m_pin.shape[0])
        m_gen = m_gen[:n, :n]
        m_pin = m_pin[:n, :n]
    arm_note = ""
    if armature is not None:
        arm_arr = np.asarray(armature, dtype=np.float64).reshape(-1)
        k = min(len(arm_arr), m_gen.shape[0])
        if k > 0:
            m_gen = m_gen.copy()
            m_gen[:k, :k] -= np.diag(arm_arr[:k])
            arm_note = f" armature={arm_arr[:k].tolist()}"
    diff = m_gen - m_pin
    pin_norm = float(np.linalg.norm(m_pin))
    rel_fro = float(np.linalg.norm(diff) / max(pin_norm, 1e-9))
    dyn = runtime_profile.dynamics
    if rel_fro <= dyn.mass_rel_fro_limit_warn:
        severity = "pass"
        passed = True
    elif rel_fro <= dyn.mass_rel_fro_limit_fail:
        severity = "warn"
        passed = False
    else:
        severity = "fail"
        passed = False
    layer = StaticLayerResult(
        layer=LAYER_L3A,
        passed=passed,
        severity=severity,
        l2_err=rel_fro,
        abs_err=np.abs(diff).reshape(-1)[:6] if diff.size else np.zeros(1),
        signed_err=diff.reshape(-1)[:6] if diff.size else np.zeros(1),
        notes=f"mass Frobenius rel err={rel_fro:.4f}{arm_note}",
    )
    return layer, rel_fro


def analyze_pin_vs_real(
    pin_gravity: np.ndarray,
    tau_real: np.ndarray,
    *,
    runtime_profile: Any,
) -> StaticLayerResult:
    dyn = runtime_profile.dynamics
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    base_abs = np.asarray(dyn.abs_err_limits, dtype=np.float64)
    pin_abs = base_abs * dyn.pin_vs_real_abs_scale
    signed = pin_gravity - tau_real
    return _layer_result(
        LAYER_L3B,
        signed,
        abs_limits=pin_abs,
        l2_limit=dyn.l2_err_limit * dyn.pin_vs_real_abs_scale,
        rel_limit=dyn.rel_err_limit,
        effort_limits=limits,
        warn_scale=1.0,
        fail_scale=1.0,
        notes="pin_G vs tau_real",
    )


def pinocchio_static_at_q(reference: Any, q: Sequence[float]) -> dict[str, Any]:
    q_arr = np.asarray(q, dtype=np.float64)
    pin_g = reference.gravity(q_arr)
    rnea_g = reference.rnea(q_arr, np.zeros_like(q_arr), np.zeros_like(q_arr))
    mass_pin = reference.mass_matrix(q_arr)
    rnea_delta = float(np.linalg.norm(pin_g - rnea_g))
    return {
        "pin_gravity": pin_g,
        "rnea_static": rnea_g,
        "rnea_delta": rnea_delta,
        "mass_pin": mass_pin,
    }


def build_static_pose_analysis(
    pose_name: str,
    q_target: Sequence[float],
    genesis_sample: Any,
    *,
    runtime_profile: Any,
    reference: Any | None = None,
    tau_real: np.ndarray | None = None,
) -> StaticPoseAnalysis:
    analysis = StaticPoseAnalysis(pose=pose_name)
    pd_residual = estimate_pd_residual(
        q_target,
        genesis_sample.q_actual,
        genesis_sample.qvel,
        kp=runtime_profile.arm.kp,
        kv=runtime_profile.arm.kv,
    )
    analysis.clamp_slack_est = np.asarray(genesis_sample.pd_hold_tau, dtype=np.float64) - pd_residual
    armature = getattr(genesis_sample, "armature", None)
    if armature is not None:
        analysis.armature = np.asarray(armature, dtype=np.float64)

    if not genesis_sample.settled:
        analysis.warnings.append("not settled: static layers may be unreliable")
        return analysis

    analysis.layers.append(analyze_genesis_internal(genesis_sample, runtime_profile=runtime_profile))

    if reference is None:
        analysis.warnings.append("pinocchio reference unavailable")
        return analysis

    static_ref = pinocchio_static_at_q(reference, genesis_sample.q_actual)
    pin_g = static_ref["pin_gravity"]
    if static_ref["rnea_delta"] > 1e-4:
        analysis.warnings.append(f"pin rnea(q,0,0) delta={static_ref['rnea_delta']:.2e}")

    analysis.layers.append(
        analyze_pd_vs_pinocchio(
            genesis_sample.pd_hold_tau,
            genesis_sample.q_actual,
            reference=reference,
            runtime_profile=runtime_profile,
        )
    )
    mass_layer, rel_fro = analyze_mass_matrices(
        genesis_sample.mass_matrix,
        static_ref["mass_pin"],
        runtime_profile=runtime_profile,
        armature=analysis.armature,
    )
    analysis.layers.append(mass_layer)
    analysis.mass_rel_fro = rel_fro
    analysis.reference_mass_matrix = static_ref["mass_pin"]

    if tau_real is not None and np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        analysis.layers.append(
            analyze_pin_vs_real(pin_g, np.asarray(tau_real, dtype=np.float64), runtime_profile=runtime_profile)
        )

    for layer in analysis.layers:
        if layer.severity == "warn":
            analysis.warnings.append(f"{layer.layer}: {layer.notes} l2={layer.l2_err:.3f}")
        elif layer.severity == "fail":
            analysis.warnings.append(f"{layer.layer}: FAIL {layer.notes} l2={layer.l2_err:.3f}")

    return analysis


def summarize_static_layers(
    analyses: Sequence[StaticPoseAnalysis],
    *,
    strict_layers: frozenset[str] | None = None,
) -> tuple[dict[str, int], bool]:
    counts = {layer: {"pass": 0, "warn": 0, "fail": 0} for layer in STATIC_LAYERS}
    strict = strict_layers or frozenset()
    strict_fail = False
    for analysis in analyses:
        for layer in analysis.layers:
            bucket = counts.get(layer.layer)
            if bucket is None:
                continue
            bucket[layer.severity] = bucket.get(layer.severity, 0) + 1
            if layer.layer in strict and layer.severity in {"warn", "fail"}:
                strict_fail = True
    return counts, strict_fail


def static_layer_l2(analysis: StaticPoseAnalysis, layer_id: str) -> float | None:
    for layer in analysis.layers:
        if layer.layer == layer_id:
            return layer.l2_err
    return None


def compare_torques(
    tau_genesis: np.ndarray,
    tau_real: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    effort_limits: np.ndarray | None = None,
    abs_limits: np.ndarray | None = None,
    l2_limit: float = L2_ERR_LIMIT,
    rel_limit: float = REL_ERR_LIMIT,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    runtime = runtime_profile or _XARM6_RUNTIME
    limits = effort_limits if effort_limits is not None else _effort_limits(runtime)
    abs_lims = abs_limits if abs_limits is not None else _abs_err_limits(runtime)

    tau_g = np.asarray(tau_genesis, dtype=np.float64).reshape(-1)
    tau_r = np.asarray(tau_real, dtype=np.float64).reshape(-1)
    abs_err = np.abs(tau_g - tau_r)
    rel_err = abs_err / limits
    l2_err = float(np.linalg.norm(abs_err))
    passed = bool((abs_err <= abs_lims).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit)
    return abs_err, rel_err, l2_err, passed


def classify_torque_result(
    *,
    settled: bool,
    saturated: bool,
    tau_real: np.ndarray | None,
    abs_err: np.ndarray | None = None,
    rel_err: np.ndarray | None = None,
    l2_err: float | None = None,
    runtime_profile: RobotRuntimeProfile | None = None,
    abs_limits: np.ndarray | None = None,
    l2_limit: float = L2_ERR_LIMIT,
    rel_limit: float = REL_ERR_LIMIT,
) -> ValidationStatus:
    if not settled:
        return ValidationStatus.NOT_SETTLED
    if saturated:
        return ValidationStatus.SATURATED
    if tau_real is None or not np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        return ValidationStatus.INSUFFICIENT_DATA
    if abs_err is None or rel_err is None or l2_err is None:
        return ValidationStatus.INSUFFICIENT_DATA

    runtime = runtime_profile or _XARM6_RUNTIME
    limits = abs_limits if abs_limits is not None else _abs_err_limits(runtime)
    if bool((abs_err <= limits).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit):
        return ValidationStatus.PASS
    failed = np.where(abs_err > limits)[0]
    if len(failed) == 1 and l2_err <= l2_limit * 1.25:
        return ValidationStatus.FAIL_BIAS
    return ValidationStatus.FAIL_MODEL


def _apply_static_analysis_fields(
    sample: DynamicsSample,
    analysis: StaticPoseAnalysis,
    *,
    reference_gravity_tau: np.ndarray | None = None,
) -> None:
    sample.clamp_slack_est = analysis.clamp_slack_est
    sample.armature = analysis.armature
    sample.reference_mass_matrix = analysis.reference_mass_matrix
    sample.static_layers = list(analysis.layers)
    sample.static_warnings = list(analysis.warnings)
    sample.l2a_l2_err = static_layer_l2(analysis, "L2a")
    sample.l2b_l2_err = static_layer_l2(analysis, "L2b")
    sample.l3a_mass_rel_fro = analysis.mass_rel_fro
    sample.l3b_l2_err = static_layer_l2(analysis, "L3b")
    if analysis.clamp_slack_est is not None:
        sample.clamp_slack_l2 = float(np.linalg.norm(analysis.clamp_slack_est))
    ref_g = reference_gravity_tau
    if ref_g is not None:
        sample.pin_G_l2 = float(np.linalg.norm(np.asarray(ref_g, dtype=np.float64)))
    elif sample.reference_gravity_tau is not None:
        sample.pin_G_l2 = float(np.linalg.norm(np.asarray(sample.reference_gravity_tau, dtype=np.float64)))


def build_dynamics_sample(
    pose: SafePose,
    genesis_sample: GenesisDynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    tau_real: np.ndarray | None = None,
    tau_real_median: np.ndarray | None = None,
    tau_real_std: np.ndarray | None = None,
    tau_real_min: np.ndarray | None = None,
    tau_real_max: np.ndarray | None = None,
    tau_direct: np.ndarray | None = None,
    n_real_samples: int = 0,
    reference_gravity_tau: np.ndarray | None = None,
    reference: Any | None = None,
    skip_reason: str = "",
    notes: list[str] | None = None,
) -> DynamicsSample:
    runtime = runtime_profile or _XARM6_RUNTIME
    abs_err = rel_err = signed_err = None
    l2_err: float | None = None
    if tau_real is not None and np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        signed_err = genesis_sample.pd_hold_tau - np.asarray(tau_real, dtype=np.float64)
        abs_err, rel_err, l2_err, _ = compare_torques(
            genesis_sample.pd_hold_tau,
            tau_real,
            runtime_profile=runtime,
            l2_limit=runtime.dynamics.l2_err_limit,
            rel_limit=runtime.dynamics.rel_err_limit,
        )

    status = classify_torque_result(
        settled=genesis_sample.settled,
        saturated=genesis_sample.saturated,
        tau_real=tau_real,
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=l2_err,
        runtime_profile=runtime,
        l2_limit=runtime.dynamics.l2_err_limit,
        rel_limit=runtime.dynamics.rel_err_limit,
    )
    if skip_reason == "not settled":
        status = ValidationStatus.NOT_SETTLED
    if skip_reason == "dry-run" and status == ValidationStatus.INSUFFICIENT_DATA:
        notes = [*(notes or []), "dry-run: no real robot torque data"]

    sample = DynamicsSample(
        pose=pose.name,
        q=pose.q,
        ee_z_mm=pose.ee_z_mm,
        status=status,
        settled=genesis_sample.settled,
        saturated=genesis_sample.saturated,
        q_actual=genesis_sample.q_actual,
        qvel=genesis_sample.qvel,
        pd_hold_tau=genesis_sample.pd_hold_tau,
        actual_dof_force=genesis_sample.actual_dof_force,
        mass_matrix=genesis_sample.mass_matrix,
        reference_gravity_tau=reference_gravity_tau,
        tau_real=tau_real,
        tau_real_median=tau_real_median,
        tau_real_std=tau_real_std,
        tau_real_min=tau_real_min,
        tau_real_max=tau_real_max,
        tau_direct=tau_direct,
        abs_err=abs_err,
        rel_err=rel_err,
        signed_err=signed_err,
        l2_err=l2_err,
        n_real_samples=n_real_samples,
        skip_reason=skip_reason,
        notes=notes or [],
    )
    static_analysis = build_static_pose_analysis(
        pose.name,
        pose.q,
        genesis_sample,
        runtime_profile=runtime,
        reference=reference,
        tau_real=tau_real,
    )
    _apply_static_analysis_fields(sample, static_analysis, reference_gravity_tau=reference_gravity_tau)
    if reference_gravity_tau is None and reference is not None:
        try:
            sample.reference_gravity_tau = reference.gravity(genesis_sample.q_actual)
            sample.pin_G_l2 = float(np.linalg.norm(sample.reference_gravity_tau))
        except Exception:
            pass
    return sample


def torque_result_from_sample(sample: DynamicsSample) -> TorqueCompareResult:
    passed = sample.status == ValidationStatus.PASS
    n = int(np.asarray(sample.q).reshape(-1).size)
    tau_g = sample.pd_hold_tau if sample.pd_hold_tau is not None else array_or_nan(n)
    tau_r = sample.tau_real if sample.tau_real is not None else array_or_nan(n)
    abs_err = sample.abs_err if sample.abs_err is not None else array_or_nan(n)
    rel_err = sample.rel_err if sample.rel_err is not None else array_or_nan(n)
    return TorqueCompareResult(
        name=sample.pose,
        q=sample.q,
        ee_z_mm=sample.ee_z_mm,
        tau_genesis=tau_g,
        tau_real=tau_r,
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=float(sample.l2_err) if sample.l2_err is not None else float("nan"),
        settled=sample.settled,
        passed=passed,
        skip_reason=sample.skip_reason,
        status=sample.status,
    )


def validate_urdf_dynamics(urdf_path: str | Path, *, com_abs_limit_m: float = 2.0) -> list[UrdfDynamicsIssue]:
    """Static URDF dynamics checks for inertial and joint dynamics blocks."""
    root = ET.parse(str(urdf_path)).getroot()
    issues: list[UrdfDynamicsIssue] = []

    for link in root.findall("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        if mass_el is None or "value" not in mass_el.attrib:
            issues.append(UrdfDynamicsIssue("ERROR", name, "missing inertial mass"))
            continue
        mass = float(mass_el.attrib["value"])
        if mass <= 0:
            issues.append(UrdfDynamicsIssue("ERROR", name, "mass must be positive", mass))

        origin = inertial.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            if xyz.size == 3 and float(np.max(np.abs(xyz))) > com_abs_limit_m:
                issues.append(UrdfDynamicsIssue("WARN", name, "COM magnitude looks too large", xyz.tolist()))

        inertia = inertial.find("inertia")
        if inertia is None:
            issues.append(UrdfDynamicsIssue("ERROR", name, "missing inertia matrix"))
            continue
        vals = {k: float(inertia.get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        mat = np.array(
            [
                [vals["ixx"], vals["ixy"], vals["ixz"]],
                [vals["ixy"], vals["iyy"], vals["iyz"]],
                [vals["ixz"], vals["iyz"], vals["izz"]],
            ],
            dtype=np.float64,
        )
        eig = np.linalg.eigvalsh(mat)
        if float(eig.min()) <= 0.0:
            issues.append(UrdfDynamicsIssue("ERROR", name, "inertia matrix is not positive definite", eig.tolist()))
        if not (
            vals["ixx"] + vals["iyy"] >= vals["izz"]
            and vals["ixx"] + vals["izz"] >= vals["iyy"]
            and vals["iyy"] + vals["izz"] >= vals["ixx"]
        ):
            issues.append(UrdfDynamicsIssue("ERROR", name, "principal inertia triangle inequality failed", vals))

    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        name = joint.get("name", "")
        limit = joint.find("limit")
        if joint_type != "continuous" and limit is None:
            issues.append(UrdfDynamicsIssue("ERROR", name, "missing joint limit"))
        if limit is not None:
            for attr in ("effort", "velocity"):
                if attr not in limit.attrib:
                    issues.append(UrdfDynamicsIssue("ERROR", name, f"missing limit {attr}"))
                elif float(limit.attrib[attr]) <= 0:
                    issues.append(UrdfDynamicsIssue("ERROR", name, f"limit {attr} must be positive", limit.attrib[attr]))
        dynamics = joint.find("dynamics")
        if dynamics is None:
            issues.append(UrdfDynamicsIssue("WARN", name, "missing joint dynamics block"))
            continue
        for attr in ("damping", "friction"):
            if attr not in dynamics.attrib:
                issues.append(UrdfDynamicsIssue("WARN", name, f"missing dynamics {attr}"))
            elif float(dynamics.attrib[attr]) < 0:
                issues.append(UrdfDynamicsIssue("ERROR", name, f"dynamics {attr} must be non-negative", dynamics.attrib[attr]))

    return issues