"""Torque plotting helpers for dynamics validation reports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ufactory.dynamics.report import DynamicsSample, has_sdk_torque, joint_torque_rows


def torque_plot_layout(dof: int) -> tuple[int, int]:
    if dof == 6:
        return 2, 3
    if dof == 7:
        return 3, 3
    return max(1, int(math.ceil(dof / 3))), 3


def write_torque_plot(
    results: Sequence[DynamicsSample],
    path: str | Path,
    *,
    runtime_profile: Any | None = None,
) -> bool:
    """Write a per-joint Genesis-vs-SDK torque plot.

    Returns ``False`` when the report has no SDK torque data, which is expected
    for dry-run/sim-only validation.
    """
    samples = [r for r in results if has_sdk_torque(r)]
    if not samples:
        return False

    dof = max((len(joint_torque_rows(r, runtime_profile=runtime_profile)) for r in samples), default=0)
    if dof <= 0:
        return False

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows, cols = torque_plot_layout(dof)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.0), squeeze=False)
    pose_names = [r.pose for r in samples]
    x = np.arange(len(samples), dtype=np.float64)

    for joint_idx in range(dof):
        ax = axes[joint_idx // cols][joint_idx % cols]
        genesis_vals: list[float] = []
        sdk_vals: list[float] = []
        sdk_std_vals: list[float] = []
        abs_err_vals: list[float] = []
        for sample in samples:
            joint_rows = joint_torque_rows(sample, runtime_profile=runtime_profile)
            row = joint_rows[joint_idx] if joint_idx < len(joint_rows) else {}
            genesis_vals.append(float(row["genesis_tau_nm"]) if row.get("genesis_tau_nm") is not None else np.nan)
            sdk_vals.append(float(row["sdk_tau_mean_nm"]) if row.get("sdk_tau_mean_nm") is not None else np.nan)
            sdk_std_vals.append(float(row["sdk_tau_std_nm"]) if row.get("sdk_tau_std_nm") is not None else np.nan)
            abs_err_vals.append(float(row["abs_err_nm"]) if row.get("abs_err_nm") is not None else np.nan)

        genesis = np.asarray(genesis_vals, dtype=np.float64)
        sdk = np.asarray(sdk_vals, dtype=np.float64)
        sdk_std = np.nan_to_num(np.asarray(sdk_std_vals, dtype=np.float64), nan=0.0)
        abs_err = np.asarray(abs_err_vals, dtype=np.float64)

        ax.plot(x, genesis, marker="o", linewidth=1.4, label="Genesis theory")
        ax.errorbar(x, sdk, yerr=sdk_std, marker="s", linewidth=1.2, capsize=2.0, label="SDK mean")
        max_err = float(np.nanmax(abs_err)) if np.isfinite(abs_err).any() else float("nan")
        title = f"J{joint_idx + 1}"
        if math.isfinite(max_err):
            title += f"  max |err|={max_err:.3f} Nm"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(pose_names, rotation=80, ha="right", fontsize=7)
        if joint_idx == 0:
            ax.legend(fontsize=8)

    for unused_idx in range(dof, rows * cols):
        axes[unused_idx // cols][unused_idx % cols].axis("off")

    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True
