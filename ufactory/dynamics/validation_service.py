"""Application service for dynamics validation; no CLI or hardware ownership."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ufactory.dynamics.analysis import validate_urdf_dynamics
from ufactory.dynamics.report import compare_report_records, read_report_records


class DynamicsValidationService:
    """Stable façade over report and static model validation operations."""

    def validate_urdf(self, urdf_path: str | Path) -> tuple[Any, ...]:
        return tuple(validate_urdf_dynamics(str(urdf_path)))

    def compare_reports(self, left: str | Path, right: str | Path) -> Sequence[dict[str, Any]]:
        return compare_report_records(read_report_records(left), read_report_records(right))
