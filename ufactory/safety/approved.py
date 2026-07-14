"""Opaque approval token created only after a successful preflight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ufactory.safety.models import PreflightReport

if TYPE_CHECKING:
    from ufactory.trajectory.segments import Program


_ISSUER = object()


@dataclass(frozen=True, init=False)
class ApprovedProgram:
    """A program bound to the exact identity and assets checked by preflight.

    The constructor is intentionally private.  Callers receive instances from
    :func:`ufactory.trajectory.preflight_program` only.
    """

    program: Program
    report: PreflightReport
    expected_serial_number: str

    def __init__(
        self,
        program: Program,
        report: PreflightReport,
        expected_serial_number: str,
        *,
        _issuer: object,
    ) -> None:
        if _issuer is not _ISSUER:
            raise TypeError("ApprovedProgram can only be created by SafetyGate")
        if not report.passed:
            raise ValueError("cannot approve a failed preflight report")
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "report", report)
        object.__setattr__(self, "expected_serial_number", expected_serial_number)


def issue_approved_program(
    program: Program,
    report: PreflightReport,
    expected_serial_number: str,
) -> ApprovedProgram:
    """Internal issuance hook used by :class:`SafetyGate`."""

    return ApprovedProgram(program, report, expected_serial_number, _issuer=_ISSUER)
