"""Structured diagnostics for incomplete formal ODE reductions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReductionDiagnostic:
    """One exact obstruction or retained resonance encountered during reduction."""

    stage: str
    code: str
    message: str
    power: int | None = None
    rank: int | None = None
    order: int | None = None
