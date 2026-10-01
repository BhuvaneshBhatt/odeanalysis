"""Sectorial dominant/recessive classification of formal ODE branches."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .formal import CompleteFormalExponentialPart, complete_formal_exponential_parts
from .operator import LinearDifferentialOperator
from .stokes import stokes_geometry_from_exponential_parts


@dataclass(frozen=True)
class DominanceSector:
    """Dominance ordering on one open sector of the common ramified cover."""

    index: int
    start_angle: sp.Expr
    end_angle: sp.Expr
    representative_angle: sp.Expr
    dominance_levels: tuple[tuple[int, ...], ...]

    @property
    def dominant_branches(self) -> tuple[int, ...]:
        """Return branches with largest exponential magnitude in the sector."""

        return self.dominance_levels[0] if self.dominance_levels else ()

    @property
    def recessive_branches(self) -> tuple[int, ...]:
        """Return branches with smallest exponential magnitude in the sector."""

        return self.dominance_levels[-1] if self.dominance_levels else ()


@dataclass(frozen=True)
class SolutionDominanceAnalysis:
    """Sectorial exponential dominance of completed formal solution branches.

    Branch indices refer to ``exponential_parts``.  Classification is at the
    exponential scale: branches with identical completed exponential polynomial
    remain tied even if algebraic or logarithmic prefactors later distinguish
    their magnitudes.
    """

    point: sp.Expr
    exponential_parts: tuple[CompleteFormalExponentialPart, ...]
    common_ramification: int
    sectors: tuple[DominanceSector, ...]
    complete: bool
    limitation: str | None = None

    @property
    def branch_count(self) -> int:
        """Return the number of completed exponential branches."""

        return len(self.exponential_parts)


def classify_solution_dominance(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    max_branches: int = 64,
) -> SolutionDominanceAnalysis:
    """Classify dominant and recessive formal branches sector by sector.

    The comparison uses completed exponential polynomials on their common
    uniformizing cover.  Equal-magnitude rays form sector boundaries.  If
    symbolic coefficients prevent exact angular ordering or sign comparison,
    ``complete`` is false and no sector ordering is guessed.
    """

    parts = complete_formal_exponential_parts(
        ode,
        function,
        variable,
        point=point,
        max_branches=max_branches,
    )
    if not parts:
        return SolutionDominanceAnalysis(
            point=sp.sympify(point),
            exponential_parts=(),
            common_ramification=1,
            sectors=(),
            complete=False,
            limitation="no irregular exponential branches were found",
        )
    if len(parts) == 1:
        ramification = int(parts[0].ramification_index)
        return SolutionDominanceAnalysis(
            point=sp.sympify(point),
            exponential_parts=parts,
            common_ramification=ramification,
            sectors=(
                DominanceSector(
                    index=0,
                    start_angle=sp.S.Zero,
                    end_angle=2 * sp.pi,
                    representative_angle=sp.pi,
                    dominance_levels=((0,),),
                ),
            ),
            complete=True,
        )

    geometry = stokes_geometry_from_exponential_parts(parts, point=point)
    sectors = tuple(
        DominanceSector(
            index=sector.index,
            start_angle=sector.start_angle,
            end_angle=sector.end_angle,
            representative_angle=sector.representative_angle,
            dominance_levels=sector.dominance_levels,
        )
        for sector in geometry.sectors
    )
    limitation = None
    if not geometry.sector_geometry_complete:
        limitation = (
            "symbolic Stokes boundaries or dominance signs could not be ordered exactly"
        )
    return SolutionDominanceAnalysis(
        point=sp.sympify(point),
        exponential_parts=parts,
        common_ramification=geometry.common_ramification,
        sectors=sectors,
        complete=geometry.sector_geometry_complete,
        limitation=limitation,
    )
