"""Certified parameter strata for turning-point and uniform-WKB structure."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from semialg import parametric_cad

from .operator import LinearDifferentialOperator, _coerce_linear_operator
from .transition_loci import turning_loci
from .turning import analyze_turning_points


@dataclass(frozen=True)
class ParameterizedTurningStratum:
    condition: sp.Expr
    sample: tuple[tuple[sp.Symbol, sp.Expr], ...]
    finite_multiplicities: tuple[int, ...]
    turning_kinds: tuple[str, ...]
    uniform_families: tuple[str | None, ...]
    certified: bool


@dataclass(frozen=True)
class ParameterizedTurningAnalysis:
    parameters: tuple[sp.Symbol, ...]
    transition_polynomials: tuple[sp.Expr, ...]
    strata: tuple[ParameterizedTurningStratum, ...]
    exhaustive: bool


def parameterized_turning_analysis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    parameters: tuple[sp.Symbol, ...],
    assumptions: sp.Expr | bool = True,
) -> ParameterizedTurningAnalysis:
    """Stratify parameter space where finite turning multiplicities are constant.

    The discriminant/degree-loss loci are computed symbolically; each CAD cell
    is checked by exact specialization and turning-point analysis.  This is the
    parameter geometry needed to choose Airy versus degenerate Weber local
    uniformization without crossing a confluence locus.
    """
    op = _coerce_linear_operator(ode, function, variable)
    loci = tuple(dict.fromkeys(turning_loci(op)))
    marker = sp.Ne(sp.prod(loci), 0, evaluate=False) if loci else sp.S.true
    geometry = parametric_cad(
        marker,
        (),
        parameters=parameters,
        assumptions=sp.sympify(assumptions),
        output="result",
    )
    strata: list[ParameterizedTurningStratum] = []
    for case in geometry.cases:
        sample = dict(case.sample)
        specialized = LinearDifferentialOperator(
            op.variable,
            op.function,
            tuple(sp.cancel(sp.sympify(c).subs(sample)) for c in op.coefficients),
            sp.cancel(sp.sympify(op.inhomogeneous).subs(sample)),
        )
        analysis = analyze_turning_points(specialized)
        finite = tuple(tp for tp in analysis.points if tp.point != sp.oo)
        strata.append(
            ParameterizedTurningStratum(
                case.condition,
                tuple(sorted(sample.items(), key=lambda item: sp.default_sort_key(item[0]))),
                tuple(tp.multiplicity for tp in finite),
                tuple(tp.kind.value for tp in finite),
                tuple(
                    "airy" if tp.multiplicity == 1 else "weber" if tp.multiplicity == 2 else None
                    for tp in finite
                ),
                geometry.status == "complete",
            )
        )
    return ParameterizedTurningAnalysis(
        parameters, loci, tuple(strata), geometry.status == "complete"
    )
