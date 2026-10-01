"""Unified local singularity structure for scalar linear ODEs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import sympy as sp
from funcprops import normalize_assumptions

from ._assumptions import zero_status
from ._local import LocalCoordinate, local_coordinate
from .formal_basis import (
    FormalBasisError,
    formal_monodromy,
    logarithmic_frobenius_basis,
)
from .frobenius import FrobeniusAnalysis, frobenius_analysis
from .fuchsian import ApparentSingularityAnalysis, apparent_singularity_analysis
from .operator import LinearDifferentialOperator, _coerce_linear_operator
from .singularities import ODESingularity, ODESingularityKind, analyze_ode_singularities


@dataclass(frozen=True)
class FrobeniusConvergence:
    """Convergence geometry certified from singularities of the ODE coefficients.

    ``radius`` is the coefficient-analyticity radius.  It is not
    presented as a maximal continuation radius for every solution: apparent
    singularities or special solutions may continue farther.
    """

    point: sp.Expr
    radius: sp.Expr | None
    nearest_singularities: tuple[sp.Expr, ...]
    domain: sp.Expr | None
    certified: bool
    coordinate: sp.Symbol
    solution_continuation_radius: sp.Expr | None = None
    solution_continuation_certified: bool = False

    @property
    def coefficient_radius(self) -> sp.Expr | None:
        return self.radius

    @property
    def coefficient_domain(self) -> sp.Expr | None:
        return self.domain


@dataclass(frozen=True)
class FrobeniusLocalMonodromy:
    """Local monodromy derived from completed Frobenius data."""

    point: sp.Expr
    matrix: sp.ImmutableMatrix | None
    certified: bool
    logarithmic: bool | None
    reason: str


@dataclass(frozen=True)
class SingularityPointStructure:
    """Unified evidence attached to one singular point."""

    singularity: ODESingularity
    frobenius: FrobeniusAnalysis | None
    convergence: FrobeniusConvergence | None
    apparent: ApparentSingularityAnalysis | None
    monodromy: FrobeniusLocalMonodromy | None
    local_coordinate: LocalCoordinate | None = None


@dataclass(frozen=True)
class SingularityStructure:
    """Unified finite/infinite singularity structure of a scalar linear ODE."""

    operator: LinearDifferentialOperator
    assumptions: sp.Expr
    points: tuple[SingularityPointStructure, ...]

    def at(self, point: sp.Expr) -> SingularityPointStructure | None:
        point = sp.sympify(point)
        return next((item for item in self.points if item.singularity.point == point), None)


def _coefficient_singularities(
    operator: LinearDifferentialOperator,
) -> tuple[tuple[sp.Expr, ...], bool]:
    """Return finite coefficient singularities and whether the list is exhaustive."""
    x = operator.variable
    points: set[sp.Expr] = set()
    exhaustive = True
    for coeff in operator.normalized().coefficients[:-1]:
        expr = sp.cancel(sp.together(coeff))
        _, den = sp.fraction(expr)
        try:
            poly = sp.Poly(den, x)
        except (sp.PolynomialError, TypeError, ValueError):
            exhaustive = False
            try:
                singular = sp.singularities(expr, x)
            except (NotImplementedError, ValueError):
                continue
            if isinstance(singular, sp.FiniteSet):
                points.update(singular)
            continue
        if poly.degree() <= 0:
            continue
        try:
            roots = sp.roots(poly.as_expr(), x, cubics=False, quartics=False)
        except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
            roots = {}
        if roots and sum(int(m) for m in roots.values()) == poly.degree():
            points.update(roots)
            continue
        try:
            all_roots = poly.all_roots()
        except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
            exhaustive = False
        else:
            if len(all_roots) == poly.degree():
                points.update(all_roots)
            else:
                exhaustive = False
    return tuple(sorted(points, key=sp.default_sort_key)), exhaustive


def frobenius_convergence(
    analysis: FrobeniusAnalysis,
    *,
    singular_points: tuple[sp.Expr, ...] | None = None,
) -> FrobeniusConvergence:
    """Return the coefficient-analyticity disk controlling a Frobenius series."""
    operator = analysis.operator
    x = operator.variable
    point = analysis.point
    if point == sp.oo:
        return FrobeniusConvergence(point, None, (), None, False, x)
    if singular_points is None:
        points, exhaustive = _coefficient_singularities(operator)
    else:
        points, exhaustive = tuple(singular_points), True
    others = tuple(p for p in points if zero_status(p - point) is not True)
    if not others:
        radius = sp.oo if exhaustive else None
        domain = sp.S.true if radius is sp.oo else None
        return FrobeniusConvergence(point, radius, (), domain, exhaustive, x)
    distances = tuple(sp.simplify(sp.Abs(p - point)) for p in others)
    radius = distances[0] if len(distances) == 1 else sp.Min(*distances)
    nearest = tuple(
        p
        for p, distance in zip(others, distances, strict=True)
        if zero_status(distance - radius) is True
    )
    domain = sp.Abs(x - point) < radius
    return FrobeniusConvergence(point, radius, nearest, domain, exhaustive, x)


def frobenius_local_monodromy(analysis: FrobeniusAnalysis) -> FrobeniusLocalMonodromy:
    """Derive local monodromy from a completed Frobenius analysis."""
    if not analysis.root_multiplicities or len(analysis.roots) != analysis.operator.order:
        return FrobeniusLocalMonodromy(
            analysis.point, None, False, None, "indicial roots are incomplete"
        )
    if not analysis.logarithm_may_be_required:
        diag = [sp.exp(2 * sp.pi * sp.I * root) for root in analysis.roots]
        return FrobeniusLocalMonodromy(
            analysis.point,
            sp.ImmutableMatrix(sp.diag(*diag)),
            True,
            False,
            "pure-power Frobenius basis gives diagonal local monodromy",
        )
    try:
        basis = logarithmic_frobenius_basis(
            analysis.operator, point=analysis.point, terms=max(analysis.terms, 8)
        )
        if not basis.complete or basis.dimension != analysis.operator.order:
            raise FormalBasisError("incomplete logarithmic Frobenius basis")
        monodromy = formal_monodromy(basis)
    except (FormalBasisError, NotImplementedError, ValueError):
        return FrobeniusLocalMonodromy(
            analysis.point,
            None,
            False,
            analysis.logarithm_required,
            "a complete logarithmic Frobenius basis could not be certified",
        )
    matrix = monodromy.local_matrix
    if matrix is None:
        return FrobeniusLocalMonodromy(
            analysis.point,
            None,
            False,
            basis.has_logarithms,
            "physical one-turn monodromy is unavailable on the ramified cover",
        )
    return FrobeniusLocalMonodromy(
        analysis.point,
        sp.ImmutableMatrix(matrix),
        True,
        basis.has_logarithms,
        "completed logarithmic Frobenius basis determines local monodromy",
    )


def _regular_point_structure(
    original: LinearDifferentialOperator,
    singularity: ODESingularity,
    assumptions: sp.Expr,
    terms: int,
    finite_points: tuple[sp.Expr, ...],
    finite_points_certified: bool,
) -> SingularityPointStructure:
    point = singularity.point
    coordinate = local_coordinate(original, point=point)
    local_operator = coordinate.operator if point == sp.oo else original
    local_point = sp.S.Zero if point == sp.oo else point
    try:
        frob = frobenius_analysis(
            local_operator, point=local_point, terms=terms, assumptions=assumptions
        )
    except (NotImplementedError, ValueError):
        return SingularityPointStructure(singularity, None, None, None, None, coordinate)
    monodromy = frobenius_local_monodromy(frob)
    if point == sp.oo:
        local_points, exhaustive = _coefficient_singularities(local_operator)
        convergence = frobenius_convergence(frob, singular_points=local_points)
        convergence = replace(
            convergence, point=sp.oo, certified=convergence.certified and exhaustive
        )
        monodromy = replace(monodromy, point=sp.oo)
    else:
        convergence = frobenius_convergence(frob, singular_points=finite_points)
        if not finite_points_certified:
            convergence = replace(convergence, certified=False)
    apparent = apparent_singularity_analysis(
        original,
        point=point,
        assumptions=assumptions,
        frobenius=frob,
        monodromy=monodromy,
    )
    return SingularityPointStructure(
        singularity, frob, convergence, apparent, monodromy, coordinate
    )


def singularity_structure(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    assumptions: sp.Expr | bool = True,
    terms: int = 8,
    include_infinity: bool = True,
) -> SingularityStructure:
    """Build one coherent local-analysis object for every singular point."""
    operator = _coerce_linear_operator(ode, function, variable)
    assumptions = normalize_assumptions(assumptions)
    global_data = analyze_ode_singularities(
        operator, include_infinity=include_infinity, assumptions=assumptions
    )
    singularities = list(global_data.finite)
    if global_data.infinity is not None:
        singularities.append(global_data.infinity)
    finite_points, finite_points_certified = _coefficient_singularities(operator)
    items: list[SingularityPointStructure] = []
    for singularity in singularities:
        if singularity.kind is ODESingularityKind.REGULAR:
            items.append(
                _regular_point_structure(
                    operator,
                    singularity,
                    assumptions,
                    terms,
                    finite_points,
                    finite_points_certified,
                )
            )
        else:
            items.append(SingularityPointStructure(singularity, None, None, None, None))
    return SingularityStructure(operator, assumptions, tuple(items))
