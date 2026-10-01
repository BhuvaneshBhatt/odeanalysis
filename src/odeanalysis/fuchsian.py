"""Global Fuchsian invariants, Riemann schemes, and apparent singularities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp
from funcprops import normalize_assumptions

from .formal_basis import FormalBasisError, logarithmic_frobenius_basis
from .operator import LinearDifferentialOperator, _coerce_linear_operator

if TYPE_CHECKING:
    from .frobenius import FrobeniusAnalysis
    from .local_structure import FrobeniusLocalMonodromy

from .singularities import (
    ODESingularity,
    ODESingularityKind,
    analyze_ode_singularities,
    classify_ode_point,
)


@dataclass(frozen=True)
class ApparentSingularityAnalysis:
    """Decision data for whether a regular singular point is apparent.

    ``apparent`` is ``True`` only when the complete local exponent multiset is
    nonnegative integral and a complete logarithmic Frobenius basis has no
    logarithms.  ``None`` means the symbolic data were insufficient to decide.
    """

    point: sp.Expr
    kind: ODESingularityKind
    apparent: bool | None
    exponents: tuple[sp.Expr, ...]
    has_logarithms: bool | None
    basis_dimension: int | None
    reason: str


@dataclass(frozen=True)
class RiemannSchemePoint:
    """One singular column of a scalar Fuchsian Riemann scheme."""

    point: sp.Expr
    exponents: tuple[sp.Expr, ...]
    indicial_polynomial: sp.Expr
    apparent: bool | None = None


@dataclass(frozen=True)
class FuchsRelation:
    """Exact exponent-sum relation for a scalar Fuchsian equation."""

    order: int
    singularity_count: int
    exponent_sum: sp.Expr
    expected_sum: sp.Expr
    residual: sp.Expr
    holds: bool | None


@dataclass(frozen=True)
class RiemannScheme:
    """Riemann scheme of a scalar equation Fuchsian on the Riemann sphere."""

    order: int
    variable: sp.Symbol
    singularities: tuple[RiemannSchemePoint, ...]
    fuchs_relation: FuchsRelation

    @property
    def points(self) -> tuple[sp.Expr, ...]:
        """Return singular points in the order used by the scheme."""

        return tuple(item.point for item in self.singularities)

    @property
    def exponent_columns(self) -> tuple[tuple[sp.Expr, ...], ...]:
        """Return one local-exponent column per singular point."""

        return tuple(item.exponents for item in self.singularities)


def _zero_decision(expression: sp.Expr) -> bool | None:
    value = sp.simplify(expression)
    if value == 0 or value.is_zero is True:
        return True
    if value.is_zero is False:
        return False
    return None


def _complete_exponents(singularity: ODESingularity) -> tuple[sp.Expr, ...] | None:
    if singularity.indicial_polynomial is None:
        return None
    r = sp.Symbol("r")
    try:
        roots = sp.roots(singularity.indicial_polynomial, r, cubics=False, quartics=False)
    except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
        roots = {}
    if roots and sum(int(mult) for mult in roots.values()) == singularity.order:
        return tuple(
            root
            for root, mult in sorted(roots.items(), key=lambda item: sp.default_sort_key(item[0]))
            for _ in range(int(mult))
        )
    if len(singularity.indicial_roots) == singularity.order:
        return tuple(sorted(singularity.indicial_roots, key=sp.default_sort_key))
    return None


def apparent_singularity_analysis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    assumptions: sp.Expr | bool = True,
    frobenius: FrobeniusAnalysis | None = None,
    monodromy: FrobeniusLocalMonodromy | None = None,
) -> ApparentSingularityAnalysis:
    """Determine whether ``point`` is an apparent regular singularity.

    Completed Frobenius and monodromy evidence is reused when supplied, so the
    apparentness decision cannot diverge from the unified local analysis.
    """
    operator = _coerce_linear_operator(ode, function, variable)
    assumptions = normalize_assumptions(assumptions)
    requested_point = sp.sympify(point)
    if requested_point == sp.oo:
        from .newton import localize_operator

        localized = localize_operator(operator, point=sp.oo)
        local = classify_ode_point(localized.operator, point=0, assumptions=assumptions)
    else:
        local = classify_ode_point(operator, point=requested_point, assumptions=assumptions)
    if local.kind is ODESingularityKind.ORDINARY:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            False,
            (),
            False,
            operator.order,
            "the point is ordinary rather than a singularity",
        )
    if local.kind is ODESingularityKind.IRREGULAR:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            False,
            (),
            None,
            None,
            "an irregular singularity cannot be apparent",
        )
    if local.kind is ODESingularityKind.UNKNOWN:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            None,
            (),
            None,
            None,
            "the local singularity type is unresolved",
        )

    exponents = (
        tuple(frobenius.roots)
        if frobenius is not None and len(frobenius.roots) == operator.order
        else _complete_exponents(local)
    )
    if exponents is None:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            None,
            (),
            None,
            None,
            "the complete indicial root multiset is unresolved",
        )
    for exponent in exponents:
        if exponent.is_integer is False or exponent.is_nonnegative is False:
            return ApparentSingularityAnalysis(
                requested_point,
                local.kind,
                False,
                exponents,
                None,
                None,
                "at least one local exponent is not a nonnegative integer",
            )
        if exponent.is_integer is not True or exponent.is_nonnegative is not True:
            return ApparentSingularityAnalysis(
                requested_point,
                local.kind,
                None,
                exponents,
                None,
                None,
                "integrality or nonnegativity of a local exponent is unresolved",
            )

    if monodromy is not None and monodromy.certified and monodromy.matrix is not None:
        identity = sp.eye(operator.order)
        trivial = all(
            _zero_decision(monodromy.matrix[i, j] - identity[i, j]) is True
            for i in range(operator.order)
            for j in range(operator.order)
        )
        if not trivial:
            return ApparentSingularityAnalysis(
                requested_point,
                local.kind,
                False,
                exponents,
                monodromy.logarithmic,
                operator.order,
                "certified local monodromy is nontrivial",
            )
        if monodromy.logarithmic is False:
            return ApparentSingularityAnalysis(
                requested_point,
                local.kind,
                True,
                exponents,
                False,
                operator.order,
                "nonnegative integral exponents and trivial certified local monodromy",
            )

    if frobenius is not None and frobenius.logarithm_required:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            False,
            exponents,
            True,
            len(frobenius.branches),
            "the completed Frobenius recurrence has a logarithmic obstruction",
        )

    integer_exponents = [int(exponent) for exponent in exponents]
    resonance_span = max(integer_exponents) - min(integer_exponents) if exponents else 0
    terms = max(8, operator.order + resonance_span + 3)
    basis_operator = frobenius.operator if frobenius is not None else operator
    basis_point = frobenius.point if frobenius is not None else requested_point
    try:
        basis = logarithmic_frobenius_basis(basis_operator, point=basis_point, terms=terms)
    except (FormalBasisError, NotImplementedError, ValueError):
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            None,
            exponents,
            None,
            None,
            "a complete logarithmic Frobenius basis could not be certified",
        )
    if not basis.complete or basis.dimension != operator.order:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            None,
            exponents,
            basis.has_logarithms,
            basis.dimension,
            "the local Frobenius basis is incomplete",
        )
    if basis.has_logarithms:
        return ApparentSingularityAnalysis(
            requested_point,
            local.kind,
            False,
            exponents,
            True,
            basis.dimension,
            "logarithmic local solutions give nontrivial local monodromy",
        )
    return ApparentSingularityAnalysis(
        requested_point,
        local.kind,
        True,
        exponents,
        False,
        basis.dimension,
        "all local solutions are holomorphic and single-valued",
    )


def is_apparent_singularity(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    assumptions: sp.Expr | bool = True,
) -> bool | None:
    """Return ``True``, ``False``, or ``None`` for apparentness at ``point``."""

    return apparent_singularity_analysis(
        ode, function, variable, point=point, assumptions=assumptions
    ).apparent


def _relation_from_points(
    order: int,
    points: tuple[RiemannSchemePoint, ...],
) -> FuchsRelation:
    exponent_sum = sp.simplify(sum((sum(item.exponents) for item in points), sp.S.Zero))
    expected = sp.Rational((len(points) - 2) * order * (order - 1), 2)
    residual = sp.simplify(exponent_sum - expected)
    return FuchsRelation(
        order=order,
        singularity_count=len(points),
        exponent_sum=exponent_sum,
        expected_sum=expected,
        residual=residual,
        holds=_zero_decision(residual),
    )


def riemann_scheme(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> RiemannScheme:
    """Return the Riemann scheme of an equation Fuchsian on the sphere.

    Every singular point, including infinity when singular, must be regular
    singular with a completely resolved indicial root multiset.  Irregular or
    unresolved singularities are rejected rather than being displayed in a
    misleading P-symbol.
    """

    operator = _coerce_linear_operator(ode, function, variable)
    if not operator.is_homogeneous:
        raise ValueError("Riemann schemes require a homogeneous linear equation")
    analysis = analyze_ode_singularities(operator, include_infinity=True)
    singularities = list(analysis.finite)
    if analysis.infinity is not None and analysis.infinity.kind is not ODESingularityKind.ORDINARY:
        singularities.append(analysis.infinity)

    points: list[RiemannSchemePoint] = []
    for singularity in singularities:
        if singularity.kind is not ODESingularityKind.REGULAR:
            raise ValueError(
                f"Riemann scheme requires a Fuchsian equation; {singularity.point!s} "
                f"is {singularity.kind.value}"
            )
        exponents = _complete_exponents(singularity)
        if exponents is None or singularity.indicial_polynomial is None:
            raise NotImplementedError(
                f"could not resolve the complete indicial data at {singularity.point!s}"
            )
        apparent = apparent_singularity_analysis(operator, point=singularity.point).apparent
        points.append(
            RiemannSchemePoint(
                point=singularity.point,
                exponents=exponents,
                indicial_polynomial=singularity.indicial_polynomial,
                apparent=apparent,
            )
        )

    scheme_points = tuple(points)
    return RiemannScheme(
        order=operator.order,
        variable=operator.variable,
        singularities=scheme_points,
        fuchs_relation=_relation_from_points(operator.order, scheme_points),
    )


def fuchs_relation(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> FuchsRelation:
    """Return the exact Fuchs exponent-sum relation for a Fuchsian equation."""

    return riemann_scheme(ode, function, variable).fuchs_relation
