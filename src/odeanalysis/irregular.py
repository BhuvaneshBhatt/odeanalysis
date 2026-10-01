"""Irregular-singularity invariants and leading WKB/exponential data."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_errors import SYMBOLIC_FAILURES
from .newton import (
    DifferentialNewtonEdge,
    DifferentialNewtonPolygon,
    differential_newton_polygon,
    poincare_rank,
)
from .operator import LinearDifferentialOperator
from .singularities import ODESingularityKind, classify_ode_point


@dataclass(frozen=True)
class IrregularSingularityInvariants:
    """Newton-theoretic invariants of an irregular scalar equation.

    ``katz_rank`` is the maximum positive Newton slope (the invariant
    Poincare--Katz rank). ``irregularity`` is the sum of each positive
    irregularity slope weighted by
    the horizontal length of its edge.  ``ramification_index`` is the least
    common multiple of the denominators of those rational slopes.

    ``euler_system_poincare_rank`` is additionally reported for the natural
    Euler-scaled companion system.  Unlike the preceding three quantities it is
    representation/gauge dependent, so it is named as such.
    """

    point: sp.Expr
    polygon: DifferentialNewtonPolygon
    katz_rank: sp.Rational
    irregularity: sp.Rational
    ramification_index: int
    euler_system_poincare_rank: sp.Rational

    @property
    def poincare_rank(self) -> sp.Rational:
        """Alias for the natural Euler-scaled companion-system rank."""

        return self.euler_system_poincare_rank


@dataclass(frozen=True)
class FormalExponentialPart:
    """Leading exponential part contributed by one irregular Newton edge."""

    point: sp.Expr
    edge: DifferentialNewtonEdge
    characteristic_variable: sp.Symbol
    characteristic_polynomial: sp.Expr
    characteristic_root: sp.Expr
    multiplicity: int
    logarithmic_derivative: sp.Expr
    local_exponent: sp.Expr
    exponent: sp.Expr
    ramification_index: int

    @property
    def exponential_factor(self) -> sp.Expr:
        return sp.exp(self.exponent)


@dataclass(frozen=True)
class WKBAnsatz:
    """A WKB-type ansatz based on a Newton-polygon exponential part."""

    point: sp.Expr
    exponential_part: FormalExponentialPart
    amplitude_function: sp.FunctionClass
    local_coordinate: sp.Expr
    local_expression: sp.Expr
    expression: sp.Expr
    conjugated_operator: LinearDifferentialOperator


def _known_roots(
    poly: sp.Expr,
    variable: sp.Symbol,
    expected_mult: int,
) -> tuple[tuple[sp.Expr, int], ...]:
    try:
        roots = sp.roots(poly, variable, cubics=False, quartics=False)
    except SYMBOLIC_FAILURES:
        roots = {}
    result = {
        sp.simplify(root): int(mult) for root, mult in roots.items() if sp.simplify(root) != 0
    }
    if sum(result.values()) != expected_mult:
        try:
            all_roots = sp.Poly(poly, variable).all_roots(radicals=False)
        except SYMBOLIC_FAILURES:
            all_roots = []
        if all_roots:
            result = {}
            for root in all_roots:
                root = sp.simplify(root)
                if root != 0:
                    result[root] = result.get(root, 0) + 1
    if sum(result.values()) != expected_mult:
        raise NotImplementedError(
            "could not resolve all nonzero roots of an irregular-edge characteristic polynomial"
        )
    return tuple(sorted(result.items(), key=lambda item: sp.default_sort_key(item[0])))


def irregular_singularity_invariants(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> IrregularSingularityInvariants:
    """Return Newton/Katz irregular-singularity invariants at ``point``."""

    polygon = differential_newton_polygon(ode, function, variable, point=point)
    kind = classify_ode_point(
        polygon.localized.operator,
        point=0,
    ).kind
    if kind is not ODESingularityKind.IRREGULAR and not polygon.is_irregular:
        raise ValueError(f"point {point!s} is not an irregular singular point")
    return IrregularSingularityInvariants(
        point=sp.sympify(point),
        polygon=polygon,
        katz_rank=polygon.katz_rank,
        irregularity=polygon.irregularity,
        ramification_index=polygon.ramification_index,
        euler_system_poincare_rank=poincare_rank(polygon),
    )


def formal_exponential_parts(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> tuple[FormalExponentialPart, ...]:
    """Return leading formal exponential parts from irregular Newton edges.

    For an edge of Newton/Katz slope ``rho > 0`` and edge characteristic root
    ``c``, the leading logarithmic derivative is ``c*h**(-(rho+1))``.
    Integration gives the leading exponential exponent

    ``Q = -c*h**(-rho)/rho``.

    This routine returns the Newton-determined *leading* formal
    exponential parts.  Lower terms in a full Hukuhara-Turrittin exponential
    polynomial require recursive Riccati/factor analysis and are not guessed.
    """

    polygon = differential_newton_polygon(ode, function, variable, point=point)
    h = polygon.localized.local_variable
    lam = sp.Symbol("lambda")
    result: list[FormalExponentialPart] = []

    for edge in polygon.irregular_edges:
        characteristic = edge.characteristic_polynomial(lam)
        for root, multiplicity in _known_roots(characteristic, lam, edge.horizontal_length):
            rho = edge.slope
            sigma = rho + 1
            log_derivative = sp.simplify(root * h ** (-sigma))
            local_q = sp.simplify(-root * h ** (-rho) / rho)
            q = polygon.localized.to_original(local_q)
            result.append(
                FormalExponentialPart(
                    point=sp.sympify(point),
                    edge=edge,
                    characteristic_variable=lam,
                    characteristic_polynomial=characteristic,
                    characteristic_root=root,
                    multiplicity=multiplicity,
                    logarithmic_derivative=log_derivative,
                    local_exponent=local_q,
                    exponent=q,
                    ramification_index=int(edge.irregularity_slope.q),
                )
            )
    return tuple(result)


def _exponential_gauge_transform(
    operator: LinearDifferentialOperator,
    exponent: sp.Expr,
    amplitude: sp.FunctionClass,
) -> LinearDifferentialOperator:
    """Return ``exp(-Q) L exp(Q)`` as an operator on the amplitude."""

    x = operator.variable
    v = amplitude(x)
    factor = sp.exp(exponent)
    transformed = sp.S.Zero
    for j, coefficient in enumerate(operator.coefficients):
        transformed += coefficient * sp.diff(factor * v, x, j) / factor
    transformed = sp.expand(transformed)
    if operator.inhomogeneous != 0:
        transformed += operator.inhomogeneous / factor
    return LinearDifferentialOperator.from_ode(transformed, amplitude, x)


def wkb_ansatze(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> tuple[WKBAnsatz, ...]:
    """Construct leading WKB-type ansätze ``y = exp(Q) A``.

    The returned conjugated operator is exact for the displayed leading Q and
    is intended as the starting point for subsequent amplitude/Frobenius or
    transseries analysis.
    """

    polygon = differential_newton_polygon(ode, function, variable, point=point)
    localized = polygon.localized
    h = localized.local_variable
    amplitude = sp.Function("_A")
    parts = formal_exponential_parts(ode, function, variable, point=point)
    ansatze: list[WKBAnsatz] = []
    for part in parts:
        local_expression = sp.exp(part.local_exponent) * amplitude(h)
        conjugated = _exponential_gauge_transform(
            localized.operator,
            part.local_exponent,
            amplitude,
        )
        original_expression = localized.to_original(local_expression)
        ansatze.append(
            WKBAnsatz(
                point=sp.sympify(point),
                exponential_part=part,
                amplitude_function=amplitude,
                local_coordinate=h,
                local_expression=local_expression,
                expression=original_expression,
                conjugated_operator=conjugated,
            )
        )
    return tuple(ansatze)
