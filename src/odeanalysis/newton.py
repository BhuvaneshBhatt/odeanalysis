"""Differential Newton polygons for scalar linear ODEs.

The convention used here attaches the point ``(j, v(a_j) - j)`` to the term
``a_j D**j`` of a local operator ``L = sum_j a_j D**j``, where ``v`` is the
local order in a uniformizing coordinate ``h``.  Positive lower-edge slopes
are the Newton/Katz irregularity slopes.  A slope ``rho > 0`` gives a leading
logarithmic derivative of order ``h**(-(rho + 1))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import lcm

import sympy as sp

from ._local import local_coordinate
from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_FAILURES
from .operator import LinearDifferentialOperator


@dataclass(frozen=True)
class LocalizedOperator:
    """An operator expressed in a local coordinate at a finite point or infinity."""

    original_operator: LinearDifferentialOperator
    point: sp.Expr
    operator: LinearDifferentialOperator
    coordinate: sp.Expr
    local_variable: sp.Symbol

    def to_original(self, expression: sp.Expr) -> sp.Expr:
        """Rewrite a local expression in the original independent variable."""

        expression = sp.sympify(expression)
        x = self.original_operator.variable
        if self.point == sp.oo:
            return sp.simplify(expression.subs(self.local_variable, 1 / x))
        return sp.simplify(expression.subs(self.local_variable, x - self.point))


@dataclass(frozen=True)
class DifferentialNewtonPoint:
    """One point ``(j, v(a_j)-j)`` of the differential Newton polygon."""

    derivative_order: int
    coefficient_valuation: sp.Rational
    height: sp.Rational
    leading_coefficient: sp.Expr


@dataclass(frozen=True)
class DifferentialNewtonEdge:
    """One compact edge of the lower differential Newton polygon."""

    left: DifferentialNewtonPoint
    right: DifferentialNewtonPoint
    points: tuple[DifferentialNewtonPoint, ...]
    slope: sp.Rational

    @property
    def horizontal_length(self) -> int:
        return self.right.derivative_order - self.left.derivative_order

    @property
    def irregularity_slope(self) -> sp.Rational:
        return sp.Rational(max(sp.S.Zero, self.slope))

    @property
    def is_irregular(self) -> bool:
        return bool(self.slope > 0)

    def characteristic_polynomial(self, symbol: sp.Symbol | None = None) -> sp.Expr:
        """Return the edge characteristic polynomial in the logarithmic derivative."""

        z = symbol or sp.Symbol("lambda")
        return sp.factor(
            sp.expand(
                sum(point.leading_coefficient * z**point.derivative_order for point in self.points)
            )
        )


@dataclass(frozen=True)
class NewtonSlopePiece:
    """One slope block of the differential-module Newton filtration.

    ``multiplicity`` is the horizontal Newton length carried by the slope.
    Nonpositive polygon slopes are collected into the regular slope-zero block.
    """

    slope: sp.Rational
    multiplicity: int
    ramification_index: int
    edges: tuple[DifferentialNewtonEdge, ...] = ()

    @property
    def irregular(self) -> bool:
        """Return whether this filtration piece has positive irregular slope."""

        return bool(self.slope > 0)


@dataclass(frozen=True)
class SlopeFiltration:
    """Newton slope filtration of a localized scalar differential module."""

    point: sp.Expr
    rank: int
    pieces: tuple[NewtonSlopePiece, ...]
    katz_rank: sp.Rational
    poincare_rank: sp.Rational
    irregularity: sp.Rational
    ramification_index: int

    @property
    def slopes(self) -> tuple[sp.Rational, ...]:
        """Return slopes with horizontal multiplicity, in nondecreasing order."""

        return tuple(
            slope for piece in self.pieces for slope in (piece.slope,) * piece.multiplicity
        )

    @property
    def irregular_pieces(self) -> tuple[NewtonSlopePiece, ...]:
        """Return the positive-slope pieces of the filtration."""

        return tuple(piece for piece in self.pieces if piece.irregular)


@dataclass(frozen=True)
class DifferentialNewtonPolygon:
    """Lower Newton polygon of a localized scalar differential operator."""

    point: sp.Expr
    localized: LocalizedOperator
    points: tuple[DifferentialNewtonPoint, ...]
    vertices: tuple[DifferentialNewtonPoint, ...]
    edges: tuple[DifferentialNewtonEdge, ...]

    @property
    def irregular_edges(self) -> tuple[DifferentialNewtonEdge, ...]:
        return tuple(edge for edge in self.edges if edge.is_irregular)

    @property
    def is_irregular(self) -> bool:
        return bool(self.irregular_edges)

    @property
    def slopes(self) -> tuple[sp.Rational, ...]:
        """Return Newton slopes with horizontal multiplicity.

        Negative lower-hull slopes belong to the regular part and are reported
        as slope zero in the differential-module filtration.
        """

        return slope_filtration(self).slopes

    @property
    def poincare_rank(self) -> sp.Rational:
        """Return the Poincare rank of the natural Euler-scaled companion system.

        This presentation rank is gauge dependent.  The invariant rational
        highest Newton slope is :attr:`katz_rank` (also called the
        Poincare--Katz rank).
        """

        return _euler_companion_poincare_rank(self)

    @property
    def slope_filtration(self) -> SlopeFiltration:
        """Return the slope filtration represented by this Newton polygon."""

        return slope_filtration(self)

    @property
    def katz_rank(self) -> sp.Rational:
        if not self.irregular_edges:
            return sp.S.Zero
        return max(edge.irregularity_slope for edge in self.irregular_edges)

    @property
    def irregularity(self) -> sp.Rational:
        return sp.simplify(
            sum(edge.horizontal_length * edge.irregularity_slope for edge in self.irregular_edges)
        )

    @property
    def ramification_index(self) -> int:
        denominators = [int(edge.irregularity_slope.q) for edge in self.irregular_edges]
        result = 1
        for denominator in denominators:
            result = lcm(result, denominator)
        return result


def localize_operator(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> LocalizedOperator:
    """Express an operator in the shared coordinate vanishing at ``point``."""
    local = local_coordinate(ode, function, variable, point=point)
    return LocalizedOperator(
        local.original_operator,
        local.point,
        local.operator,
        local.variable,
        local.variable,
    )


def local_order_and_leading_coefficient(
    expression: sp.Expr,
    variable: sp.Symbol,
) -> tuple[sp.Rational, sp.Expr]:
    """Return finite local order and its leading coefficient at ``variable = 0``.

    Integer valuations cover meromorphic coefficients.  Rational valuations are
    also accepted so that Puiseux-local coefficients can participate in the
    Newton polygon.  Essential/oscillatory behavior with no finite rational
    valuation is rejected rather than assigned a misleading polygon point.
    """

    expression = sp.cancel(sp.together(sp.sympify(expression)))
    if expression == 0:
        raise ValueError("the zero coefficient has no finite Newton valuation")

    try:
        lead = analytic_powsimp(expression.as_leading_term(variable))
        exponent = sp.sympify(lead.as_powers_dict().get(variable, 0))
        if exponent.is_Rational:
            valuation = sp.Rational(exponent)
            coefficient = sp.simplify(sp.limit(expression / variable**valuation, variable, 0))
            if not coefficient.has(sp.oo, -sp.oo, sp.zoo, sp.nan) and coefficient != 0:
                return valuation, coefficient
    except SYMBOLIC_FAILURES:
        pass

    # Rational-function fallback avoids relying on series heuristics.
    try:
        num, den = sp.fraction(expression)
        pn = sp.Poly(num, variable)
        pd = sp.Poly(den, variable)

        def multiplicity(poly: sp.Poly) -> int:
            count = 0
            q = poly
            factor = sp.Poly(variable, variable)
            while q.degree() >= 1:
                quotient, remainder = divmod(q, factor)
                if not remainder.is_zero:
                    break
                count += 1
                q = quotient
            return count

        valuation = sp.Rational(multiplicity(pn) - multiplicity(pd))
        coefficient = sp.simplify(sp.limit(expression / variable**valuation, variable, 0))
        if not coefficient.has(sp.oo, -sp.oo, sp.zoo, sp.nan) and coefficient != 0:
            return valuation, coefficient
    except SYMBOLIC_FAILURES:
        pass

    raise NotImplementedError(
        f"could not determine a finite rational local valuation for coefficient {expression!s}"
    )


def _cross(
    a: DifferentialNewtonPoint,
    b: DifferentialNewtonPoint,
    c: DifferentialNewtonPoint,
) -> sp.Expr:
    return sp.expand(
        (b.derivative_order - a.derivative_order) * (c.height - a.height)
        - (b.height - a.height) * (c.derivative_order - a.derivative_order)
    )


def _edge_slope(a: DifferentialNewtonPoint, b: DifferentialNewtonPoint) -> sp.Rational:
    return sp.Rational(b.height - a.height, b.derivative_order - a.derivative_order)


def differential_newton_polygon(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> DifferentialNewtonPolygon:
    """Construct the lower differential Newton polygon at ``point``.

    Zero coefficients are omitted.  Multiplying the complete operator by a
    nonzero scalar coefficient translates all valuations vertically and hence
    leaves slopes, Katz rank, ramification, and formal exponential data intact.
    """

    localized = localize_operator(ode, function, variable, point=point)
    if not localized.operator.is_homogeneous:
        raise ValueError("differential Newton polygons require a homogeneous ODE")

    points: list[DifferentialNewtonPoint] = []
    for j, coefficient in enumerate(localized.operator.coefficients):
        if coefficient == 0:
            continue
        valuation, leading = local_order_and_leading_coefficient(
            coefficient, localized.local_variable
        )
        points.append(DifferentialNewtonPoint(j, valuation, valuation - j, leading))

    if len(points) < 2:
        raise ValueError("at least two nonzero derivative coefficients are required")

    hull: list[DifferentialNewtonPoint] = []
    for point_data in points:
        while len(hull) >= 2:
            cross = sp.simplify(_cross(hull[-2], hull[-1], point_data))
            if cross.is_nonpositive is True:
                hull.pop()
                continue
            if cross.is_positive is True:
                break
            raise NotImplementedError("could not order symbolic Newton-polygon valuations")
        hull.append(point_data)

    edges: list[DifferentialNewtonEdge] = []
    for left, right in pairwise(hull):
        slope = _edge_slope(left, right)
        on_edge = tuple(
            p
            for p in points
            if left.derivative_order <= p.derivative_order <= right.derivative_order
            and sp.simplify(
                (p.height - left.height) * (right.derivative_order - left.derivative_order)
                - (right.height - left.height) * (p.derivative_order - left.derivative_order)
            )
            == 0
        )
        edges.append(DifferentialNewtonEdge(left, right, on_edge, slope))

    return DifferentialNewtonPolygon(
        point=sp.sympify(point),
        localized=localized,
        points=tuple(points),
        vertices=tuple(hull),
        edges=tuple(edges),
    )


def _euler_companion_poincare_rank(
    polygon: DifferentialNewtonPolygon,
) -> sp.Rational:
    """Return the pole rank of the natural Euler-scaled companion system."""

    operator = polygon.localized.operator.normalized()
    h = polygon.localized.local_variable
    rank = sp.S.Zero
    for derivative_order in range(operator.order):
        coefficient = sp.cancel(
            sp.together(
                h ** (operator.order - derivative_order) * operator.coefficients[derivative_order]
            )
        )
        if coefficient == 0:
            continue
        valuation, _ = local_order_and_leading_coefficient(coefficient, h)
        rank = max(rank, -valuation)
    return sp.Rational(rank)


def slope_filtration(
    polygon_or_ode: DifferentialNewtonPolygon | sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> SlopeFiltration:
    """Return the Newton slope filtration at ``point``.

    The horizontal length of each lower-hull edge is its slope multiplicity.
    Every nonpositive edge contributes to the regular slope-zero piece; any
    rank not represented to the left of the first nonzero derivative
    coefficient is also regular.  Positive rational slopes are retained exactly.
    """

    polygon = (
        polygon_or_ode
        if isinstance(polygon_or_ode, DifferentialNewtonPolygon)
        else differential_newton_polygon(polygon_or_ode, function, variable, point=point)
    )
    by_slope: dict[sp.Rational, list[DifferentialNewtonEdge]] = {}
    multiplicities: dict[sp.Rational, int] = {}
    represented = 0
    for edge in polygon.edges:
        slope = sp.Rational(max(sp.S.Zero, edge.slope))
        by_slope.setdefault(slope, []).append(edge)
        multiplicities[slope] = multiplicities.get(slope, 0) + edge.horizontal_length
        represented += edge.horizontal_length

    residual_regular = polygon.localized.operator.order - represented
    if residual_regular < 0:
        raise ValueError("Newton edge lengths exceed the differential-operator rank")
    if residual_regular:
        multiplicities[sp.S.Zero] = multiplicities.get(sp.S.Zero, 0) + residual_regular
        by_slope.setdefault(sp.S.Zero, [])

    pieces = tuple(
        NewtonSlopePiece(
            slope=slope,
            multiplicity=multiplicities[slope],
            ramification_index=int(slope.q) if slope > 0 else 1,
            edges=tuple(by_slope[slope]),
        )
        for slope in sorted(multiplicities, key=sp.default_sort_key)
    )
    return SlopeFiltration(
        point=polygon.point,
        rank=polygon.localized.operator.order,
        pieces=pieces,
        katz_rank=polygon.katz_rank,
        poincare_rank=polygon.poincare_rank,
        irregularity=polygon.irregularity,
        ramification_index=polygon.ramification_index,
    )


def newton_slopes(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> tuple[sp.Rational, ...]:
    """Return the differential-module Newton slopes with multiplicity."""

    return slope_filtration(ode, function, variable, point=point).slopes


def katz_rank(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> sp.Rational:
    """Return the invariant rational Katz rank at ``point``."""

    return differential_newton_polygon(ode, function, variable, point=point).katz_rank


def poincare_rank(
    polygon_or_ode: DifferentialNewtonPolygon | sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> sp.Rational:
    """Return the Poincare rank of the natural Euler-scaled companion system.

    Unlike :func:`katz_rank`, this is a rank of a specified system
    presentation and may decrease after meromorphic gauge reduction.
    """

    polygon = (
        polygon_or_ode
        if isinstance(polygon_or_ode, DifferentialNewtonPolygon)
        else differential_newton_polygon(polygon_or_ode, function, variable, point=point)
    )
    return _euler_companion_poincare_rank(polygon)
