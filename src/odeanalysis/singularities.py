"""Singularity analysis for scalar linear ordinary differential equations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import sympy as sp
from funcprops import normalize_assumptions

from ._assumptions import assumption_substitutions
from ._local import linear_ode_data
from ._local import pole_order as _pole_order
from ._local import pole_order_upper_bound as _pole_order_upper_bound
from ._symbolic_errors import SYMBOLIC_FAILURES
from .operator import (
    LinearDifferentialOperator,
    _coerce_linear_operator,
    as_ode_expression,
    function_class,
)


class ODESingularityKind(StrEnum):
    """Local classification of a scalar linear ODE."""

    ORDINARY = "ordinary"
    REGULAR = "regular_singular"
    IRREGULAR = "irregular_singular"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ODESingularity:
    """Local singularity data for a scalar linear ODE."""

    point: sp.Expr
    kind: ODESingularityKind
    order: int
    normalized_coefficients: tuple[sp.Expr, ...]
    pole_orders: tuple[int | None, ...]
    indicial_polynomial: sp.Expr | None = None
    indicial_roots: tuple[sp.Expr, ...] = ()
    transformed_equation: sp.Expr | None = None

    @property
    def regular(self) -> bool:
        return self.kind in {ODESingularityKind.ORDINARY, ODESingularityKind.REGULAR}


@dataclass(frozen=True)
class ODESingularityAnalysis:
    """Singularity analysis of a scalar linear ODE."""

    equation: sp.Expr
    function: sp.FunctionClass
    variable: sp.Symbol
    order: int
    finite: tuple[ODESingularity, ...]
    infinity: ODESingularity | None = None
    operator: LinearDifferentialOperator | None = None

    @property
    def singular_points(self) -> tuple[sp.Expr, ...]:
        points = tuple(
            item.point for item in self.finite if item.kind is not ODESingularityKind.ORDINARY
        )
        if self.infinity is not None and self.infinity.kind is not ODESingularityKind.ORDINARY:
            points += (sp.oo,)
        return points


def _finite_singular_candidates(
    normalized: tuple[sp.Expr, ...], variable: sp.Symbol
) -> tuple[sp.Expr, ...]:
    points: set[sp.Expr] = set()
    for coeff in normalized[:-1]:
        try:
            singular = sp.singularities(coeff, variable)
        except (NotImplementedError, ValueError):
            continue
        if isinstance(singular, sp.FiniteSet):
            points.update(singular)
    return tuple(sorted(points, key=sp.default_sort_key))


def _indicial_data(
    normalized: tuple[sp.Expr, ...],
    order: int,
    variable: sp.Symbol,
    point: sp.Expr,
) -> tuple[sp.Expr | None, tuple[sp.Expr, ...]]:
    r = sp.Symbol("r")
    poly = sp.ff(r, order)
    for j in range(order):
        scale = (variable - point) ** (order - j)
        try:
            b = sp.simplify(sp.limit(scale * normalized[j], variable, point))
        except SYMBOLIC_FAILURES:
            return None, ()
        if b.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            return None, ()
        poly += b * sp.ff(r, j)
    poly = sp.factor(sp.expand(poly))
    roots: tuple[sp.Expr, ...] = ()
    try:
        root_dict = sp.roots(poly, r)
        if root_dict:
            roots = tuple(
                root for root, multiplicity in root_dict.items() for _ in range(multiplicity)
            )
    except SYMBOLIC_FAILURES:
        pass
    return poly, roots


def classify_ode_point(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    point: sp.Expr = 0,
    *,
    assumptions: sp.Expr | bool = True,
) -> ODESingularity:
    """Classify one finite point of a scalar linear ODE or operator."""

    data = linear_ode_data(ode, function, variable)
    operator = data.operator
    normalized = data.normalized_coefficients
    x = operator.variable
    order = operator.order
    point = sp.sympify(point)
    assumptions = normalize_assumptions(assumptions)
    substitutions = assumption_substitutions(assumptions)
    if substitutions:
        normalized = tuple(sp.cancel(c.subs(substitutions)) for c in normalized)
    pole_orders = (
        *tuple(
            _pole_order(
                normalized[j],
                x,
                point,
                assumptions=sp.S.true
                if (_pole_order_upper_bound(normalized[j], x, point) or 0) <= order - j
                else assumptions,
            )
            for j in range(order)
        ),
        0,
    )
    known = pole_orders[:-1]
    if all(value == 0 for value in known):
        kind = ODESingularityKind.ORDINARY
    elif all(
        (value is not None and value <= order - j)
        or (value is None and (_pole_order_upper_bound(normalized[j], x, point) or 0) <= order - j)
        for j, value in enumerate(known)
    ):
        kind = ODESingularityKind.REGULAR
    elif any(value is None for value in known):
        kind = ODESingularityKind.UNKNOWN
    else:
        kind = ODESingularityKind.IRREGULAR

    indicial = None
    roots: tuple[sp.Expr, ...] = ()
    if kind is ODESingularityKind.REGULAR:
        indicial, roots = _indicial_data(normalized, order, x, point)
    return ODESingularity(
        point=point,
        kind=kind,
        order=order,
        normalized_coefficients=normalized,
        pole_orders=pole_orders,
        indicial_polynomial=indicial,
        indicial_roots=roots,
    )


def change_ode_variable_reciprocal(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    new_function: sp.FunctionClass | sp.Expr | None = None,
    new_variable: sp.Symbol | None = None,
) -> sp.Expr:
    """Transform an ODE under ``variable = 1/new_variable``."""

    operator = _coerce_linear_operator(ode, function, variable)
    if new_function is None or new_variable is None:
        raise TypeError("new_function and new_variable are required")
    transformed = operator.reciprocal_transform(new_function, new_variable)
    return sp.factor(sp.together(transformed.expression))


def analyze_ode_singularities(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    include_infinity: bool = True,
    assumptions: sp.Expr | bool = True,
) -> ODESingularityAnalysis:
    """Find and classify singular points of a scalar linear ODE."""

    operator = _coerce_linear_operator(ode, function, variable)
    assumptions = normalize_assumptions(assumptions)
    normalized = operator.normalized().coefficients
    finite = tuple(
        classify_ode_point(operator, point=p, assumptions=assumptions)
        for p in _finite_singular_candidates(normalized, operator.variable)
    )

    infinity: ODESingularity | None = None
    if include_infinity:
        t = sp.Dummy("t", positive=True)
        u = sp.Function("_u")
        transformed_operator = operator.reciprocal_transform(u, t)
        local = classify_ode_point(transformed_operator, point=sp.S.Zero, assumptions=assumptions)
        infinity = ODESingularity(
            point=sp.oo,
            kind=local.kind,
            order=local.order,
            normalized_coefficients=local.normalized_coefficients,
            pole_orders=local.pole_orders,
            indicial_polynomial=local.indicial_polynomial,
            indicial_roots=local.indicial_roots,
            transformed_equation=transformed_operator.expression,
        )

    return ODESingularityAnalysis(
        equation=operator.expression,
        function=operator.function,
        variable=operator.variable,
        order=operator.order,
        finite=finite,
        infinity=infinity,
        operator=operator,
    )


def ode_singular_points(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    include_infinity: bool = True,
    assumptions: sp.Expr | bool = True,
) -> tuple[tuple[sp.Expr, ODESingularityKind], ...]:
    """Compact ``(point, classification)`` view of singularity analysis."""

    analysis = analyze_ode_singularities(
        ode,
        function,
        variable,
        include_infinity=include_infinity,
        assumptions=assumptions,
    )
    result = tuple((item.point, item.kind) for item in analysis.finite)
    if analysis.infinity is not None:
        result += ((sp.oo, analysis.infinity.kind),)
    return result


# Backward-compatible aliases for helpers that were private in 0.1.0.
_as_expression = as_ode_expression
_function_class = function_class
