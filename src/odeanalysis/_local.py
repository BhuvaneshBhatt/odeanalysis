"""Shared local-coordinate, valuation, and linear-ODE data utilities."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from funcprops import normalize_assumptions

from ._assumptions import zero_status
from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_FAILURES
from .operator import LinearDifferentialOperator, _coerce_linear_operator


@dataclass(frozen=True)
class LinearODEData:
    """Cheap canonical extraction shared by scalar linear-ODE analyses."""

    operator: LinearDifferentialOperator
    normalized: LinearDifferentialOperator

    @property
    def variable(self) -> sp.Symbol:
        return self.operator.variable

    @property
    def order(self) -> int:
        return self.operator.order

    @property
    def coefficients(self) -> tuple[sp.Expr, ...]:
        return self.operator.coefficients

    @property
    def normalized_coefficients(self) -> tuple[sp.Expr, ...]:
        return self.normalized.coefficients


def linear_ode_data(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> LinearODEData:
    """Return canonical and monic operator data without property queries."""
    operator = _coerce_linear_operator(ode, function, variable)
    return LinearODEData(operator, operator.normalized())


@dataclass(frozen=True)
class LocalCoordinate:
    """Uniform local coordinate at a finite point or infinity."""

    original_operator: LinearDifferentialOperator
    point: sp.Expr
    operator: LinearDifferentialOperator
    variable: sp.Symbol

    @property
    def is_infinity(self) -> bool:
        return self.point == sp.oo

    def to_original(self, expression: sp.Expr) -> sp.Expr:
        x = self.original_operator.variable
        if self.is_infinity:
            return sp.simplify(sp.sympify(expression).subs(self.variable, 1 / x))
        return sp.simplify(sp.sympify(expression).subs(self.variable, x - self.point))


def local_coordinate(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> LocalCoordinate:
    """Localize an operator so the requested point is represented by ``t=0``."""
    operator = _coerce_linear_operator(ode, function, variable)
    point = sp.sympify(point)
    t = sp.Dummy("t", positive=True)
    u = sp.Function("_u")
    if point == sp.oo:
        local = operator.reciprocal_transform(u, t)
    else:
        coeffs = tuple(
            sp.cancel(sp.together(c.subs(operator.variable, point + t)))
            for c in operator.coefficients
        )
        remainder = sp.cancel(
            sp.together(operator.inhomogeneous.subs(operator.variable, point + t))
        )
        local = LinearDifferentialOperator(t, u, coeffs, remainder)
    return LocalCoordinate(operator, point, local, t)


def rational_valuation(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = 0,
    *,
    assumptions: sp.Expr | bool = True,
) -> int | None:
    """Return an exact integer local valuation for a rational expression when provable."""
    assumptions = normalize_assumptions(assumptions)
    t = sp.Dummy("t", positive=True)
    try:
        local = sp.cancel(sp.together(sp.sympify(expr).subs(variable, point + t)))
        num, den = sp.fraction(local)
        if zero_status(num, assumptions) is True:
            return None
        pn, pd = sp.Poly(num, t), sp.Poly(den, t)

        def valuation(poly: sp.Poly) -> int | None:
            for k in range(poly.degree() + 1):
                status = zero_status(poly.nth(k), assumptions)
                if status is False:
                    return k
                if status is None:
                    return None
            return None

        vn, vd = valuation(pn), valuation(pd)
        if vn is not None and vd is not None:
            return vn - vd
        return None
    except (sp.PolynomialError, TypeError, ValueError):
        pass
    try:
        local = sp.cancel(sp.together(sp.sympify(expr).subs(variable, point + t)))
        lead = analytic_powsimp(local.as_leading_term(t))
        exponent = sp.sympify(lead.as_powers_dict().get(t, 0))
        if exponent.is_Integer:
            return int(exponent)
    except SYMBOLIC_FAILURES:
        pass
    return None


def pole_order(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = 0,
    *,
    assumptions: sp.Expr | bool = True,
) -> int | None:
    """Return exact pole order, zero for analytic expressions, or ``None``."""
    expression = sp.cancel(sp.together(sp.sympify(expr)))
    if zero_status(expression, assumptions) is True:
        return 0
    valuation = rational_valuation(expression, variable, point, assumptions=assumptions)
    return None if valuation is None else max(0, -valuation)


def pole_order_upper_bound(expr: sp.Expr, variable: sp.Symbol, point: sp.Expr = 0) -> int | None:
    """Return a rational upper bound on local pole order."""
    t = sp.Dummy("t")
    try:
        local = sp.cancel(sp.together(sp.sympify(expr).subs(variable, point + t)))
        _, den = sp.fraction(local)
        poly = sp.Poly(den, t)
        valuation = next((k for k in range(poly.degree() + 1) if poly.nth(k) != 0), 0)
        return max(0, valuation)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
