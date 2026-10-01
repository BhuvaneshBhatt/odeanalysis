"""Wronskian evolution and basis-independence certificates."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from funcprops import normalize_assumptions

from ._assumptions import zero_status
from ._local import linear_ode_data
from .operator import LinearDifferentialOperator


@dataclass(frozen=True)
class WronskianAnalysis:
    """Abel identity and optional explicit-basis independence evidence."""

    operator: LinearDifferentialOperator
    differential_equation: sp.Equality
    abel_factor: sp.Expr
    wronskian: sp.Expr | None
    independent: bool | None
    assumptions: sp.Expr
    witness_point: sp.Expr | None = None
    abel_identity_verified: bool | None = None


def abel_wronskian(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    constant: sp.Expr | None = None,
) -> sp.Expr:
    """Return Abel's general Wronskian factor for a homogeneous scalar ODE."""
    data = linear_ode_data(ode, function, variable)
    operator = data.operator
    if not operator.is_homogeneous:
        raise ValueError("Abel's identity requires a homogeneous equation")
    x = operator.variable
    a_nm1 = data.normalized_coefficients[operator.order - 1]
    c = sp.Symbol("C_W", nonzero=True) if constant is None else sp.sympify(constant)
    return c * sp.exp(-sp.Integral(a_nm1, x))


def _nonzero_witness(expr: sp.Expr, variable: sp.Symbol, assumptions: sp.Expr) -> sp.Expr | None:
    """Find a cheap exact point witnessing a nonzero expression."""
    for point in (sp.S.Zero, sp.S.One, -sp.S.One, sp.Integer(2)):
        try:
            value = sp.cancel(sp.together(expr.subs(variable, point)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if value.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            continue
        if zero_status(value, assumptions) is False:
            return point
    return None


def wronskian_analysis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    basis: tuple[sp.Expr, ...] | list[sp.Expr] | None = None,
    assumptions: sp.Expr | bool = True,
) -> WronskianAnalysis:
    """Return Abel evolution and certify independence for an explicit basis when possible."""
    data = linear_ode_data(ode, function, variable)
    operator = data.operator
    if not operator.is_homogeneous:
        raise ValueError("Wronskian analysis requires a homogeneous equation")
    assumptions = normalize_assumptions(assumptions)
    x = operator.variable
    a_nm1 = data.normalized_coefficients[operator.order - 1]
    w = sp.Function("W")(x)
    equation = sp.Eq(sp.diff(w, x), -a_nm1 * w)
    factor = abel_wronskian(operator)
    explicit = None
    independent = None
    witness = None
    abel_verified = None
    if basis is not None:
        if len(basis) != operator.order:
            raise ValueError("basis length must equal the ODE order")
        explicit = sp.factor(sp.wronskian(tuple(map(sp.sympify, basis)), x))
        z = zero_status(explicit, assumptions)
        independent = None if z is None else not z
        if independent is None:
            residual = sp.cancel(sp.together(sp.diff(explicit, x) + a_nm1 * explicit))
            abel_verified = zero_status(residual, assumptions)
            if abel_verified is True:
                witness = _nonzero_witness(explicit, x, assumptions)
                if witness is not None:
                    independent = True
        elif independent is True:
            witness = _nonzero_witness(explicit, x, assumptions)
    return WronskianAnalysis(
        operator,
        equation,
        factor,
        explicit,
        independent,
        assumptions,
        witness,
        abel_verified,
    )
