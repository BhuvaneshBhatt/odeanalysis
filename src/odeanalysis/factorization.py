"""Exact scalar differential-operator factorization."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.solvers.ode.riccati import solve_riccati

from .operator import LinearDifferentialOperator, _coerce_linear_operator


@dataclass(frozen=True)
class FirstOrderFactorization:
    """A factorization ``D^2+pD+q = (D+a)(D+b)`` over a coefficient field."""

    operator: LinearDifferentialOperator
    left_coefficient: sp.Expr
    right_coefficient: sp.Expr

    def verify(self) -> bool:
        """Verify the noncommutative differential-operator product exactly."""

        normalized = self.operator.normalized()
        p = normalized.coefficients[1]
        q = normalized.coefficients[0]
        a = self.left_coefficient
        b = self.right_coefficient
        x = self.operator.variable
        return (
            sp.simplify(a + b - p) == 0 and sp.simplify(sp.diff(b, x) + a * b - q) == 0
        )

    @property
    def logarithmic_derivative(self) -> sp.Expr:
        """Return ``w=-b`` for a solution ``y=exp(integral(w dx))``."""

        return sp.simplify(-self.right_coefficient)

    @property
    def solution(self) -> sp.Expr:
        """Return the exponential-integral solution supplied by the right factor."""

        x = self.operator.variable
        return sp.exp(sp.Integral(self.logarithmic_derivative, x))


def factor_differential_operator(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> tuple[FirstOrderFactorization, ...]:
    """Find all certified rational first-order factorizations of a second-order operator.

    The normalized equation ``y'' + p y' + q y = 0`` factors on the right when
    ``w=y'/y`` is a rational solution of ``w' + w**2 + p*w + q = 0``.
    """

    op = _coerce_linear_operator(ode, function, variable)
    if not op.is_homogeneous or op.order != 2:
        raise ValueError("factorization requires a homogeneous second-order operator")
    normalized = op.normalized()
    x = normalized.variable
    p, q = normalized.coefficients[1], normalized.coefficients[0]
    wfun = sp.Function("_odeanalysis_w")
    try:
        solutions = solve_riccati(wfun(x), x, -q, -p, -sp.S.One, gensol=False)
    except (ValueError, sp.PolynomialError, NotImplementedError):
        return ()
    factors: list[FirstOrderFactorization] = []
    for equality in solutions:
        w = sp.cancel(equality.rhs)
        b = sp.simplify(-w)
        a = sp.simplify(p + w)
        factor = FirstOrderFactorization(op, a, b)
        if factor.verify() and factor not in factors:
            factors.append(factor)
    return tuple(factors)


def is_reducible_operator(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> bool:
    """Return whether a rational first-order right factor is certified."""

    return bool(factor_differential_operator(ode, function, variable))
