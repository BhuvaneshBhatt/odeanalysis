"""Certified rational scalar differential-operator factorization tests."""

import sympy as sp

from odeanalysis.factorization import (
    factor_differential_operator,
    is_reducible_operator,
)

x = sp.symbols("x")
y = sp.Function("y")


def test_constant_coefficient_operator_factors_both_ways():
    equation = sp.diff(y(x), x, 2) - y(x)
    factors = factor_differential_operator(equation, y, x)
    assert len(factors) == 2
    assert all(factor.verify() for factor in factors)
    assert {sp.simplify(f.logarithmic_derivative) for f in factors} == {-1, 1}


def test_euler_operator_has_rational_first_order_factors():
    equation = x**2 * sp.diff(y(x), x, 2) - 2 * y(x)
    factors = factor_differential_operator(equation, y, x)
    logarithmic_derivatives = {sp.cancel(f.logarithmic_derivative) for f in factors}
    assert -1 / x in logarithmic_derivatives
    assert any(
        sp.simplify(w.subs(next(iter(w.free_symbols - {x})), 0) - 2 / x) == 0
        for w in logarithmic_derivatives
        if w.free_symbols - {x}
    )
    assert all(f.verify() for f in factors)


def test_non_rational_riccati_example_reports_no_rational_factor():
    airy = sp.diff(y(x), x, 2) - x * y(x)
    assert factor_differential_operator(airy, y, x) == ()
    assert not is_reducible_operator(airy, y, x)
