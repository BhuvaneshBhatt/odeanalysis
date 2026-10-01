"""Metamorphic invariants that should survive harmless representation changes."""

from __future__ import annotations

import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from odeanalysis import LinearDifferentialOperator, analyze_ode_singularities
from odeanalysis.newton import differential_newton_polygon
from odeanalysis.singularities import change_ode_variable_reciprocal


@given(st.integers(min_value=-7, max_value=7).filter(lambda value: value != 0))
def test_nonzero_scalar_multiple_preserves_singularity_kinds(multiplier):
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + y(x)

    base = analyze_ode_singularities(ode, y, x)
    scaled = analyze_ode_singularities(multiplier * ode, y, x)
    assert tuple(item.kind for item in scaled.finite) == tuple(
        item.kind for item in base.finite
    )
    assert scaled.infinity.kind is base.infinity.kind


def test_algebraically_reordered_operator_has_same_newton_polygon():
    x = sp.symbols("x")
    y = sp.Function("y")
    terms = [x**4 * sp.diff(y(x), x, 2), -y(x), x**3 * sp.diff(y(x), x)]
    left = differential_newton_polygon(sum(terms), y, x, point=0)
    right = differential_newton_polygon(sum(reversed(terms)), y, x, point=0)

    assert left.points == right.points
    assert left.edges == right.edges
    assert left.katz_rank == right.katz_rank


def test_reciprocal_coordinate_change_is_consistent_with_infinity_localization():
    x, t = sp.symbols("x t")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)
    operator = LinearDifferentialOperator.from_ode(ode, y, x)
    u = sp.Function("u")

    transformed = change_ode_variable_reciprocal(
        operator, new_function=u, new_variable=t
    )
    infinity = analyze_ode_singularities(operator).infinity
    transformed_origin = analyze_ode_singularities(transformed, u, t).finite[0]

    assert infinity.kind is transformed_origin.kind


def test_constant_gauge_conjugation_preserves_formal_block_invariants():
    from odeanalysis import MatrixLaurentSeries, formal_block_diagonalize
    from odeanalysis._formal_gauge import constant_series, formal_gauge_transform

    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.diag(t**-2, -(t**-2)), t)
    change = constant_series(t, sp.Matrix([[1, 1], [1, -1]]))
    conjugated = formal_gauge_transform(connection, change, max_power=3)

    base = formal_block_diagonalize(connection, max_power=3)
    transformed = formal_block_diagonalize(conjugated, max_power=3)
    assert transformed.block_dimensions == base.block_dimensions
    assert tuple(split.dimensions for split in transformed.spectral_splits) == tuple(
        split.dimensions for split in base.spectral_splits
    )


def test_reciprocal_coordinate_change_applied_twice_recovers_operator():
    x, t = sp.symbols("x t")
    y = sp.Function("y")
    u = sp.Function("u")
    v = sp.Function("v")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    first = change_ode_variable_reciprocal(ode, y, x, u, t)
    second = change_ode_variable_reciprocal(first, u, t, v, x)
    recovered = LinearDifferentialOperator.from_ode(second, v, x).normalized()
    expected = LinearDifferentialOperator.from_ode(
        sp.diff(v(x), x, 2) - x * v(x), v, x
    ).normalized()

    assert all(
        sp.simplify(left - right) == 0
        for left, right in zip(
            recovered.coefficients, expected.coefficients, strict=True
        )
    )


def test_riemann_scheme_and_apparentness_ignore_nonzero_scalar_operator_factor():
    from odeanalysis import is_apparent_singularity, riemann_scheme

    x = sp.symbols("x")
    y = sp.Function("y")
    a = sp.Rational(1, 3)
    b = sp.Rational(1, 2)
    c = sp.Rational(2, 3)
    ode = (
        x * (1 - x) * sp.diff(y(x), x, 2)
        + (c - (a + b + 1) * x) * sp.diff(y(x), x)
        - a * b * y(x)
    )
    scaled = 7 * ode
    assert riemann_scheme(ode, y, x) == riemann_scheme(scaled, y, x)

    apparent = sp.diff(y(x), x, 2) - sp.diff(y(x), x) / x
    assert is_apparent_singularity(apparent, y, x, point=0) is True
    assert is_apparent_singularity(11 * apparent, y, x, point=0) is True
