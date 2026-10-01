import sympy as sp

from odeanalysis import (
    ODESingularityKind,
    analyze_ode_singularities,
    classify_ode_point,
    ode_singular_points,
)
from odeanalysis.singularities import change_ode_variable_reciprocal


def test_bessel_regular_zero_irregular_infinity():
    x, nu = sp.symbols("x nu")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - nu**2) * y(x)
    analysis = analyze_ode_singularities(ode, y, x)
    assert len(analysis.finite) == 1
    zero = analysis.finite[0]
    assert zero.point == 0
    assert zero.kind is ODESingularityKind.REGULAR
    r = sp.Symbol("r")
    assert sp.expand(zero.indicial_polynomial - (r**2 - nu**2)) == 0
    assert analysis.infinity is not None
    assert analysis.infinity.kind is ODESingularityKind.IRREGULAR


def test_airy_has_only_irregular_infinity():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)
    analysis = analyze_ode_singularities(ode, y, x)
    assert analysis.finite == ()
    assert analysis.infinity is not None
    assert analysis.infinity.kind is ODESingularityKind.IRREGULAR


def test_euler_regular_zero_and_regular_infinity():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * x * sp.diff(y(x), x) + b * y(x)
    analysis = analyze_ode_singularities(ode, y, x)
    assert analysis.finite[0].kind is ODESingularityKind.REGULAR
    assert analysis.infinity is not None
    assert analysis.infinity.kind is ODESingularityKind.REGULAR


def test_apparent_leading_coefficient_zero_is_removed_by_normalization():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x * sp.diff(y(x), x, 2) + x * y(x)
    analysis = analyze_ode_singularities(ode, y, x, include_infinity=False)
    assert analysis.finite == ()
    assert classify_ode_point(ode, y, x, 0).kind is ODESingularityKind.ORDINARY


def test_reciprocal_change_matches_chain_rule():
    x, t = sp.symbols("x t")
    y = sp.Function("y")
    u = sp.Function("u")
    transformed = change_ode_variable_reciprocal(sp.diff(y(x), x, 2) + y(x), y, x, u, t)
    expected = t**4 * sp.diff(u(t), t, 2) + 2 * t**3 * sp.diff(u(t), t) + u(t)
    assert sp.simplify(transformed - expected) == 0


def test_compact_api_includes_infinity():
    x = sp.symbols("x")
    y = sp.Function("y")
    result = ode_singular_points(sp.diff(y(x), x, 2) + y(x), y, x)
    assert result == ((sp.oo, ODESingularityKind.IRREGULAR),)
