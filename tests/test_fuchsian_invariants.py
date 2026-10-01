import sympy as sp

from odeanalysis import (
    ODESingularityKind,
    apparent_singularity_analysis,
    fuchs_relation,
    is_apparent_singularity,
    riemann_scheme,
)


def _hypergeometric_equation():
    x = sp.symbols("x")
    y = sp.Function("y")
    a = sp.Rational(1, 3)
    b = sp.Rational(1, 2)
    c = sp.Rational(2, 3)
    ode = (
        x * (1 - x) * sp.diff(y(x), x, 2) + (c - (a + b + 1) * x) * sp.diff(y(x), x) - a * b * y(x)
    )
    return x, y, ode


def test_gauss_hypergeometric_riemann_scheme_and_fuchs_relation():
    x, y, ode = _hypergeometric_equation()
    scheme = riemann_scheme(ode, y, x)

    assert scheme.points == (0, 1, sp.oo)
    assert scheme.exponent_columns == (
        (0, sp.Rational(1, 3)),
        (sp.Rational(-1, 6), 0),
        (sp.Rational(1, 3), sp.Rational(1, 2)),
    )
    assert scheme.fuchs_relation.exponent_sum == 1
    assert scheme.fuchs_relation.expected_sum == 1
    assert scheme.fuchs_relation.residual == 0
    assert scheme.fuchs_relation.holds is True
    assert fuchs_relation(ode, y, x) == scheme.fuchs_relation


def test_riemann_scheme_rejects_irregular_equation():
    x = sp.symbols("x")
    y = sp.Function("y")
    airy = sp.diff(y(x), x, 2) - x * y(x)

    try:
        riemann_scheme(airy, y, x)
    except ValueError as exc:
        assert "Fuchsian" in str(exc)
    else:
        raise AssertionError("Airy equation should not have a Fuchsian Riemann scheme")


def test_apparent_singularity_has_integral_exponents_and_no_logarithms():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - sp.diff(y(x), x) / x

    result = apparent_singularity_analysis(ode, y, x, point=0)
    assert result.kind is ODESingularityKind.REGULAR
    assert result.exponents == (0, 2)
    assert result.has_logarithms is False
    assert result.basis_dimension == 2
    assert result.apparent is True
    assert is_apparent_singularity(ode, y, x, point=0) is True


def test_logarithmic_regular_singularity_is_not_apparent():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) + sp.diff(y(x), x) / x

    result = apparent_singularity_analysis(ode, y, x, point=0)
    assert result.exponents == (0, 0)
    assert result.has_logarithms is True
    assert result.apparent is False


def test_negative_integral_exponent_is_not_holomorphic_apparent():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) - y(x)

    result = apparent_singularity_analysis(ode, y, x, point=0)
    assert set(result.exponents) == {-1, 1}
    assert result.apparent is False
    assert "nonnegative integer" in result.reason
