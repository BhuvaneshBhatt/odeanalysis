"""Exact recognition tests for classical canonical second-order equations."""

import sympy as sp

from odeanalysis.canonical import (
    CanonicalEquationFamily,
    recognize_canonical_equation,
    transform_to_canonical,
)

x = sp.symbols("x")
y = sp.Function("y")


def test_recognizes_affine_airy_pullback():
    equation = sp.diff(y(x), x, 2) - 4 * (2 * x + 3) * y(x)
    result = transform_to_canonical(equation, y, x)
    assert result.family is CanonicalEquationFamily.AIRY
    assert sp.simplify(result.scale - 2) == 0
    assert sp.simplify(result.shift - 3) == 0
    assert result.verify()


def test_recognizes_shifted_scaled_bessel():
    t = x - 3
    equation = t**2 * sp.diff(y(x), x, 2) + t * sp.diff(y(x), x) + (4 * t**2 - 9) * y(x)
    result = transform_to_canonical(equation, y, x)
    assert result.family is CanonicalEquationFamily.BESSEL
    assert result.parameter_map["nu"] == 3
    assert sp.simplify(result.scale - 2) == 0
    assert sp.simplify(result.shift + 6) == 0
    assert result.verify()


def test_recognizes_modified_bessel_without_complex_relabeling():
    equation = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) - (4 * x**2 + 9) * y(x)
    result = transform_to_canonical(equation, y, x)
    assert result.family is CanonicalEquationFamily.MODIFIED_BESSEL
    assert result.parameter_map["nu"] == 3
    assert result.verify()


def test_recognizes_gauss_hypergeometric_parameters():
    a, b, c = sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2, 3)
    equation = (
        x * (1 - x) * sp.diff(y(x), x, 2) + (c - (a + b + 1) * x) * sp.diff(y(x), x) - a * b * y(x)
    )
    result = transform_to_canonical(equation, y, x)
    assert result.family is CanonicalEquationFamily.HYPERGEOMETRIC
    assert result.parameter_map == {"a": a, "b": b, "c": c}
    assert result.verify()


def test_recognizes_affine_confluent_hypergeometric_pullback():
    z = 2 * x - 4
    a, c = sp.Rational(1, 2), sp.Rational(2, 3)
    # Pull back z*u'' + (c-z)*u' - a*u = 0 under z=2x-4.
    equation = z * sp.diff(y(x), x, 2) + 2 * (c - z) * sp.diff(y(x), x) - 4 * a * y(x)
    result = transform_to_canonical(equation, y, x)
    assert result.family is CanonicalEquationFamily.CONFLUENT_HYPERGEOMETRIC
    assert result.parameter_map == {"a": a, "c": c}
    assert sp.simplify(result.scale - 2) == 0
    assert sp.simplify(result.shift + 4) == 0
    assert result.verify()


def test_unrecognized_equation_returns_none():
    equation = sp.diff(y(x), x, 2) + (x**2 + 1) * y(x)
    assert recognize_canonical_equation(equation, y, x) is None


def _gauged_projective_hypergeometric_equation():
    z = sp.symbols("z")
    a = sp.Rational(2, 5)
    b = sp.Rational(3, 7)
    c = sp.Rational(5, 6)
    z_of_x = sp.cancel(2 * (x - 2) / (x + 1))
    alpha = sp.Rational(1, 3)
    beta = -sp.Rational(1, 4)
    z_prime = sp.diff(z_of_x, x)
    canonical_p = sp.cancel((c - (a + b + 1) * z) / (z * (1 - z)))
    canonical_q = sp.cancel(-a * b / (z * (1 - z)))
    h = sp.cancel(alpha * z_prime / z_of_x - beta * z_prime / (1 - z_of_x))
    p = sp.cancel(z_prime * canonical_p.subs(z, z_of_x) - 2 * h - sp.diff(z_prime, x) / z_prime)
    q = sp.cancel(z_prime**2 * canonical_q.subs(z, z_of_x) - p * h - sp.diff(h, x) - h**2)
    return sp.diff(y(x), x, 2) + p * sp.diff(y(x), x) + q * y(x)


def test_recognizes_mobius_pullback_with_dependent_variable_gauge():
    result = transform_to_canonical(_gauged_projective_hypergeometric_equation(), y, x)
    assert result.family is CanonicalEquationFamily.HYPERGEOMETRIC
    assert not result.is_affine
    assert result.scale is None
    assert result.shift is None
    assert sp.simplify(result.dependent_gauge - 1) != 0
    assert result.verify()


def test_projective_recognition_maps_three_finite_singularities_to_standard_triple():
    result = transform_to_canonical(_gauged_projective_hypergeometric_equation(), y, x)
    images = []
    for point in (-1, 2, 5):
        value = sp.limit(result.variable_transform, x, point)
        images.append(value)
    finite_images = {value for value in images if value.is_finite is True}
    infinite_images = [value for value in images if value in (sp.oo, -sp.oo)]
    assert finite_images == {sp.S.Zero, sp.S.One}
    assert len(infinite_images) == 1
    assert result.verify()
