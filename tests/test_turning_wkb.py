from dataclasses import replace

import sympy as sp

from odeanalysis.turning import (
    TurningPointKind,
    airy_uniformization,
    analyze_turning_points,
    classify_turning_point,
    liouville_normal_form,
    weber_uniformization,
    wkb_expansion,
)


def test_liouville_normal_form_removes_first_derivative_exactly():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) + 2 * sp.diff(y(x), x) + (1 - x) * y(x)

    normal = liouville_normal_form(ode, y, x)

    assert sp.simplify(normal.potential - x) == 0
    assert sp.simplify(normal.gauge_log_derivative + 1) == 0
    assert normal.verify()


def test_turning_points_resolve_multiplicity_and_kind():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * (x - 1) ** 2 * (x + 2) ** 3 * y(x)

    analysis = analyze_turning_points(ode, y, x)

    assert analysis.complete
    assert analysis.verify()
    data = {point.point: (point.multiplicity, point.kind) for point in analysis.points}
    assert data == {
        sp.Integer(-2): (3, TurningPointKind.HIGHER),
        sp.Integer(0): (1, TurningPointKind.SIMPLE),
        sp.Integer(1): (2, TurningPointKind.DOUBLE),
    }


def test_classify_turning_point_rejects_nonzero_potential():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    point = classify_turning_point(ode, y, x, point=0)
    assert point.kind is TurningPointKind.SIMPLE
    assert point.verify()

    try:
        classify_turning_point(ode, y, x, point=1)
    except ValueError as exc:
        assert "not a turning point" in str(exc)
    else:
        raise AssertionError("non-turning point should be rejected")


def test_wkb_riccati_recurrence_for_airy_potential():
    x = sp.symbols("x", positive=True)
    epsilon = sp.symbols("epsilon", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    plus, minus = wkb_expansion(ode, y, x, order=2, parameter=epsilon)

    assert plus.verify() and minus.verify()
    assert sp.simplify(plus.coefficients[0] - sp.sqrt(x)) == 0
    assert sp.simplify(minus.coefficients[0] + sp.sqrt(x)) == 0
    assert sp.simplify(plus.coefficients[1] + 1 / (4 * x)) == 0
    assert sp.simplify(minus.coefficients[1] + 1 / (4 * x)) == 0
    assert sp.simplify(plus.coefficients[2] + 5 / (32 * x ** sp.Rational(5, 2))) == 0
    assert sp.simplify(minus.coefficients[2] - 5 / (32 * x ** sp.Rational(5, 2))) == 0


def test_airy_uniformization_is_exact_for_linear_potential():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    reduction = airy_uniformization(ode, y, x, point=0)

    assert reduction.canonical_family == "airy"
    assert sp.simplify(reduction.variable_transform - x) == 0
    assert reduction.is_exact
    assert reduction.verify()


def test_airy_uniformization_retains_nonzero_residual_for_generic_simple_point():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * (1 + x) * y(x)

    reduction = airy_uniformization(ode, y, x, point=0)

    assert reduction.verify()
    assert not reduction.is_exact
    assert reduction.residual != 0


def test_weber_uniformization_is_exact_for_quadratic_potential():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x**2 * y(x)

    reduction = weber_uniformization(ode, y, x, point=0)

    assert reduction.canonical_family == "weber"
    assert sp.simplify(reduction.variable_transform - x) == 0
    assert reduction.is_exact
    assert reduction.verify()


def test_uniformizers_reject_wrong_turning_point_multiplicity():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    simple = sp.diff(y(x), x, 2) - x * y(x)
    double = sp.diff(y(x), x, 2) - x**2 * y(x)

    for function, ode in (
        (airy_uniformization, double),
        (weber_uniformization, simple),
    ):
        try:
            function(ode, y, x, point=0)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "uniformizer should enforce turning-point multiplicity"
            )


def test_uniform_reduction_verifier_rejects_corrupted_residual():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)
    reduction = airy_uniformization(ode, y, x, point=0)

    assert not replace(reduction, residual=sp.Integer(1)).verify()


def test_wkb_rejects_identically_zero_potential():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2)

    try:
        wkb_expansion(ode, y, x)
    except ValueError as exc:
        assert "nonzero normal-form potential" in str(exc)
    else:
        raise AssertionError("identically zero WKB potential should be rejected")
