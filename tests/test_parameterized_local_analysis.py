import sympy as sp

from odeanalysis import (
    ODESingularityKind,
    classify_ode_point,
    frobenius_analysis,
    local_parameter_analysis,
)


def test_assumptions_change_parameter_dependent_singularity_kind():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * sp.diff(y(x), x) + y(x)
    assert classify_ode_point(ode, y, x, 0).kind is ODESingularityKind.UNKNOWN
    assert (
        classify_ode_point(ode, y, x, 0, assumptions=sp.Eq(a, 0)).kind is ODESingularityKind.REGULAR
    )
    assert (
        classify_ode_point(ode, y, x, 0, assumptions=sp.Ne(a, 0)).kind
        is ODESingularityKind.IRREGULAR
    )


def test_parameterized_analysis_stratifies_coefficient_valuation():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * sp.diff(y(x), x) + y(x)
    result = local_parameter_analysis(ode, y, x, 0)
    assert result.exhaustive
    zero = result.select(sp.Eq(a, 0))
    nonzero = result.select(sp.Ne(a, 0))
    assert zero is not None and zero.singularity.kind is ODESingularityKind.REGULAR
    assert nonzero is not None and nonzero.singularity.kind is ODESingularityKind.IRREGULAR


def test_quadratic_indicial_discriminant_gets_parameter_strata():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * x * sp.diff(y(x), x) + b * y(x)
    result = local_parameter_analysis(ode, y, x, 0)
    d = sp.expand((a - 1) ** 2 - 4 * b)
    repeated = result.select(sp.Eq(d, 0))
    generic = result.select(sp.Ne(d, 0))
    assert repeated is not None and repeated.frobenius is not None
    assert generic is not None and generic.frobenius is not None
    assert repeated.frobenius.has_resonance
    assert repeated.frobenius.root_multiplicities[0][1] == 2


def test_frobenius_distinguishes_resonance_from_certified_log_obstruction():
    x = sp.symbols("x")
    y = sp.Function("y")
    euler = x**2 * sp.diff(y(x), x, 2) - x * sp.diff(y(x), x)
    pure = frobenius_analysis(euler, y, x, terms=4)
    assert pure.has_resonance
    assert not pure.logarithm_required

    bessel = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
    obstructed = frobenius_analysis(bessel, y, x, terms=5)
    assert obstructed.logarithm_required


def test_parameterized_analysis_at_infinity_uses_reciprocal_local_problem():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * x * sp.diff(y(x), x) + b * y(x)
    result = local_parameter_analysis(ode, y, x, sp.oo)
    assert result.exhaustive
    assert all(s.singularity.point == sp.oo for s in result.strata)
    assert all(s.singularity.kind is ODESingularityKind.REGULAR for s in result.strata)
