import sympy as sp

from odeanalysis import (
    ODESingularityKind,
    frobenius_analysis,
    frobenius_convergence,
    singularity_structure,
    wronskian_analysis,
)


def test_symbolic_coefficient_singularity_gives_symbolic_certified_radius():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = x * sp.diff(y(x), x, 2) + sp.diff(y(x), x) + y(x) / (x - a)
    data = frobenius_analysis(ode, y, x, point=0)
    convergence = frobenius_convergence(data)
    assert convergence.coefficient_radius == sp.Abs(a)
    assert convergence.nearest_singularities == (a,)
    assert convergence.certified
    assert convergence.solution_continuation_radius is None
    assert not convergence.solution_continuation_certified


def test_multiple_symbolic_singularities_never_collapse_to_infinity():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = x * sp.diff(y(x), x, 2) + sp.diff(y(x), x) + y(x) / ((x - a) * (x - b))
    data = frobenius_analysis(ode, y, x, point=0)
    convergence = frobenius_convergence(data)
    assert convergence.coefficient_radius == sp.Min(sp.Abs(a), sp.Abs(b))
    assert convergence.coefficient_radius != sp.oo
    assert convergence.certified


def test_regular_singular_infinity_has_full_frobenius_and_monodromy_data():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) - sp.Rational(1, 4) * y(x)
    infinity = singularity_structure(ode, y, x).at(sp.oo)
    assert infinity is not None
    assert infinity.singularity.kind is ODESingularityKind.REGULAR
    assert infinity.local_coordinate is not None and infinity.local_coordinate.is_infinity
    assert infinity.frobenius is not None
    assert infinity.convergence is not None and infinity.convergence.point == sp.oo
    assert infinity.monodromy is not None and infinity.monodromy.certified
    assert infinity.monodromy.point == sp.oo


def test_parameter_assumptions_change_infinity_classification():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) + a * y(x)
    zero = singularity_structure(ode, y, x, assumptions=sp.Eq(a, 0)).at(sp.oo)
    nonzero = singularity_structure(ode, y, x, assumptions=sp.Ne(a, 0)).at(sp.oo)
    assert zero is not None and zero.singularity.kind is ODESingularityKind.REGULAR
    assert nonzero is not None and nonzero.singularity.kind is ODESingularityKind.IRREGULAR


def test_apparent_singularity_reuses_certified_trivial_monodromy():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - sp.diff(y(x), x) / x
    zero = singularity_structure(ode, y, x).at(0)
    assert zero is not None and zero.apparent is not None
    assert zero.apparent.apparent is True
    assert "monodromy" in zero.apparent.reason


def test_wronskian_independence_uses_symbolic_assumptions():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - a * sp.diff(y(x), x)
    result = wronskian_analysis(ode, y, x, basis=(1, sp.exp(a * x)), assumptions=sp.Ne(a, 0))
    assert result.independent is True
    assert result.wronskian == a * sp.exp(a * x)
