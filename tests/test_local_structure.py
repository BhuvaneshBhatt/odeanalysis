import sympy as sp

from odeanalysis import (
    ODESingularityKind,
    abel_wronskian,
    apparent_singularity_analysis,
    frobenius_analysis,
    frobenius_convergence,
    frobenius_local_monodromy,
    local_parameter_analysis,
    singularity_structure,
    wronskian_analysis,
)


def test_frobenius_convergence_uses_nearest_coefficient_singularity():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x * (1 - x) * sp.diff(y(x), x, 2) + (1 - 2 * x) * sp.diff(y(x), x) - y(x)
    data = frobenius_analysis(ode, y, x, point=0)
    conv = frobenius_convergence(data)
    assert conv.radius == 1
    assert conv.nearest_singularities == (1,)
    assert conv.domain == (sp.Abs(x) < 1)
    assert conv.certified


def test_apparent_singularity_accepts_assumptions_contract():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - sp.diff(y(x), x) / x
    result = apparent_singularity_analysis(ode, y, x, point=0, assumptions=True)
    assert result.apparent is True


def test_abel_identity_and_explicit_basis_independence():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) + sp.diff(y(x), x) / x
    factor = abel_wronskian(ode, y, x)
    assert (
        sp.simplify(sp.diff(sp.log(factor / sp.Symbol("C_W", nonzero=True)), x) + 1 / x)
        == 0
    )
    explicit = wronskian_analysis(
        ode, y, x, basis=(1, sp.log(x)), assumptions=sp.Q.positive(x)
    )
    assert sp.simplify(explicit.wronskian - 1 / x) == 0
    assert explicit.independent is True


def test_finite_resonance_strata_are_recorded_without_breaking_primary_partition():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * x * sp.diff(y(x), x) + b * y(x)
    result = local_parameter_analysis(ode, y, x, max_resonance_order=3)
    d = sp.factor((a - 1) ** 2 - 4 * b)
    conditions = {str(item.condition) for item in result.resonance_strata}
    assert str(sp.Eq(d, 1, evaluate=False)) in conditions
    assert str(sp.Eq(d, 4, evaluate=False)) in conditions
    assert str(sp.Eq(d, 9, evaluate=False)) in conditions
    assert result.select(sp.Ne(d, 0)) is not None


def test_unified_singularity_structure_collects_local_evidence():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - sp.diff(y(x), x) / x
    structure = singularity_structure(ode, y, x)
    zero = structure.at(0)
    assert zero is not None
    assert zero.singularity.kind is ODESingularityKind.REGULAR
    assert zero.frobenius is not None
    assert zero.apparent is not None and zero.apparent.apparent is True
    assert zero.monodromy is not None and zero.monodromy.certified
    assert zero.monodromy.matrix == sp.eye(2)


def test_local_monodromy_from_nonresonant_frobenius_data_is_diagonal():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) - sp.Rational(1, 4) * y(x)
    data = frobenius_analysis(ode, y, x)
    monodromy = frobenius_local_monodromy(data)
    assert monodromy.certified
    assert monodromy.matrix == sp.diag(-1, -1)
