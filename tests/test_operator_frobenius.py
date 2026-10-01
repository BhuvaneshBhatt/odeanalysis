import sympy as sp

from odeanalysis import (
    FrobeniusAnalysis,
    LinearDifferentialOperator,
    ODESingularityKind,
    analyze_ode_singularities,
    frobenius_analysis,
)


def test_operator_uses_sympy_order_and_extracts_inhomogeneous_remainder():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = (1 + x) * sp.diff(y(x), x, 3) + x * sp.diff(y(x), x) + 2 * y(x) - sp.sin(x)
    op = LinearDifferentialOperator.from_ode(ode, y, x)
    assert op.order == 3
    assert op.coefficients == (sp.Integer(2), x, sp.Integer(0), x + 1)
    assert sp.simplify(op.inhomogeneous + sp.sin(x)) == 0
    monic = op.normalized()
    assert monic.leading_coefficient == 1
    assert sp.simplify(monic.coefficient(1) - x / (x + 1)) == 0


def test_singularity_analysis_accepts_operator_directly():
    x, nu = sp.symbols("x nu")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - nu**2) * y(x)
    op = LinearDifferentialOperator.from_ode(ode, y, x)
    analysis = analyze_ode_singularities(op)
    assert analysis.operator == op
    assert analysis.finite[0].kind is ODESingularityKind.REGULAR


def test_arbitrary_order_euler_frobenius_indicial_polynomial():
    x, a, b, c = sp.symbols("x a b c")
    y = sp.Function("y")
    ode = (
        x**3 * sp.diff(y(x), x, 3)
        + a * x**2 * sp.diff(y(x), x, 2)
        + b * x * sp.diff(y(x), x)
        + c * y(x)
    )
    result = frobenius_analysis(ode, y, x, point=0, terms=4)
    assert isinstance(result, FrobeniusAnalysis)
    r = result.indicial_variable
    expected = sp.ff(r, 3) + a * sp.ff(r, 2) + b * r + c
    assert sp.expand(result.indicial_polynomial - expected) == 0


def test_nontrivial_third_order_frobenius_recurrence():
    x = sp.symbols("x")
    y = sp.Function("y")
    # x^3 y''' + x^2 y'' - x y' + x^3 y = 0.
    # The regularized coefficient b_0=-x+x^3 contributes actual recurrence terms.
    ode = (
        x**3 * sp.diff(y(x), x, 3) + x**2 * sp.diff(y(x), x, 2) - x * sp.diff(y(x), x) + x**3 * y(x)
    )
    result = frobenius_analysis(ode, y, x, point=0, terms=5)
    assert result.indicial_polynomial is not None
    assert result.branches
    # Each returned branch is normalized with a_0=1 and has the requested prefix.
    assert all(branch.coefficients[0] == 1 for branch in result.branches)
    assert all(len(branch.coefficients) == 5 for branch in result.branches)


def test_bessel_integer_resonance_is_retained_not_divided_away():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
    result = frobenius_analysis(ode, y, x, point=0, terms=5)
    assert any(item.difference == 2 for item in result.resonances)
    smaller = next(branch for branch in result.branches if branch.exponent == -1)
    assert 2 in smaller.resonant_orders
    assert 2 in smaller.obstructed_orders
    assert smaller.logarithm_may_be_required


def test_repeated_indicial_root_flags_logarithmic_companion_possibility():
    x = sp.symbols("x")
    y = sp.Function("y")
    # x^2 y'' + x y' = 0 has indicial polynomial r^2.
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x)
    result = frobenius_analysis(ode, y, x, point=0, terms=3)
    assert result.root_multiplicities == ((sp.Integer(0), 2),)
    assert result.has_resonance
    assert result.branches[0].multiplicity == 2
    assert result.branches[0].logarithm_may_be_required


def test_operator_converts_to_sympy_holonomic_for_polynomial_coefficients():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
    op = LinearDifferentialOperator.from_ode(ode, y, x)
    holonomic = op.to_sympy_holonomic_operator()
    assert "Dx**2" in str(holonomic)
    assert "x**2 - 1" in str(holonomic)


def test_frobenius_diagnostics_explain_logarithmic_obstruction():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
    result = frobenius_analysis(ode, y, x, point=0, terms=5)
    diagnostics = [item for item in result.diagnostics if item.code == "logarithmic-obstruction"]
    assert diagnostics
    assert any(item.order == 2 for item in diagnostics)
