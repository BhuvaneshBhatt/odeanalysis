import pytest
import sympy as sp

from odeanalysis import (
    FirstOrderSystem,
    LinearDifferentialOperator,
    analyze_system_singularity,
    formal_system_analysis,
    local_monodromy,
    parameterized_turning_analysis,
    recognize_canonical_equation,
    scalar_system_correspondence,
    scalarize_system,
    system_formal_type_stratification,
)


def test_adaptive_formal_reduction_certificate_is_verified():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
    result = formal_system_analysis(
        system, adaptive=True, max_depth=1, max_adaptive_depth=4
    )
    assert result.certificate.verified
    assert result.certificate.attempted_depths
    assert result.certificate.ramification_index == result.ramification_index


def test_cyclic_scalarization_round_trip_companion():
    x = sp.symbols("x")
    y = sp.Function("y")
    op = LinearDifferentialOperator(x, y, (x, 1 + x, 1))
    correspondence = scalar_system_correspondence(op)
    assert correspondence.verify()
    assert correspondence.scalarization.complete


def test_noncyclic_output_is_explicit():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(1, 2))
    result = scalarize_system(system, output_row=sp.ImmutableMatrix([[1, 0]]))
    assert not result.complete
    assert result.operator is None
    assert result.limitation


def test_euler_cauchy_recognition_and_monodromy():
    x = sp.symbols("x")
    y = sp.Function("y")
    alpha, beta = sp.symbols("alpha beta")
    ode = x**2 * sp.diff(y(x), x, 2) + alpha * x * sp.diff(y(x), x) + beta * y(x)
    rec = recognize_canonical_equation(ode, y, x)
    assert rec is not None
    assert rec.family.value == "euler"
    assert rec.verify()
    monodromy = local_monodromy("euler", 0, alpha=alpha, beta=beta)
    assert monodromy.verify()


def test_parameterized_turning_confluence_strata():
    x = sp.symbols("x")
    a = sp.symbols("a", real=True)
    y = sp.Function("y")
    op = LinearDifferentialOperator(x, y, (-(x**2 - a), 0, 1))
    result = parameterized_turning_analysis(op, parameters=(a,))
    assert result.exhaustive
    assert result.transition_polynomials
    assert len(result.strata) >= 2
    assert any("airy" in stratum.uniform_families for stratum in result.strata)
    assert any("weber" in stratum.uniform_families for stratum in result.strata)


def test_formal_type_stratification_regular_singular_family():
    x = sp.symbols("x")
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[a / x, 0], [0, -a / x]]))
    result = system_formal_type_stratification(system, (a,), max_resonance_order=2)
    assert result.base.exhaustive
    assert result.strata
    assert {stratum.signature.singularity_kind for stratum in result.strata} <= {
        "ordinary",
        "regular_singular",
    }
    assert any(
        stratum.signature.singularity_kind == "regular_singular"
        for stratum in result.strata
    )


def test_constant_gauge_preserves_system_singularity_spectrum():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[1 / x, 0], [0, 2 / x]]))
    gauge = sp.ImmutableMatrix([[1, 1], [0, 1]])
    transformed = system.gauge_transform(gauge)
    left = analyze_system_singularity(system)
    right = analyze_system_singularity(transformed)
    assert left.kind == right.kind
    assert left.exponents == right.exponents


def test_singularity_analysis_rejects_unstratified_parameter_dependent_zero_entry():
    x = sp.symbols("x")
    a = sp.symbols("a")
    system = FirstOrderSystem(x, sp.diag(a / x**2, 1 / x))
    with pytest.raises(NotImplementedError, match="stratify or specialize"):
        analyze_system_singularity(system, 0)


def test_singularity_analysis_accepts_certified_nonzero_parameter_entry():
    x = sp.symbols("x")
    a = sp.symbols("a", nonzero=True)
    system = FirstOrderSystem(x, sp.diag(a / x**2, 1 / x))
    analysis = analyze_system_singularity(system, 0)
    assert analysis.pole_order == 2
    assert analysis.kind == "irregular"
