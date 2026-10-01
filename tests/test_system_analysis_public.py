import sympy as sp

from odeanalysis import (
    FirstOrderSystem,
    analyze_system_singularity,
    formal_system_analysis,
    system_parameter_analysis,
    system_stokes_geometry,
)


def test_regular_system_residue_exponents_and_resonance():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[0, 0], [0, 2]]) / x)
    result = analyze_system_singularity(system)
    assert result.kind == "regular_singular"
    assert result.residue == sp.ImmutableMatrix([[0, 0], [0, 2]])
    assert result.exponents == (0, 2)
    assert len(result.resonances) == 1
    assert abs(result.resonances[0].difference) == 2


def test_infinity_uses_exact_reciprocal_connection_transform():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[1 / x, 0], [0, 2 / x]]))
    result = analyze_system_singularity(system, sp.oo)
    assert result.kind == "regular_singular"
    assert result.residue == sp.ImmutableMatrix([[-1, 0], [0, -2]])
    assert result.exponents == (-2, -1)


def test_irregular_system_reports_raw_rank_and_formal_reduction_verifies():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[1 / x**2, 0], [0, -1 / x**2]]))
    local = analyze_system_singularity(system)
    assert local.kind == "irregular"
    assert local.poincare_rank == 1
    formal = formal_system_analysis(system, max_power=2)
    assert formal.complete
    assert formal.verify()


def test_system_stokes_geometry_is_structural_only():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[1 / x**2, 0], [0, -1 / x**2]]))
    formal = formal_system_analysis(system, max_power=2)
    assert set(formal.exponential_parts) == {
        -1 / formal.connection.variable,
        1 / formal.connection.variable,
    }
    geometry = system_stokes_geometry(formal)
    assert geometry.structural_only
    assert len(geometry.pairs) == 1
    assert len(geometry.pairs[0].equal_magnitude_rays) == 2
    assert len(geometry.pairs[0].phase_alignment_rays) == 2


def test_parameterized_system_finds_rank_and_collision_loci():
    x = sp.symbols("x")
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[a / x, 0], [0, -a / x]]))
    result = system_parameter_analysis(system, (a,))
    assert any(
        sp.simplify(p / a) in (1, -1, 2, -2, 4, -4)
        for p in result.transition_polynomials
        if p != 0
    )
    assert result.exhaustive
    assert result.strata


def test_parameterized_system_resonance_categories_are_explicit():
    x = sp.symbols("x")
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[a / x, 0], [0, -a / x]]))
    result = system_parameter_analysis(system, (a,), max_resonance_order=2)
    assert result.rank_loci
    assert result.collision_loci
    assert result.block_loci
    assert result.resonance_loci


def test_system_stokes_parameter_loci_use_real_coordinates():
    x = sp.symbols("x")
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[1 / x**2, 0], [0, -1 / x**2]]))
    result = system_parameter_analysis(system, (a,), exponential_parts=(a / x, -a / x))
    assert result.stokes_loci


def test_symbolic_integer_difference_is_not_overclaimed_as_nonzero_resonance():
    from odeanalysis.system_analysis import _resonances

    n = sp.symbols("n", integer=True)
    assert _resonances((n, sp.S.Zero)) == ()
