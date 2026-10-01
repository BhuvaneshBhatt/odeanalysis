"""Adversarial and metamorphic tests for parameterized system formal types."""

import sympy as sp

from odeanalysis import (
    FirstOrderSystem,
    analyze_system_singularity,
    system_formal_type_stratification,
    system_parameter_analysis,
)

x = sp.symbols("x")


def _signature_at(system, parameter, value, max_resonance_order=2):
    specialized = FirstOrderSystem(
        x, sp.ImmutableMatrix(sp.Matrix(system.matrix).subs(parameter, value))
    )
    local = analyze_system_singularity(specialized)
    multiplicities = (
        tuple(sorted(sp.Matrix(local.residue).eigenvals().values()))
        if local.residue is not None
        else ()
    )
    resonance_orders = tuple(
        sorted(
            abs(int(r.difference))
            for r in local.resonances
            if r.difference.is_Integer and abs(int(r.difference)) <= max_resonance_order
        )
    )
    return local.kind, local.leading_rank, multiplicities, resonance_orders


def test_residue_family_cells_have_independently_constant_discrete_signature():
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(a / x, -a / x))
    result = system_formal_type_stratification(system, (a,), max_resonance_order=2)
    assert result.exhaustive
    # Check two interior points in every open interval cell.  Equality cells
    # already have a unique parameter value.
    probes = {
        "a < -1": (-2, -3),
        "(a > -1) & (a < -1/2)": (sp.Rational(-3, 4), sp.Rational(-2, 3)),
        "(a > -1/2) & (a < 0)": (sp.Rational(-1, 4), sp.Rational(-1, 3)),
        "(a > 0) & (a < 1/2)": (sp.Rational(1, 4), sp.Rational(1, 3)),
        "(a > 1/2) & (a < 1)": (sp.Rational(3, 4), sp.Rational(2, 3)),
        "a > 1": (2, 3),
    }
    for stratum in result.strata:
        values = probes.get(str(stratum.condition))
        if values:
            left = _signature_at(system, a, values[0])
            right = _signature_at(system, a, values[1])
            assert left == right
            assert left[:2] == (
                stratum.signature.singularity_kind,
                stratum.signature.leading_rank,
            )


def test_parameter_reparameterization_preserves_transition_values():
    a = sp.symbols("a", real=True)
    u = sp.symbols("u", real=True)
    original = FirstOrderSystem(x, sp.ImmutableMatrix.diag(a / x, -a / x))
    renamed = FirstOrderSystem(x, sp.ImmutableMatrix.diag(u / x, -u / x))
    left = system_parameter_analysis(original, (a,), max_resonance_order=2)
    right = system_parameter_analysis(renamed, (u,), max_resonance_order=2)
    lf = {sp.factor(p.subs(a, u)) for p in left.transition_polynomials}
    rf = {sp.factor(p) for p in right.transition_polynomials}

    # Compare zero sets up to nonzero scalar multiples via monic polynomials.
    def monic(expr):
        poly = sp.Poly(expr, u)
        return sp.Poly(poly.monic(), u).as_expr()

    assert {monic(p) for p in lf} == {monic(p) for p in rf}


def test_rank_collision_and_resonance_loci_are_distinct_categories():
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[a / x, 0], [0, (a + 1) / x]]))
    result = system_parameter_analysis(system, (a,), max_resonance_order=2)
    assert result.rank_loci
    # Eigenvalue difference is identically one, so no parameter-dependent
    # collision/resonance hypersurface should be invented.
    assert result.collision_loci == ()
    assert result.resonance_loci == ()


def test_stokes_transition_equations_are_invariant_under_common_exponential_shift():
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
    left = system_parameter_analysis(system, (a,), exponential_parts=(a / x, -a / x))
    right = system_parameter_analysis(system, (a,), exponential_parts=((a + 3) / x, (3 - a) / x))

    def monic_set(items):
        out = set()
        for p in items:
            poly = sp.Poly(p, a)
            out.add(poly.monic().as_expr())
        return out

    assert monic_set(left.stokes_loci) == monic_set(right.stokes_loci)


def test_irregular_parameter_formal_types_are_representative_not_overcertified():
    a = sp.symbols("a", real=True)
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(a / x**2, -a / x**2))
    result = system_formal_type_stratification(system, (a,))
    assert result.strata
    assert any(s.signature.singularity_kind == "irregular" for s in result.strata)
    assert not result.exhaustive
    assert all(
        not s.certified for s in result.strata if s.signature.singularity_kind == "irregular"
    )
