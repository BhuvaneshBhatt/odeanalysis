"""Stable mathematical invariants for canonical local ODE examples."""

from __future__ import annotations

import sympy as sp

from odeanalysis import (
    MatrixLaurentSeries,
    ODESingularityKind,
    analyze_ode_singularities,
    formal_block_diagonalize,
    levelt_structure,
    levelt_turrittin_reduce,
    moser_reduce,
)


def test_euler_cauchy_has_regular_singularity_and_logarithmic_levelt_block():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x)

    analysis = analyze_ode_singularities(ode, y, x)
    assert analysis.finite[0].kind is ODESingularityKind.REGULAR
    structure = levelt_structure(ode, y, x, point=0, terms=4)
    assert structure.complete
    assert structure.ramification_index == 1
    assert structure.blocks[0].has_logarithms


def test_bessel_one_retains_integer_resonance_at_zero():
    from odeanalysis import frobenius_analysis

    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
    result = frobenius_analysis(ode, y, x, point=0, terms=5)

    assert {root for root, _ in result.root_multiplicities} == {-1, 1}
    assert any(item.difference == 2 for item in result.resonances)


def test_airy_at_infinity_has_two_ramified_exponential_blocks():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    airy = sp.diff(y(x), x, 2) - x * y(x)
    structure = levelt_structure(airy, y, x, point=sp.oo, terms=4)

    assert structure.complete
    assert structure.ramification_index == 2
    assert tuple(block.dimension for block in structure.blocks) == (1, 1)
    assert {sp.simplify(block.exponential_polynomial) for block in structure.blocks} == {
        -2 * x ** sp.Rational(3, 2) / 3,
        2 * x ** sp.Rational(3, 2) / 3,
    }


def test_diagonal_irregular_system_splits_spectrally():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.diag(t**-2, -(t**-2)), t)
    reduction = formal_block_diagonalize(connection, max_power=3)

    assert reduction.complete
    assert reduction.block_dimensions == (1, 1)
    assert reduction.spectral_splits
    assert reduction.verify()


def test_nilpotent_irregular_system_uses_moser_reduction():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [1, 0]]), t)
    reduction = moser_reduce(connection, max_power=3)

    assert reduction.complete
    assert len(reduction.steps) == 1
    assert reduction.verify()


def test_fractional_shear_system_requires_ramification():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [t**-1, 0]]), t)
    direct = formal_block_diagonalize(connection, max_power=3)
    reduction = levelt_turrittin_reduce(connection, max_power=3, max_depth=1)

    assert not direct.complete
    assert reduction.complete
    assert reduction.ramification_index == 2
    assert reduction.final_diagonalization.block_dimensions == (1, 1)
    assert reduction.verify()


def test_rank_one_scalar_irregular_example_has_expected_newton_invariant():
    from odeanalysis.newton import differential_newton_polygon

    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)
    polygon = differential_newton_polygon(ode, y, x, point=0)

    assert polygon.katz_rank == 1
    assert polygon.irregularity == 2
