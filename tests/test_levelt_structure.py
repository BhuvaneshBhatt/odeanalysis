import sympy as sp

from odeanalysis import (
    MatrixLaurentSeries,
    formal_block_diagonalize,
    levelt_structure,
    moser_reduce,
)
from odeanalysis.levelt import levelt_reduce_regular_singular


def test_moser_shearing_reduces_nilpotent_single_eigenvalue_irregular_block():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [1, 0]]), t)

    reduction = moser_reduce(connection, max_power=3)

    assert reduction.complete
    assert reduction.verify()
    assert reduction.gauge_verification is not None
    assert len(reduction.steps) == 1
    step = reduction.steps[0]
    assert step.pivot_power == -2
    assert step.eigenvalue == 0
    assert step.nilpotent_rank_before == 1
    assert step.exponents == (0, 1)
    assert reduction.transformed_connection.to_matrix() == sp.Matrix(
        [[0, 1 / t], [1 / t, -1 / t]]
    )


def test_formal_block_diagonalizer_uses_moser_reducer_instead_of_stopping():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [1, 0]]), t)

    decomposition = formal_block_diagonalize(connection, max_power=3)

    assert decomposition.complete
    assert decomposition.limitation is None
    assert len(decomposition.moser_steps) == 1
    assert decomposition.block_dimensions == (2,)


def test_regular_singular_levelt_reduction_splits_semisimple_and_nilpotent_residue():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, 1 / t], [0, 0]]), t)

    reduction = levelt_reduce_regular_singular(connection, max_power=2)

    assert reduction.complete
    assert reduction.semisimple_residue == sp.zeros(2)
    assert reduction.nilpotent_residue == sp.Matrix([[0, 1], [0, 0]])
    assert reduction.resonant_terms == ()
    assert reduction.cover_monodromy_if_nonresonant == sp.Matrix(
        [[1, 2 * sp.pi * sp.I], [0, 1]]
    )


def test_levelt_integer_transform_absorbs_positive_degree_resonance():
    t, c = sp.symbols("t c")
    matrix = sp.diag(1 / t, 0)
    matrix[0, 1] = c
    connection = MatrixLaurentSeries.from_matrix(matrix, t)

    reduction = levelt_reduce_regular_singular(connection, max_power=2)

    assert reduction.complete
    assert reduction.resonant_terms == ()
    assert len(reduction.absorbed_resonant_terms) == 1
    term = reduction.absorbed_resonant_terms[0]
    assert term.power == 0
    assert term.nullity == 1
    assert sorted(reduction.integer_shifts) == [0, 1]
    assert reduction.semisimple_residue == sp.zeros(2)
    assert reduction.nilpotent_residue.rank() == 1
    assert reduction.nilpotent_residue**2 == sp.zeros(2)
    assert reduction.transformed_connection.coefficient(0) == sp.zeros(2)
    assert reduction.cover_monodromy_if_nonresonant == reduction.cover_monodromy


def test_levelt_integer_transform_absorbs_second_order_resonance():
    t, c = sp.symbols("t c")
    matrix = sp.diag(2 / t, 0)
    matrix[0, 1] = c * t
    connection = MatrixLaurentSeries.from_matrix(matrix, t)

    reduction = levelt_reduce_regular_singular(connection, max_power=3)

    assert reduction.complete
    assert reduction.resonant_terms == ()
    assert len(reduction.absorbed_resonant_terms) == 1
    assert reduction.absorbed_resonant_terms[0].power == 1
    assert sorted(reduction.integer_shifts) == [0, 2]
    assert reduction.semisimple_residue == sp.zeros(2)
    assert reduction.nilpotent_residue.rank() == 1
    assert reduction.transformed_connection.coefficient(1) == sp.zeros(2)


def test_levelt_precleanup_prevents_nonresonant_term_from_becoming_irregular():
    t, c, d = sp.symbols("t c d")
    matrix = sp.diag(2 / t, 0)
    matrix[0, 1] = d + c * t
    connection = MatrixLaurentSeries.from_matrix(matrix, t)

    reduction = levelt_reduce_regular_singular(connection, max_power=3)

    assert reduction.complete
    assert reduction.resonant_terms == ()
    assert all(power >= -1 for power, _ in reduction.transformed_connection.terms)
    assert reduction.nilpotent_residue.rank() == 1


def test_levelt_structure_recovers_logarithmic_nilpotent_part_and_monodromy():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x)

    structure = levelt_structure(ode, y, x, terms=5)

    assert structure.complete
    assert structure.dimension == 2
    block = structure.blocks[0]
    assert block.semisimple_exponent == sp.zeros(2)
    assert block.nilpotent_exponent == sp.Matrix([[0, 1], [0, 0]])
    assert block.formal_exponent_matrix == sp.Matrix([[0, 1], [0, 0]])
    assert block.cover_monodromy == sp.Matrix([[1, 2 * sp.pi * sp.I], [0, 1]])
    assert block.has_logarithms


def test_completed_moser_and_levelt_reductions_have_no_unresolved_diagnostics():
    t = sp.symbols("t")
    irregular = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [1, 0]]), t)
    moser = moser_reduce(irregular, max_power=3)
    assert moser.complete
    assert moser.diagnostics == ()

    regular = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, 1 / t], [0, 0]]), t)
    levelt = levelt_reduce_regular_singular(regular, max_power=2)
    assert levelt.complete
    assert levelt.diagnostics == ()


def test_levelt_turrittin_ramification_resolves_fractional_shear_block():
    from odeanalysis import levelt_turrittin_reduce

    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [t**-1, 0]]), t)

    base = formal_block_diagonalize(connection, max_power=3)
    assert not base.complete

    reduction = levelt_turrittin_reduce(connection, max_power=3, max_depth=1)

    assert reduction.complete
    assert reduction.ramification_index == 2
    assert reduction.final_diagonalization.block_dimensions == (1, 1)
    assert len(reduction.ramifications) == 1
    assert reduction.verify()
    assert reduction.ramifications[0].index == 2


def test_levelt_turrittin_ramifies_the_partially_reduced_connection():
    from odeanalysis import levelt_turrittin_reduce

    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-3], [1, 0]]), t)
    base = formal_block_diagonalize(connection, max_power=3)
    assert not base.complete
    assert len(base.moser_steps) == 1

    reduction = levelt_turrittin_reduce(connection, max_power=3, max_depth=1)

    assert reduction.complete
    assert reduction.ramification_index == 2
    assert len(reduction.formal_stages) == 2
    assert reduction.ramifications[0].source == base.transformed_connection
    assert reduction.verify()


def test_levelt_turrittin_verification_checks_composite_metadata():
    from dataclasses import replace

    from odeanalysis import levelt_turrittin_reduce

    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [t**-1, 0]]), t)
    reduction = levelt_turrittin_reduce(connection, max_power=3, max_depth=1)

    assert reduction.verify()
    assert not replace(reduction, ramification_index=3).verify()
    assert not replace(reduction, transformed_connection=connection).verify()
    assert not replace(reduction, complete=False).verify()
    assert not replace(reduction, formal_stages=()).verify()


def test_levelt_turrittin_zero_depth_never_ramifies():
    from odeanalysis import levelt_turrittin_reduce

    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [t**-1, 0]]), t)

    reduction = levelt_turrittin_reduce(connection, max_power=3, max_depth=0)

    assert reduction.ramification_index == 1
    assert reduction.ramifications == ()
    assert reduction.verify()


def test_ramification_step_verification_rejects_invalid_metadata_without_raising():
    from odeanalysis.block_decomposition import RamificationStep

    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[1 / t]]), t)
    step = RamificationStep(index=0, source=connection, transformed=connection)

    assert step.verify() is False
