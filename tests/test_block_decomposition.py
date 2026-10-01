from dataclasses import replace

import sympy as sp

from odeanalysis import (
    MatrixLaurentSeries,
    formal_block_diagonalize,
    formal_monodromy,
)
from odeanalysis.block_decomposition import (
    cyclic_scalar_operator,
    exponential_block_decomposition,
)
from odeanalysis.formal_basis import formal_logarithmic_basis


def _matrix_zero(matrix):
    return all(sp.simplify(entry) == 0 for entry in matrix)


def _mixed_repeated_operator():
    x = sp.symbols("x")
    y = sp.Function("y")
    # The solution space is spanned by
    # exp(1/x), exp(1/x)*log(x), exp(-1/x).
    a0 = (2 * x**2 - 3 * x + 2) / (x**6 * (x - 2))
    a1 = (2 * x**3 - 6 * x**2 - 5 * x + 2) / (x**4 * (x - 2))
    a2 = (4 * x**2 - 9 * x - 2) / (x**2 * (x - 2))
    ode = sp.diff(y(x), x, 3) + a2 * sp.diff(y(x), x, 2) + a1 * sp.diff(y(x), x) + a0 * y(x)
    return x, y, ode


def test_formal_block_diagonalization_removes_coupling_by_sylvester_recursion():
    t = sp.symbols("t")
    reduced = sp.diag(t**-3, t**-3, -(t**-3))
    reduced[0, 1] = t**-1
    coupling = sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    gauge = sp.eye(3) + t * coupling
    # If Y=GZ and Z'=BZ, then Y'=(GBG^-1+G'G^-1)Y.
    connection = gauge * reduced * gauge.inv() + gauge.diff(t) * gauge.inv()
    truncated = sp.Matrix(
        3,
        3,
        lambda i, j: sp.series(connection[i, j], t, 0, 5).removeO().expand(),
    )
    series = MatrixLaurentSeries.from_matrix(truncated, t)

    decomposition = formal_block_diagonalize(series, max_power=3)

    assert decomposition.complete
    assert decomposition.block_dimensions == (1, 2)
    assert decomposition.off_block_residual().is_zero
    assert len(decomposition.spectral_splits) == 1
    split = decomposition.spectral_splits[0]
    assert split.pivot_power == -3
    assert split.eigenvalues == (-1, 1)
    assert split.dimensions == (1, 2)
    assert all(_matrix_zero(P * P - P) for P in split.projectors)
    assert _matrix_zero(sum((sp.Matrix(P) for P in split.projectors), sp.zeros(3)) - sp.eye(3))
    assert split.verify()
    assert decomposition.verify()

    broken_projector = sp.ImmutableMatrix(sp.zeros(3))
    broken_evidence = replace(
        split.verification,
        projectors=(broken_projector, *split.projectors[1:]),
    )
    assert not replace(split, verification=broken_evidence).verify()

    broken_gauge = replace(
        decomposition.gauge_verification,
        transformed=decomposition.original_connection,
    )
    assert not replace(decomposition, gauge_verification=broken_gauge).verify()


def test_scalar_exponential_block_decomposition_isolates_repeated_mixed_block():
    x, y, ode = _mixed_repeated_operator()
    decomposition = exponential_block_decomposition(ode, y, x, max_power=6)

    assert decomposition.complete
    assert decomposition.limitation is None
    assert decomposition.ramification_index == 1
    assert decomposition.shearing_exponent == 2
    assert sorted(decomposition.block_dimensions) == [1, 2]
    assert decomposition.diagonalization.off_block_residual().is_zero

    repeated = next(block for block in decomposition.blocks if block.dimension == 2)
    simple = next(block for block in decomposition.blocks if block.dimension == 1)
    assert repeated.metadata is not None
    assert simple.metadata is not None
    assert sp.simplify(repeated.metadata.exponential_polynomial - 1 / x) == 0
    assert sp.simplify(simple.metadata.exponential_polynomial + 1 / x) == 0
    assert sp.simplify(repeated.parameter_exponential_polynomial - 1 / decomposition.parameter) == 0


def test_cyclic_scalarization_of_repeated_block_recovers_its_solution_space():
    x, y, ode = _mixed_repeated_operator()
    decomposition = exponential_block_decomposition(ode, y, x, max_power=6)
    repeated = next(block for block in decomposition.blocks if block.dimension == 2)
    operator = cyclic_scalar_operator(repeated.connection, repeated.output_row)
    t = operator.variable

    for solution in (sp.exp(1 / t), sp.exp(1 / t) * sp.log(t)):
        residual = sum(
            operator.coefficients[j] * sp.diff(solution, t, j) for j in range(operator.order + 1)
        )
        assert sp.simplify(residual) == 0


def test_mixed_repeated_irregular_basis_is_now_complete_and_has_monodromy():
    x, y, ode = _mixed_repeated_operator()
    basis = formal_logarithmic_basis(ode, y, x, terms=5)

    assert basis.complete
    assert basis.limitation is None
    assert basis.dimension == 3
    repeated = next(block for block in basis.blocks if block.dimension == 2)
    expressions = tuple(sp.simplify(vector.expression) for vector in repeated.basis_vectors)
    assert expressions == (sp.exp(1 / x), sp.exp(1 / x) * sp.log(x))

    monodromy = formal_monodromy(basis)
    assert monodromy.local_matrix == sp.Matrix([[1, 2 * sp.pi * sp.I, 0], [0, 1, 0], [0, 0, 1]])
