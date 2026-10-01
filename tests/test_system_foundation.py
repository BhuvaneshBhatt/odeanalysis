import sympy as sp

from odeanalysis import (
    FirstOrderSystem,
    MatrixLaurentSeries,
    complete_formal_exponential_parts,
)
from odeanalysis.system import (
    companion_system,
    formal_block_partition,
)


def _matrix_zero(matrix):
    return all(sp.simplify(entry) == 0 for entry in matrix)


def test_companion_system_preserves_scalar_equation_and_forcing():
    x = sp.symbols("x")
    y = sp.Function("y")
    p, q, r = sp.symbols("p q r")
    system = companion_system(
        sp.diff(y(x), x, 2) + p * sp.diff(y(x), x) + q * y(x) + r, y, x
    )

    assert system.dimension == 2
    assert system.matrix == sp.ImmutableMatrix([[0, 1], [-q, -p]])
    assert system.forcing == sp.ImmutableMatrix([[0], [-r]])
    assert not system.is_homogeneous


def test_exact_gauge_transform_formula():
    x = sp.symbols("x")
    A = sp.Matrix([[0, 1], [x, 0]])
    G = sp.Matrix([[sp.exp(x), 0], [0, 1 + x]])
    system = FirstOrderSystem(x, sp.ImmutableMatrix(A))
    transformed = system.gauge_transform(G)

    expected = G.inv() * A * G - G.inv() * G.diff(x)
    assert _matrix_zero(sp.Matrix(transformed.matrix) - expected)


def test_exponential_gauge_subtracts_scalar_log_derivative():
    x = sp.symbols("x")
    q = x**2 / 2
    A = sp.Matrix([[x, 1], [0, x + 1]])
    transformed = FirstOrderSystem(x, sp.ImmutableMatrix(A)).exponential_gauge(q)
    assert transformed.matrix == sp.ImmutableMatrix([[0, 1], [0, 1]])


def test_ramification_transforms_connection_and_accumulates_index():
    x, t, u = sp.symbols("x t u")
    A = sp.Matrix([[1 / x, x], [0, 2 / x]])
    system = FirstOrderSystem(x, sp.ImmutableMatrix(A))
    ramified = system.ramify(t, 2)

    assert ramified.variable == t
    assert ramified.ramification_index == 2
    assert ramified.matrix == sp.ImmutableMatrix([[2 / t, 2 * t**3], [0, 4 / t]])

    twice = ramified.ramify(u, 3)
    assert twice.ramification_index == 6
    assert sp.simplify(twice.matrix[0, 0] - 6 / u) == 0


def test_matrix_laurent_arithmetic_derivative_and_inverse():
    t = sp.symbols("t")
    A0 = sp.Matrix([[1, 1], [0, 1]])
    A1 = sp.Matrix([[1, 0], [2, -1]])
    series = MatrixLaurentSeries.from_mapping(t, {0: A0, 1: A1})
    inverse = series.inverse(max_power=4)

    product = series.multiply(inverse, max_power=4)
    assert product.coefficient(0) == sp.eye(2)
    for power in range(1, 5):
        assert product.coefficient(power).is_zero_matrix

    shifted = series.shift(-2)
    derivative = shifted.derivative(ramification_index=2)
    # Direct coefficient checks avoid branch-sensitive substitution simplification.
    assert derivative.coefficient(-4) == -A0
    assert derivative.coefficient(-3) == -sp.Rational(1, 2) * A1


def test_matrix_laurent_from_matrix_and_noncommutative_product():
    t = sp.symbols("t")
    matrix = sp.Matrix([[1 + t, t**-1], [2 * t**2, 3]])
    series = MatrixLaurentSeries.from_matrix(matrix, t)
    assert _matrix_zero(sp.Matrix(series.to_matrix()) - matrix)

    left = MatrixLaurentSeries.from_mapping(t, {0: sp.Matrix([[0, 1], [0, 0]])})
    right = MatrixLaurentSeries.from_mapping(t, {0: sp.Matrix([[0, 0], [1, 0]])})
    assert left.multiply(right).coefficient(0) == sp.Matrix([[1, 0], [0, 0]])
    assert right.multiply(left).coefficient(0) == sp.Matrix([[0, 0], [0, 1]])


def test_formal_block_partition_tracks_repeated_completed_parts():
    x = sp.symbols("x")
    y = sp.Function("y")
    # This equation has a repeated primary exponential that splits only after
    # secondary Newton--Puiseux refinement, giving two distinct completed Q's.
    ode = (
        sp.diff(y(x), x, 2)
        - 2 / x**3 * sp.diff(y(x), x)
        + (1 / x**6 + 3 / x**4 - 1 / x**3) * y(x)
    )
    parts = complete_formal_exponential_parts(ode, y, x)
    partition = formal_block_partition(parts)

    assert partition.total_dimension == 2
    assert partition.ramification_index == 2
    assert partition.block_dimensions == (1, 1)


def test_formal_block_partition_combines_equal_parts_and_multiplicity():
    # Duck-typed minimal completed-part objects keep block metadata independent
    # of the implementation details of the Riccati solver.
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Part:
        local_exponential_polynomial: sp.Expr
        exponential_polynomial: sp.Expr
        ramification_index: int
        multiplicity: int

    x = sp.symbols("x")
    parts = [
        Part(1 / x, 1 / x, 1, 2),
        Part(1 / x, 1 / x, 2, 1),
        Part(-1 / x, -1 / x, 1, 1),
    ]
    partition = formal_block_partition(parts)
    assert partition.total_dimension == 4
    assert partition.ramification_index == 2
    assert partition.block_dimensions == (3, 1)
    assert partition.blocks[0].part_indices == (0, 1)
    assert partition.blocks[0].ramification_index == 2


def test_gauge_transform_uses_generic_function_field_invertibility():
    x = sp.symbols("x")
    a = sp.symbols("a")
    system = FirstOrderSystem(x, sp.eye(2))
    transformed = system.gauge_transform(sp.diag(a, 1))
    assert transformed.matrix == sp.eye(2)
    # The formula is valid on the open locus a != 0; callers doing parameter
    # certification must retain that exceptional locus rather than specialize a=0.


def test_gauge_transform_accepts_assumed_nonzero_parameter():
    x = sp.symbols("x")
    a = sp.symbols("a", nonzero=True)
    system = FirstOrderSystem(x, sp.eye(2))
    transformed = system.gauge_transform(sp.diag(a, 1))
    assert transformed.matrix == sp.eye(2)
