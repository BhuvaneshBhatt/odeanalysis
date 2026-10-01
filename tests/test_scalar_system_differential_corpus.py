"""Differential tests between scalar operators and their companion systems."""

from dataclasses import dataclass

import pytest
import sympy as sp

from odeanalysis import LinearDifferentialOperator, scalar_system_correspondence


@dataclass(frozen=True)
class ScalarCase:
    name: str
    coefficients: tuple[sp.Expr, ...]


x = sp.symbols("x")
y = sp.Function("y")
CASES = (
    ScalarCase("constant-second-order", (1, 0, 1)),
    ScalarCase("euler", (-1, x, x**2)),
    ScalarCase("bessel-local", (x**2 - 1, x, x**2)),
    ScalarCase("simple-irregular", (1, 0, x**3)),
    ScalarCase("third-order-ordinary", (1, x, 0, 1)),
    ScalarCase("third-order-fuchs", (1, x, x**2, x**3)),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_scalar_companion_scalarization_is_exact(case: ScalarCase):
    op = LinearDifferentialOperator(x, y, case.coefficients)
    correspondence = scalar_system_correspondence(op)
    assert correspondence.verify()
    recovered = correspondence.scalarization.operator
    assert recovered is not None
    left = op.normalized().coefficients
    right = recovered.normalized().coefficients
    assert len(left) == len(right)
    assert all(sp.cancel(a - b) == 0 for a, b in zip(left, right, strict=True))


def test_companion_state_reconstructs_derivative_jet_exactly():
    op = LinearDifferentialOperator(x, y, (x + 1, 2 * x, 1))
    corr = scalar_system_correspondence(op)
    assert corr.verify()
    cyclic = corr.scalarization.cyclic_matrix
    # For the standard companion and first-coordinate output, the cyclic rows
    # are exactly y and y', hence the cyclic transformation is the identity.
    assert cyclic == sp.eye(2)


def test_scalarization_survives_constant_similarity_of_companion_system():
    op = LinearDifferentialOperator(x, y, (x + 1, 2 * x, 1))
    corr = scalar_system_correspondence(op)
    G = sp.ImmutableMatrix([[1, 1], [1, -1]])
    transformed = corr.system.gauge_transform(G)
    # y is the first companion coordinate, so after Y=G Z the same scalar
    # output is the first row of G.
    from odeanalysis import scalarize_system

    recovered = scalarize_system(transformed, output_row=G[0, :], function=y)
    assert recovered.verify()
    assert recovered.operator is not None
    assert all(
        sp.cancel(a - b) == 0
        for a, b in zip(
            op.normalized().coefficients,
            recovered.operator.normalized().coefficients,
            strict=True,
        )
    )
