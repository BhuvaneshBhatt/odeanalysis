"""Algebraic/enclosure corpus for the Arb certified continuation backend."""

import pytest
import sympy as sp

from odeanalysis import FirstOrderSystem, certified_system_continuation

flint = pytest.importorskip("flint")
from flint import acb  # noqa: E402

x = sp.symbols("x")


def _balls(result):
    assert result.complete and result.enclosure is not None
    return [[acb(entry) for entry in row] for row in result.enclosure.rows]


def _contains_zero(ball):
    return ball.contains(0)


@pytest.mark.parametrize(
    "matrix,start,end,expected",
    [
        (sp.ImmutableMatrix([[0]]), 0, 7, sp.eye(1)),
        (sp.ImmutableMatrix.diag(1, -1), 0, 1, sp.diag(sp.E, sp.exp(-1))),
        (sp.ImmutableMatrix([[0, 1], [0, 0]]), 0, 2, sp.Matrix([[1, 2], [0, 1]])),
        (
            sp.ImmutableMatrix([[0, -1], [1, 0]]),
            0,
            sp.pi / 2,
            sp.Matrix([[0, -1], [1, 0]]),
        ),
    ],
)
def test_certified_reference_transports_contain_exact_values(matrix, start, end, expected):
    result = certified_system_continuation(
        FirstOrderSystem(x, matrix), start, end, precision_bits=192
    )
    balls = _balls(result)
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            # Arb's string parser is used only to recover the returned ball;
            # containment is then checked against a high-precision point ball.
            target = acb(str(sp.N(expected[i, j], 60)))
            assert not (balls[i][j] - target).is_finite() or (balls[i][j] - target).contains(0)


def test_reverse_transport_product_contains_identity():
    A = sp.ImmutableMatrix([[1, 2], [0, -1]])
    forward = _balls(
        certified_system_continuation(
            FirstOrderSystem(x, A), 0, sp.Rational(3, 2), precision_bits=192
        )
    )
    reverse = _balls(
        certified_system_continuation(
            FirstOrderSystem(x, A), sp.Rational(3, 2), 0, precision_bits=192
        )
    )
    for i in range(2):
        for j in range(2):
            value = sum(reverse[i][k] * forward[k][j] for k in range(2))
            assert (value - (1 if i == j else 0)).contains(0)


def test_concatenation_product_contains_direct_transport():
    A = sp.ImmutableMatrix([[0, 1], [-2, -3]])
    ab = _balls(certified_system_continuation(FirstOrderSystem(x, A), 0, 1, precision_bits=192))
    bc = _balls(certified_system_continuation(FirstOrderSystem(x, A), 1, 2, precision_bits=192))
    ac = _balls(certified_system_continuation(FirstOrderSystem(x, A), 0, 2, precision_bits=192))
    for i in range(2):
        for j in range(2):
            composed = sum(bc[i][k] * ab[k][j] for k in range(2))
            assert (composed - ac[i][j]).contains(0)


def test_precision_contract_and_refusal_are_explicit():
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[1]]))
    with pytest.raises(ValueError):
        certified_system_continuation(system, 0, 1, precision_bits=32)
    variable = certified_system_continuation(FirstOrderSystem(x, sp.ImmutableMatrix([[x]])), 0, 1)
    assert not variable.complete
    assert "variable-coefficient" in variable.limitation


def test_rounded_float_is_not_misrepresented_as_exact_arb_input():
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[sp.Float("0.1")]]))
    result = certified_system_continuation(system, 0, 1, precision_bits=192)
    assert not result.complete
    assert result.enclosure is None
    assert "rounded decimal surrogate" in result.limitation


def test_exact_pi_endpoint_uses_rigorous_constant_conversion():
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[0, -1], [1, 0]]))
    result = certified_system_continuation(system, 0, sp.pi / 2, precision_bits=192)
    balls = _balls(result)
    assert balls[0][0].contains(0)
    assert (balls[0][1] + 1).contains(0)
    assert (balls[1][0] - 1).contains(0)
    assert balls[1][1].contains(0)
