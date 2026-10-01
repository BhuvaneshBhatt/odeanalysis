import sympy as sp

from odeanalysis import (
    formal_monodromy,
    logarithmic_frobenius_basis,
)
from odeanalysis.formal_basis import formal_logarithmic_basis


def test_repeated_frobenius_root_builds_logarithmic_basis_and_monodromy():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x)

    basis = logarithmic_frobenius_basis(ode, y, x, point=0, terms=5)
    assert basis.complete
    assert basis.dimension == 2
    assert basis.has_logarithms
    assert [sp.simplify(v.expression) for v in basis.vectors] == [1, sp.log(x)]

    monodromy = formal_monodromy(basis)
    expected = sp.Matrix([[1, 2 * sp.pi * sp.I], [0, 1]])
    assert monodromy.local_matrix == expected
    assert monodromy.cover_matrix == expected
    assert monodromy.eigenvalues == (1, 1)


def test_resonant_bessel_branch_is_regularized_by_exponent_derivative():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)

    basis = logarithmic_frobenius_basis(ode, y, x, point=0, terms=6)
    assert basis.complete
    assert basis.dimension == 2

    smaller = next(v for v in basis.vectors if v.source_exponent == -1)
    larger = next(v for v in basis.vectors if v.source_exponent == 1)
    assert smaller.pole_order == 1
    assert smaller.derivative_order == 1
    assert smaller.logarithmic_degree == 1
    assert larger.pole_order == 0
    assert larger.logarithmic_degree == 0

    monodromy = formal_monodromy(basis)
    assert monodromy.local_matrix is not None
    assert monodromy.eigenvalues == (1, 1)
    assert monodromy.local_matrix != sp.eye(2)


def test_irregular_coalesced_exponential_block_gets_logarithmic_companion():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    q = 1 / x
    w = sp.diff(q, x)

    # Gauge-conjugate Euler theta^2 v = 0 by y = exp(q) v.
    ode = sp.expand(
        x**2 * (sp.diff(y(x), x, 2) - 2 * w * sp.diff(y(x), x) + (w**2 - sp.diff(w, x)) * y(x))
        + x * (sp.diff(y(x), x) - w * y(x))
    )

    basis = formal_logarithmic_basis(ode, y, x, point=0, terms=5)
    assert basis.complete
    assert basis.dimension == 2
    assert len(basis.blocks) == 1
    assert basis.blocks[0].has_logarithms
    assert [sp.simplify(v.expression) for v in basis.vectors] == [
        sp.exp(1 / x),
        sp.exp(1 / x) * sp.log(x),
    ]

    monodromy = formal_monodromy(basis)
    assert monodromy.local_matrix == sp.Matrix([[1, 2 * sp.pi * sp.I], [0, 1]])


def test_airy_ramified_local_monodromy_swaps_exponential_blocks():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    basis = formal_logarithmic_basis(ode, y, x, point=sp.oo, terms=4)
    assert basis.complete
    assert basis.dimension == 2
    assert all(block.ramification_index == 2 for block in basis.blocks)

    monodromy = formal_monodromy(basis)
    assert monodromy.cover_matrix == -sp.eye(2)
    assert monodromy.local_matrix == sp.Matrix([[0, sp.I], [sp.I, 0]])
    assert sp.simplify(monodromy.local_matrix**2 - monodromy.cover_matrix) == sp.zeros(2)
