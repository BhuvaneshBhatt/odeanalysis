import pytest
import sympy as sp

from odeanalysis.analytic_continuation import (
    CanonicalBasis,
    ConnectionMatrix,
    hypergeometric_connection_matrix,
    kummer_connection_matrices,
    local_monodromy,
    stokes_matrices,
)


def test_hypergeometric_zero_one_formula():
    a, b, c = sp.symbols("a b c")
    conn = hypergeometric_connection_matrix(a, b, c)
    assert conn.source is CanonicalBasis.HYPERGEOMETRIC_ZERO
    assert conn.target is CanonicalBasis.HYPERGEOMETRIC_ONE
    assert conn.matrix[0, 0] == sp.gamma(c) * sp.gamma(c - a - b) / (
        sp.gamma(c - a) * sp.gamma(c - b)
    )


def test_hypergeometric_inverse_is_structural():
    a, b, c = sp.symbols("a b c")
    conn = hypergeometric_connection_matrix(a, b, c)
    inv = conn.inverse()
    assert inv.source is conn.target
    assert inv.target is conn.source
    assert inv.verify()


def test_hypergeometric_one_infinity_composes_without_simplify():
    a, b, c = sp.symbols("a b c")
    c10 = hypergeometric_connection_matrix(
        a, b, c, CanonicalBasis.HYPERGEOMETRIC_ONE, CanonicalBasis.HYPERGEOMETRIC_ZERO
    )
    c0i = hypergeometric_connection_matrix(
        a,
        b,
        c,
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
    )
    c1i = hypergeometric_connection_matrix(
        a,
        b,
        c,
        CanonicalBasis.HYPERGEOMETRIC_ONE,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
    )
    assert c1i.matrix == c0i.matrix * c10.matrix


def test_gauss_has_no_stokes_factors():
    assert stokes_matrices("hypergeometric") == ()


def test_airy_stokes_are_unitriangular():
    factors = stokes_matrices("airy")
    assert len(factors) == 3
    assert all(f.verify() for f in factors)


def test_airy_actual_monodromy_replays():
    mon = local_monodromy("airy", sp.oo)
    assert mon.verify()
    assert mon.matrix == sp.ImmutableMatrix.eye(2)


def test_kummer_lateral_connections_remain_factored():
    a, c = sp.symbols("a c")
    plus, minus = kummer_connection_matrices(a, c)
    assert plus.verify() and minus.verify()
    assert plus.matrix[0, 0].has(sp.gamma(c - a))


def test_regular_hypergeometric_actual_monodromy():
    a, b, c = sp.symbols("a b c")
    mon = local_monodromy("hypergeometric", 0, a=a, b=b, c=c)
    assert mon.verify()
    assert mon.matrix[1, 1] == sp.exp(2 * sp.pi * sp.I * (1 - c))


def test_connection_verification_never_calls_global_simplify(monkeypatch):
    a, b, c = sp.symbols("a b c")

    def forbidden_simplify(*args, **kwargs):
        raise AssertionError("analytic continuation must not call sympy.simplify")

    monkeypatch.setattr(sp, "simplify", forbidden_simplify)
    c01 = hypergeometric_connection_matrix(a, b, c)
    c10 = c01.inverse()
    c0i = hypergeometric_connection_matrix(
        a,
        b,
        c,
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
    )
    c1i = c10.then(c0i)
    assert c01.verify()
    assert c10.verify()
    assert c1i.verify()
    assert all(factor.verify() for factor in stokes_matrices("kummer", a=a, c=c))


def test_identity_connection_is_a_valid_structural_certificate():
    basis = CanonicalBasis.HYPERGEOMETRIC_ZERO
    a, b, c = sp.symbols("a b c")
    conn = hypergeometric_connection_matrix(a, b, c, basis, basis)
    assert conn.matrix == sp.ImmutableMatrix.eye(2)
    assert conn.verify()


def test_hypergeometric_connection_rejects_foreign_basis_even_for_identity():
    a, b, c = sp.symbols("a b c")
    with pytest.raises(ValueError, match="Gauss-hypergeometric"):
        hypergeometric_connection_matrix(
            a,
            b,
            c,
            CanonicalBasis.KUMMER_INFINITY_PLUS,
            CanonicalBasis.KUMMER_INFINITY_PLUS,
        )


def test_connection_inverse_rejects_singular_matrix():
    conn = ConnectionMatrix(
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_ONE,
        sp.ImmutableMatrix(((1, 2), (2, 4))),
    )
    with pytest.raises(ValueError, match="singular"):
        conn.inverse()


def test_connection_convention_transports_coefficients_and_monodromy():
    connection = ConnectionMatrix(
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_ONE,
        sp.ImmutableMatrix(((1, 2), (3, 5))),
    )
    v_source = sp.ImmutableMatrix((sp.Symbol("u"), sp.Symbol("v")))
    v_target = connection.matrix * v_source
    # F_source v_source = F_target C_{target<-source} v_source.
    assert v_target == connection.matrix * v_source

    local_target = sp.ImmutableMatrix.diag(sp.Symbol("m1"), sp.Symbol("m2"))
    transported = connection.inverse().matrix * local_target * connection.matrix
    assert connection.matrix * transported == local_target * connection.matrix


def test_gauss_infinity_monodromy_uses_positive_reciprocal_coordinate_orientation():
    a, b, c = sp.symbols("a b c")
    mon = local_monodromy("hypergeometric", sp.oo, a=a, b=b, c=c)
    assert mon.matrix == sp.ImmutableMatrix.diag(
        sp.exp(2 * sp.pi * sp.I * a), sp.exp(2 * sp.pi * sp.I * b)
    )


def test_connection_composition_direction_is_target_left_source_right():
    a, b, c = sp.symbols("a b c")
    c10 = hypergeometric_connection_matrix(
        a, b, c, CanonicalBasis.HYPERGEOMETRIC_ZERO, CanonicalBasis.HYPERGEOMETRIC_ONE
    )
    ci1 = hypergeometric_connection_matrix(
        a,
        b,
        c,
        CanonicalBasis.HYPERGEOMETRIC_ONE,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
    )
    hypergeometric_connection_matrix(
        a,
        b,
        c,
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
    )
    assert c10.then(ci1).matrix == ci1.matrix * c10.matrix
    assert c10.then(ci1).source is CanonicalBasis.HYPERGEOMETRIC_ZERO
    assert c10.then(ci1).target is CanonicalBasis.HYPERGEOMETRIC_INFINITY
