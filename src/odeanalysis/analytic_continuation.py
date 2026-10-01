"""Exact analytic-continuation data for supported canonical equations.

The implementation keeps Gamma-function coefficients factored and
verifies matrix relations structurally.  It does not feed large symbolic
connection matrices to ``sympy.simplify``; that is both unnecessary and a
source of severe order-dependent test-suite degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp


class CanonicalBasis(Enum):
    """Named canonical local/sectorial bases."""

    HYPERGEOMETRIC_ZERO = "hypergeometric_zero"
    HYPERGEOMETRIC_ONE = "hypergeometric_one"
    HYPERGEOMETRIC_INFINITY = "hypergeometric_infinity"
    KUMMER_INFINITY_PLUS = "kummer_infinity_plus"
    KUMMER_INFINITY_MINUS = "kummer_infinity_minus"
    AIRY_WKB = "airy_wkb"
    EULER_ZERO = "euler_zero"


@dataclass(frozen=True)
class ConnectionMatrix:
    """Connection ``C_{target<-source}`` with ``F_source = F_target*C``.

    Consequently coefficient columns transport as ``v_target = C*v_source``.
    """

    source: CanonicalBasis
    target: CanonicalBasis
    matrix: sp.ImmutableMatrix
    exact: bool = True

    def inverse(self) -> ConnectionMatrix:
        """Reverse a nonsingular rank-two connection matrix explicitly."""

        if self.matrix.shape != (2, 2):
            raise ValueError("connection matrix must be 2-by-2")
        a, b = self.matrix[0, 0], self.matrix[0, 1]
        c, d = self.matrix[1, 0], self.matrix[1, 1]
        det = a * d - b * c
        if det.is_zero is True:
            raise ValueError("connection matrix is singular")
        inv = sp.ImmutableMatrix(((d / det, -b / det), (-c / det, a / det)))
        return ConnectionMatrix(self.target, self.source, inv, self.exact)

    def then(self, other: ConnectionMatrix) -> ConnectionMatrix:
        """Compose ``C_{B<-A}`` then ``C_{C<-B}`` as ``C_{C<-A}``."""

        if self.target is not other.source:
            raise ValueError("connection bases do not compose")
        # Y_A=Y_B C_AB and Y_B=Y_C C_BC => C_AC=C_BC*C_AB.
        return ConnectionMatrix(
            self.source,
            other.target,
            other.matrix * self.matrix,
            self.exact and other.exact,
        )

    def verify(self) -> bool:
        """Check the inexpensive structural invariants of the stored connection."""

        if self.matrix.shape != (2, 2):
            return False
        if self.source is self.target:
            return self.matrix == sp.ImmutableMatrix.eye(2)
        return True


@dataclass(frozen=True)
class StokesMatrix:
    """Exact Stokes jump in a fixed canonical formal normalization."""

    family: str
    ray_index: int
    matrix: sp.ImmutableMatrix
    exact: bool = True

    def verify(self) -> bool:
        """A normalized rank-two Stokes factor is unitriangular."""

        if self.matrix.shape != (2, 2):
            return False
        return self.matrix.det() == 1 and self.matrix[0, 0] == 1 and self.matrix[1, 1] == 1


@dataclass(frozen=True)
class LocalMonodromy:
    """Actual monodromy for one positive circuit in the stated local coordinate.

    At a finite point the local coordinate is ``t=x-x0``.  At infinity it is
    ``t=1/x``; a positive counterclockwise circuit in ``t`` is therefore a
    clockwise circuit in the global ``x``-plane.
    """

    family: str
    point: sp.Expr
    basis: str
    matrix: sp.ImmutableMatrix
    formal_matrix: sp.ImmutableMatrix
    stokes_factors: tuple[StokesMatrix, ...] = ()

    def verify(self) -> bool:
        """Replay the ordered formal/Stokes product without global simplify."""

        product = self.formal_matrix
        for factor in self.stokes_factors:
            product = factor.matrix * product
        return product == self.matrix


def _gamma(x: sp.Expr) -> sp.Expr:
    return sp.gamma(x)


def hypergeometric_connection_matrix(
    a: sp.Expr,
    b: sp.Expr,
    c: sp.Expr,
    source: CanonicalBasis = CanonicalBasis.HYPERGEOMETRIC_ZERO,
    target: CanonicalBasis = CanonicalBasis.HYPERGEOMETRIC_ONE,
) -> ConnectionMatrix:
    """Return an exact nonresonant Gauss-hypergeometric connection matrix.

    Primitive formulae are retained in factored Gamma form.  Other directions
    are obtained by cheap 2-by-2 inversion/composition, never ``simplify``.
    """

    a, b, c = map(sp.sympify, (a, b, c))
    allowed = {
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_ONE,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
    }
    if source not in allowed or target not in allowed:
        raise ValueError("source and target must be Gauss-hypergeometric canonical bases")
    zero_one = sp.ImmutableMatrix(
        (
            (
                _gamma(c) * _gamma(c - a - b) / (_gamma(c - a) * _gamma(c - b)),
                _gamma(2 - c) * _gamma(c - a - b) / (_gamma(1 - a) * _gamma(1 - b)),
            ),
            (
                _gamma(c) * _gamma(a + b - c) / (_gamma(a) * _gamma(b)),
                _gamma(2 - c) * _gamma(a + b - c) / (_gamma(a - c + 1) * _gamma(b - c + 1)),
            ),
        )
    )
    # Infinity basis is chosen so the two standard continuation coefficients
    # are the columns below; this absorbs the fixed branch phases of the
    # second zero solution into the basis normalization.
    zero_inf = sp.ImmutableMatrix(
        (
            (
                _gamma(c) * _gamma(b - a) / (_gamma(b) * _gamma(c - a)),
                _gamma(2 - c) * _gamma(b - a) / (_gamma(b - c + 1) * _gamma(1 - a)),
            ),
            (
                _gamma(c) * _gamma(a - b) / (_gamma(a) * _gamma(c - b)),
                _gamma(2 - c) * _gamma(a - b) / (_gamma(a - c + 1) * _gamma(1 - b)),
            ),
        )
    )
    c01 = ConnectionMatrix(
        CanonicalBasis.HYPERGEOMETRIC_ZERO, CanonicalBasis.HYPERGEOMETRIC_ONE, zero_one
    )
    c0i = ConnectionMatrix(
        CanonicalBasis.HYPERGEOMETRIC_ZERO,
        CanonicalBasis.HYPERGEOMETRIC_INFINITY,
        zero_inf,
    )
    if source is target:
        return ConnectionMatrix(source, target, sp.ImmutableMatrix.eye(2))
    primitive = {(c01.source, c01.target): c01, (c0i.source, c0i.target): c0i}
    if (source, target) in primitive:
        return primitive[(source, target)]
    if (target, source) in primitive:
        return primitive[(target, source)].inverse()
    # Only remaining nontrivial pair is 1 <-> infinity.
    one_zero = c01.inverse()
    one_inf = one_zero.then(c0i)
    if (
        source is CanonicalBasis.HYPERGEOMETRIC_ONE
        and target is CanonicalBasis.HYPERGEOMETRIC_INFINITY
    ):
        return one_inf
    if (
        source is CanonicalBasis.HYPERGEOMETRIC_INFINITY
        and target is CanonicalBasis.HYPERGEOMETRIC_ONE
    ):
        return one_inf.inverse()
    raise ValueError("unsupported hypergeometric basis")


def connection_matrix(
    family: str,
    source: CanonicalBasis,
    target: CanonicalBasis,
    **params: sp.Expr,
) -> ConnectionMatrix:
    """Return an exact canonical connection matrix when implemented."""

    name = family.lower()
    if name in {"hypergeometric", "gauss"}:
        return hypergeometric_connection_matrix(
            params["a"], params["b"], params["c"], source, target
        )
    if name in {"kummer", "confluent_hypergeometric"}:
        plus, minus = kummer_connection_matrices(params["a"], params["c"])
        for candidate in (plus, minus):
            if candidate.source is source and candidate.target is target:
                return candidate
            if candidate.target is source and candidate.source is target:
                return candidate.inverse()
        raise ValueError("unsupported Kummer basis pair")
    raise ValueError(f"exact connection matrices are not implemented for {family!r}")


def kummer_connection_matrices(a: sp.Expr, c: sp.Expr) -> tuple[ConnectionMatrix, ConnectionMatrix]:
    """Return the two exact lateral Kummer connection matrices at infinity."""

    a, c = map(sp.sympify, (a, c))
    # A compact formal normalization: only the branch-sensitive algebraic
    # coefficient changes laterally.  Keeping the common Gamma factors
    # factored is essential for predictable symbolic cost.
    algebraic = _gamma(c) / _gamma(c - a)
    exponential = _gamma(c) / _gamma(a)
    phase = sp.exp(sp.I * sp.pi * a)
    plus = sp.ImmutableMatrix(((algebraic * phase, 0), (exponential, 1)))
    minus = sp.ImmutableMatrix(((algebraic / phase, 0), (exponential, 1)))
    return (
        ConnectionMatrix(
            CanonicalBasis.HYPERGEOMETRIC_ZERO,
            CanonicalBasis.KUMMER_INFINITY_PLUS,
            plus,
        ),
        ConnectionMatrix(
            CanonicalBasis.HYPERGEOMETRIC_ZERO,
            CanonicalBasis.KUMMER_INFINITY_MINUS,
            minus,
        ),
    )


def stokes_matrices(family: str, **params: sp.Expr) -> tuple[StokesMatrix, ...]:
    """Return exact normalized Stokes factors for supported canonical families."""

    name = family.lower()
    if name == "airy":
        i = sp.I
        return (
            StokesMatrix("airy", 0, sp.ImmutableMatrix(((1, i), (0, 1)))),
            StokesMatrix("airy", 1, sp.ImmutableMatrix(((1, 0), (i, 1)))),
            StokesMatrix("airy", 2, sp.ImmutableMatrix(((1, i), (0, 1)))),
        )
    if name in {"hypergeometric", "gauss"}:
        return ()
    if name in {"kummer", "confluent_hypergeometric"}:
        a = sp.sympify(params["a"])
        c = sp.sympify(params["c"])
        # Standard formal normalization.  These multipliers stay factored;
        # no Gamma reflection simplification is performed here.
        upper = 2 * sp.pi * sp.I * sp.exp(sp.I * sp.pi * (c - a)) / (_gamma(a) * _gamma(1 + a - c))
        lower = 2 * sp.pi * sp.I * sp.exp(sp.I * sp.pi * a) / (_gamma(c - a) * _gamma(1 - a))
        return (
            StokesMatrix("kummer", 0, sp.ImmutableMatrix(((1, upper), (0, 1)))),
            StokesMatrix("kummer", 1, sp.ImmutableMatrix(((1, 0), (lower, 1)))),
        )
    raise ValueError(f"exact Stokes matrices are not implemented for {family!r}")


def local_monodromy(family: str, point: sp.Expr, **params: sp.Expr) -> LocalMonodromy:
    """Return actual monodromy for a positive circuit in the local coordinate.

    Infinity uses ``t=1/x``.  Thus an infinity basis with powers ``x**(-a)``
    and ``x**(-b)`` has local multipliers ``exp(2*pi*I*a)`` and
    ``exp(2*pi*I*b)``.
    """

    name = family.lower()
    point = sp.sympify(point)
    if name in {"euler", "euler_cauchy"}:
        if point != 0:
            raise ValueError("Euler--Cauchy monodromy is implemented at the origin")
        alpha, beta = map(sp.sympify, (params["alpha"], params["beta"]))
        discriminant = sp.sqrt((alpha - 1) ** 2 - 4 * beta)
        rho_plus = sp.simplify((1 - alpha + discriminant) / 2)
        rho_minus = sp.simplify((1 - alpha - discriminant) / 2)
        matrix = sp.ImmutableMatrix.diag(
            sp.exp(2 * sp.pi * sp.I * rho_plus),
            sp.exp(2 * sp.pi * sp.I * rho_minus),
        )
        return LocalMonodromy("euler", point, "power", matrix, matrix)
    if name in {"hypergeometric", "gauss"}:
        a, b, c = map(sp.sympify, (params["a"], params["b"], params["c"]))
        if point == 0:
            matrix = sp.ImmutableMatrix.diag(1, sp.exp(2 * sp.pi * sp.I * (1 - c)))
        elif point == 1:
            matrix = sp.ImmutableMatrix.diag(1, sp.exp(2 * sp.pi * sp.I * (c - a - b)))
        elif point == sp.oo:
            matrix = sp.ImmutableMatrix.diag(
                sp.exp(2 * sp.pi * sp.I * a), sp.exp(2 * sp.pi * sp.I * b)
            )
        else:
            raise ValueError("Gauss monodromy point must be 0, 1, or infinity")
        return LocalMonodromy("hypergeometric", point, "frobenius", matrix, matrix)
    if name == "airy" and point == sp.oo:
        factors = stokes_matrices("airy")
        formal = sp.ImmutableMatrix(((0, -sp.I), (-sp.I, 0)))
        product = formal
        for factor in factors:
            product = factor.matrix * product
        return LocalMonodromy("airy", point, "exact_wkb", product, formal, factors)
    raise ValueError(f"actual local monodromy is not implemented for {family!r} at {point}")
