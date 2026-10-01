"""Bell-polynomial utilities used by the formal ODE layer.

The complete exponential Bell polynomial is constructed directly from
integer-partition multiplicities.  SymPy exposes the incomplete/partial
polynomial ``bell(n, k, symbols)``; the direct partition formula avoids
recomputing the overlapping partial polynomials when the complete polynomial
is wanted in expanded monomial form.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import factorial

import sympy as sp
from sympy.utilities.iterables import partitions


def complete_exponential_bell_polynomial(
    n: int,
    variables: Iterable[sp.Expr],
) -> sp.Expr:
    r"""Return the complete exponential Bell polynomial ``B_n``.

    The convention is

    .. math::

       B_n(x_1,\ldots,x_n)
       = \sum_{\sum i c_i=n}
         \frac{n!}{\prod_i c_i! (i!)^{c_i}}
         \prod_i x_i^{c_i}.

    The summation is performed directly over integer-partition multiplicity
    dictionaries, so each generic Bell monomial is constructed exactly once.
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.S.One

    xs = tuple(sp.sympify(x) for x in variables)
    if len(xs) < n:
        raise ValueError(f"B_{n} requires at least {n} variables")

    factorials = tuple(factorial(i) for i in range(n + 1))
    n_factorial = factorials[n]
    terms: list[sp.Expr] = []

    for multiplicities in partitions(n):
        coefficient = n_factorial
        factors: list[sp.Expr] = []
        for part, multiplicity in multiplicities.items():
            coefficient //= factorials[multiplicity]
            coefficient //= factorials[part] ** multiplicity
            factors.append(sp.Pow(xs[part - 1], multiplicity, evaluate=True))
        terms.append(sp.Integer(coefficient) * sp.Mul(*factors))

    return sp.Add(*terms)


def complete_exponential_bell_via_sympy(
    n: int,
    variables: Iterable[sp.Expr],
) -> sp.Expr:
    """Reference construction using SymPy's partial Bell polynomials.

    This helper is primarily useful for validation and compatibility tests.
    The partition-based :func:`complete_exponential_bell_polynomial` is the
    direct complete-polynomial implementation used by ``odeanalysis``.
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.S.One

    xs = tuple(sp.sympify(x) for x in variables)
    if len(xs) < n:
        raise ValueError(f"B_{n} requires at least {n} variables")

    return sp.Add(*(sp.bell(n, k, xs[: n - k + 1]) for k in range(1, n + 1)))
