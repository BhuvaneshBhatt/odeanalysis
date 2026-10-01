"""Branch-aware power simplification helpers.

``powsimp(..., force=True)`` is PowerExpand-like for unconstrained symbolic
bases.  ODE coefficients are analytic data even when the local coordinate is a
formal uniformizer, so whole ODE expressions must use branch-aware power
simplification.  A separate formal helper is retained only for data structures
that explicitly define their factors as formal monomials.
"""

from __future__ import annotations

import sympy as sp


def analytic_powsimp(expr: sp.Expr) -> sp.Expr:
    """Simplify powers without assuming branch identities not proved by SymPy."""

    return sp.powsimp(sp.sympify(expr), force=False)


def formal_powsimp(expr: sp.Expr) -> sp.Expr:
    """Canonicalize an expression known in its entirety to be a formal monomial."""

    return sp.powsimp(sp.sympify(expr), force=True)


def mixed_powsimp(coefficient: sp.Expr, monomial: sp.Expr) -> sp.Expr:
    """Canonicalize a formal monomial without forcing identities in its coefficient."""

    coefficient = analytic_powsimp(coefficient)
    monomial = formal_powsimp(monomial)
    return analytic_powsimp(coefficient * monomial)
