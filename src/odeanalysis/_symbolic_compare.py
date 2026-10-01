"""Small exact-comparison helpers shared by structural algorithms."""

from __future__ import annotations

import sympy as sp


def expressions_equal(left: sp.Expr, right: sp.Expr) -> bool:
    """Return whether two expressions simplify to the same exact value."""

    if left == right:
        return True
    difference = sp.simplify(left - right)
    if difference == 0:
        return True
    return sp.simplify(sp.expand(difference)) == 0
