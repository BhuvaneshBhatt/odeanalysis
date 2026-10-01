"""Shared assumptions and proof-oriented zero decisions."""

from __future__ import annotations

import sympy as sp
from funcprops import entails, normalize_assumptions

try:
    from exprtest import zerotest as _exprtest_zerotest
except ImportError:  # Optional stronger oracle; funcprops remains sufficient.
    _exprtest_zerotest = None


def zero_status(expr: sp.Expr, assumptions: sp.Expr | bool = True) -> bool | None:
    """Prove zero/nonzero when cheap, returning ``None`` when unresolved.

    Fast structural and SymPy assumption checks run first.  ``funcprops`` handles
    bounded logical entailment.  If installed, ``exprtest`` is used last and
    only with ``confidence='certified'`` so heuristic nonzero evidence can never
    steer an ODE classification branch.
    """
    assumptions = normalize_assumptions(assumptions)
    value = sp.sympify(expr)
    if value == 0 or value.is_zero is True:
        return True
    if value.is_zero is False:
        return False
    simplified = sp.cancel(sp.together(value)) if sp.count_ops(value) <= 24 else value
    if simplified == 0 or simplified.is_zero is True:
        return True
    if simplified.is_zero is False:
        return False
    if sp.count_ops(simplified) <= 24:
        try:
            decision = entails(sp.Eq(simplified, 0, evaluate=False), assumptions)
        except (RecursionError, TypeError, ValueError):
            decision = None
        if decision is not None:
            return decision
    use_exprtest = _exprtest_zerotest is not None and not simplified.has(
        sp.nan, sp.zoo, sp.oo, -sp.oo
    )
    if simplified.free_symbols and simplified.is_rational_function(
        *simplified.free_symbols
    ):
        use_exprtest = False
    if use_exprtest:
        try:
            return _exprtest_zerotest(
                simplified, assumptions=assumptions, confidence="certified"
            )
        except (RecursionError, TypeError, ValueError):
            return None
    return None


def nonzero_status(expr: sp.Expr, assumptions: sp.Expr | bool = True) -> bool | None:
    z = zero_status(expr, assumptions)
    return None if z is None else not z


def assumption_substitutions(
    assumptions: sp.Expr | bool = True,
) -> dict[sp.Symbol, sp.Expr]:
    """Extract direct symbol equalities from a conjunction of assumptions."""
    assumptions = normalize_assumptions(assumptions)
    result: dict[sp.Symbol, sp.Expr] = {}
    for clause in sp.And.make_args(assumptions):
        if not isinstance(clause, sp.Equality):
            continue
        lhs, rhs = clause.lhs, clause.rhs
        if isinstance(lhs, sp.Symbol) and lhs not in rhs.free_symbols:
            result[lhs] = rhs
        elif isinstance(rhs, sp.Symbol) and rhs not in lhs.free_symbols:
            result[rhs] = lhs
            continue
        expr = sp.expand(lhs - rhs)
        for symbol in sorted(expr.free_symbols, key=sp.default_sort_key, reverse=True):
            try:
                poly = sp.Poly(expr, symbol)
            except (sp.PolynomialError, TypeError, ValueError):
                continue
            if poly.degree() == 1:
                aa, bb = poly.all_coeffs()
                value = sp.cancel(-bb / aa)
                if symbol not in value.free_symbols:
                    result[symbol] = value
                    break
    return result
