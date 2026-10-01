"""Validated numerical continuation with complex ball arithmetic.

The initial certified backend handles homogeneous constant-coefficient systems
exactly along a straight segment.  It uses python-flint/Arb matrix exponential,
so every returned entry is a complex ball enclosure rather than a tolerance-only
floating-point estimate.  Variable-coefficient validated integration remains an
explicitly unsupported extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import sympy as sp

from .system import FirstOrderSystem

_ARB_CONTEXT_LOCK = RLock()


@dataclass(frozen=True)
class CertifiedMatrixEnclosure:
    """Serializable complex-ball enclosure for a transported fundamental matrix."""

    rows: tuple[tuple[str, ...], ...]
    precision_bits: int
    method: str
    certified: bool = True


@dataclass(frozen=True)
class CertifiedContinuationResult:
    start: sp.Expr
    end: sp.Expr
    enclosure: CertifiedMatrixEnclosure | None
    complete: bool
    limitation: str | None = None


def _exact_acb(expr: sp.Expr):
    """Convert a supported exact SymPy expression to an Arb complex ball.

    Certification must never pass a rounded decimal approximation to Arb as
    though it were the exact input.  The converter therefore builds balls only
    from exact rational arithmetic and rigorously evaluated named constants.
    Unsupported exact constants/functions are refused conservatively.
    """

    from flint import acb, arb

    expr = sp.sympify(expr)
    if expr.free_symbols:
        raise ValueError(
            "certified continuation requires numeric endpoints and coefficients"
        )
    if expr.is_Integer:
        return acb(int(expr))
    if expr.is_Rational:
        return acb(int(expr.p)) / int(expr.q)
    if expr == sp.I:
        return acb(0, 1)
    if expr == sp.pi:
        return acb(arb.pi())
    if expr.is_Add:
        total = acb(0)
        for term in expr.args:
            total += _exact_acb(term)
        return total
    if expr.is_Mul:
        total = acb(1)
        for factor in expr.args:
            total *= _exact_acb(factor)
        return total
    if expr.is_Pow and expr.exp.is_Integer:
        return _exact_acb(expr.base) ** int(expr.exp)
    raise ValueError(
        f"exact Arb conversion is not implemented for {expr!s}; "
        "refusing to certify a rounded decimal surrogate"
    )


def certified_system_continuation(
    system: FirstOrderSystem,
    start: sp.Expr,
    end: sp.Expr,
    *,
    precision_bits: int = 160,
) -> CertifiedContinuationResult:
    """Enclose the fundamental transport for a constant homogeneous system.

    For ``Y'=A Y`` with constant ``A``, transport from ``a`` to ``b`` is
    ``exp((b-a) A)``.  Arb's complex-ball matrix exponential encloses this
    quantity rigorously.  Variable coefficients are rejected rather than
    treated with an uncertified floating-point ODE solver.
    """
    start = sp.sympify(start)
    end = sp.sympify(end)
    if precision_bits < 64:
        raise ValueError("precision_bits must be at least 64")
    if not system.is_homogeneous:
        return CertifiedContinuationResult(
            start,
            end,
            None,
            False,
            "inhomogeneous systems are not supported by the certified fundamental-matrix backend",
        )
    if any(sp.sympify(entry).has(system.variable) for entry in system.matrix):
        return CertifiedContinuationResult(
            start,
            end,
            None,
            False,
            "validated variable-coefficient integration is not implemented",
        )
    try:
        from flint import acb_mat, ctx
    except ImportError:
        return CertifiedContinuationResult(
            start,
            end,
            None,
            False,
            "python-flint is required for certified ball arithmetic",
        )
    # python-flint's context precision is process-global.  Serialize context
    # mutation so concurrent certified calls cannot silently change one
    # another's working precision.
    with _ARB_CONTEXT_LOCK:
        old_prec = ctx.prec
        try:
            ctx.prec = precision_bits
            delta = _exact_acb(end - start)
            matrix = acb_mat(
                [
                    [
                        _exact_acb(system.matrix[i, j]) * delta
                        for j in range(system.dimension)
                    ]
                    for i in range(system.dimension)
                ]
            )
            transport = matrix.exp()
            rows = tuple(
                tuple(str(transport[i, j]) for j in range(system.dimension))
                for i in range(system.dimension)
            )
        except (ValueError, TypeError, ArithmeticError) as exc:
            return CertifiedContinuationResult(
                start, end, None, False, f"ball continuation failed: {exc}"
            )
        finally:
            ctx.prec = old_prec
    return CertifiedContinuationResult(
        start,
        end,
        CertifiedMatrixEnclosure(rows, precision_bits, "arb-matrix-exponential"),
        True,
    )
