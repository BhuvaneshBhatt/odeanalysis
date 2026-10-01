"""Conservative exact zero classification for certification boundaries."""

from __future__ import annotations

from enum import Enum

import sympy as sp


class ZeroStatus(Enum):
    ZERO = "zero"
    NONZERO = "nonzero"
    UNKNOWN = "unknown"


def exact_zero_status(expression: sp.Expr) -> ZeroStatus:
    """Classify exact zero/nonzero without assuming generic parameter values."""
    expression = sp.sympify(expression)
    if expression.is_zero is True:
        return ZeroStatus.ZERO
    if expression.is_zero is False:
        return ZeroStatus.NONZERO
    simplified = sp.cancel(sp.together(expression))
    if simplified.is_zero is True or simplified == 0:
        return ZeroStatus.ZERO
    if simplified.is_zero is False:
        return ZeroStatus.NONZERO
    return ZeroStatus.UNKNOWN
