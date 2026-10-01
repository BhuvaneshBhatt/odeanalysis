"""Exception groups for recoverable symbolic-backend failures."""

from sympy.core.function import PoleError
from sympy.matrices.exceptions import MatrixError
from sympy.polys.polyerrors import CoercionFailed, GeneratorsNeeded, PolynomialError

SYMBOLIC_FAILURES = (
    TypeError,
    ValueError,
    NotImplementedError,
    ArithmeticError,
    PoleError,
    MatrixError,
    PolynomialError,
    CoercionFailed,
    GeneratorsNeeded,
)

NUMERIC_CONVERSION_FAILURES = (TypeError, ValueError, OverflowError)
