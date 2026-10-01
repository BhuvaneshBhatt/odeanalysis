"""First-order linear systems and formal block metadata.

The scalar algorithms in :mod:`odeanalysis` discover exponential parts very
well.  Formal decomposition of repeated irregular blocks is naturally a
problem about differential modules, however, so this module provides the
system-level foundation used by subsequent block-reduction code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING

import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError

from ._zero import ZeroStatus, exact_zero_status
from .operator import LinearDifferentialOperator

if TYPE_CHECKING:
    from .formal import CompleteFormalExponentialPart


def _immutable_column(
    vector: sp.MatrixBase | Sequence[sp.Expr], size: int
) -> sp.ImmutableMatrix:
    matrix = sp.Matrix(vector)
    if matrix.shape == (size,):
        matrix = matrix.reshape(size, 1)
    if matrix.shape != (size, 1):
        raise ValueError(f"forcing vector must have shape ({size}, 1)")
    return sp.ImmutableMatrix(matrix)


@dataclass(frozen=True)
class FirstOrderSystem:
    r"""A linear first-order system ``Y' = A(x) Y + b(x)``.

    The formal differential-module machinery normally uses homogeneous
    systems.  ``forcing`` is nevertheless retained so scalar companion
    conversion and exact gauge transformations preserve inhomogeneous input
    while preserving it.

    ``ramification_index`` records the accumulated cover index relative to the
    original local coordinate.  It is metadata; :meth:`ramify` performs the
    actual variable transformation.
    """

    variable: sp.Symbol
    matrix: sp.ImmutableMatrix
    forcing: sp.ImmutableMatrix | None = None
    ramification_index: int = 1

    def __post_init__(self) -> None:
        matrix = sp.ImmutableMatrix(self.matrix)
        if matrix.rows != matrix.cols:
            raise ValueError("a first-order system matrix must be square")
        if self.ramification_index < 1:
            raise ValueError("ramification_index must be positive")
        if self.forcing is None:
            forcing = sp.ImmutableMatrix(sp.zeros(matrix.rows, 1))
        else:
            forcing = _immutable_column(self.forcing, matrix.rows)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "forcing", forcing)

    @property
    def dimension(self) -> int:
        return self.matrix.rows

    def _forcing(self) -> sp.ImmutableMatrix:
        """Return the normalized forcing vector or report an internal invariant failure."""

        if self.forcing is None:
            raise RuntimeError(
                "FirstOrderSystem forcing was not normalized during construction"
            )
        return self.forcing

    @property
    def is_homogeneous(self) -> bool:
        return bool(self._forcing().is_zero_matrix)

    def equation(self, dependent: sp.FunctionClass | None = None) -> sp.Equality:
        """Return a matrix equation representing the system."""

        if dependent is None:
            dependent = sp.Function("Y")
        y = dependent(self.variable)
        return sp.Eq(sp.diff(y, self.variable), self.matrix * y + self._forcing())

    def change_variable(
        self,
        new_variable: sp.Symbol,
        old_variable_expression: sp.Expr,
        *,
        ramification_multiplier: int = 1,
    ) -> FirstOrderSystem:
        r"""Apply an exact independent-variable change ``x = phi(t)``.

        If ``dY/dx = A(x)Y+b(x)`` and ``x=phi(t)``, then

        ``dY/dt = phi'(t) A(phi(t)) Y + phi'(t) b(phi(t))``.
        """

        if ramification_multiplier < 1:
            raise ValueError("ramification_multiplier must be positive")
        phi = sp.sympify(old_variable_expression)
        jacobian = sp.diff(phi, new_variable)
        if jacobian == 0:
            raise ValueError("variable transformation must have nonzero derivative")
        substitutions = {self.variable: phi}
        matrix = self.matrix.applyfunc(
            lambda entry: sp.cancel(sp.together(jacobian * entry.subs(substitutions)))
        )
        forcing = self._forcing().applyfunc(
            lambda entry: sp.cancel(sp.together(jacobian * entry.subs(substitutions)))
        )
        return FirstOrderSystem(
            new_variable,
            sp.ImmutableMatrix(matrix),
            sp.ImmutableMatrix(forcing),
            self.ramification_index * ramification_multiplier,
        )

    def ramify(self, new_variable: sp.Symbol, index: int) -> FirstOrderSystem:
        r"""Move to the cover ``x = t**index``.

        The transformed connection matrix is
        ``index*t**(index-1)*A(t**index)``.
        """

        if index < 1:
            raise ValueError("ramification index must be positive")
        return self.change_variable(
            new_variable,
            new_variable**index,
            ramification_multiplier=index,
        )

    def gauge_transform(self, gauge: sp.MatrixBase) -> FirstOrderSystem:
        r"""Apply the exact gauge transformation ``Y = G Z``.

        The transformed system is

        ``Z' = (G**-1 A G - G**-1 G') Z + G**-1 b``.
        """

        gauge = sp.Matrix(gauge)
        if gauge.shape != self.matrix.shape:
            raise ValueError(
                "gauge matrix must have the same square shape as the system"
            )
        determinant = sp.cancel(sp.together(gauge.det()))
        determinant_status = exact_zero_status(determinant)
        # Gauge transformations live in a symbolic/meromorphic function field:
        # they require det(G) not to be identically zero, not globally nonzero at
        # every parameter/coordinate specialization.  A determinant such as
        # ``a`` or ``(1+x)*exp(x)`` therefore defines a valid generic/local gauge
        # with an explicit exceptional locus det(G)=0.
        if determinant_status is ZeroStatus.ZERO:
            raise ValueError("gauge matrix must be invertible")
        try:
            gauge_inverse = gauge.inv()
        except (ValueError, ZeroDivisionError, NonInvertibleMatrixError) as exc:
            raise ValueError("gauge matrix must be invertible") from exc
        gauge_derivative = gauge.diff(self.variable)
        transformed = (
            gauge_inverse * sp.Matrix(self.matrix) * gauge
            - gauge_inverse * gauge_derivative
        )
        forcing = gauge_inverse * sp.Matrix(self._forcing())
        return FirstOrderSystem(
            self.variable,
            sp.ImmutableMatrix(
                transformed.applyfunc(lambda entry: sp.cancel(sp.together(entry)))
            ),
            sp.ImmutableMatrix(
                forcing.applyfunc(lambda entry: sp.cancel(sp.together(entry)))
            ),
            self.ramification_index,
        )

    def exponential_gauge(self, exponent: sp.Expr) -> FirstOrderSystem:
        r"""Remove a scalar exponential factor ``Y = exp(Q) Z``.

        This is the system-level analogue of scalar operator conjugation and
        simply subtracts ``Q' I`` from the connection matrix.  The forcing is
        multiplied by ``exp(-Q)`` when present.
        """

        exponent = sp.sympify(exponent)
        derivative = sp.diff(exponent, self.variable)
        matrix = sp.Matrix(self.matrix) - derivative * sp.eye(self.dimension)
        forcing = sp.Matrix(self._forcing()) * sp.exp(-exponent)
        return FirstOrderSystem(
            self.variable,
            sp.ImmutableMatrix(
                matrix.applyfunc(lambda entry: sp.cancel(sp.together(entry)))
            ),
            sp.ImmutableMatrix(forcing),
            self.ramification_index,
        )

    @classmethod
    def from_scalar_operator(
        cls,
        operator: LinearDifferentialOperator,
        *,
        normalize: bool = True,
    ) -> FirstOrderSystem:
        """Return the standard companion system of a scalar linear operator."""

        op = operator.normalized() if normalize else operator
        n = op.order
        lead = op.leading_coefficient
        if not normalize and sp.simplify(lead - 1) != 0:
            coefficients = tuple(
                sp.cancel(sp.together(c / lead)) for c in op.coefficients
            )
            inhomogeneous = sp.cancel(sp.together(op.inhomogeneous / lead))
        else:
            coefficients = op.coefficients
            inhomogeneous = op.inhomogeneous

        matrix = sp.zeros(n, n)
        for row in range(n - 1):
            matrix[row, row + 1] = 1
        for column in range(n):
            matrix[n - 1, column] = -coefficients[column]

        forcing = sp.zeros(n, 1)
        # L[y] + r = 0 -> y^(n) = ... - r for a monic operator.
        forcing[n - 1, 0] = -inhomogeneous
        return cls(
            op.variable,
            sp.ImmutableMatrix(matrix),
            sp.ImmutableMatrix(forcing),
        )


def companion_system(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> FirstOrderSystem:
    """Construct the standard first-order companion system of a scalar ODE."""

    if isinstance(ode, LinearDifferentialOperator):
        operator = ode
    else:
        if function is None or variable is None:
            raise TypeError(
                "function and variable are required when ode is not an operator"
            )
        operator = LinearDifferentialOperator.from_ode(ode, function, variable)
    return FirstOrderSystem.from_scalar_operator(operator)


@dataclass(frozen=True)
class FormalExponentialBlockMetadata:
    """Metadata for formal branches sharing one completed exponential part."""

    local_exponential_polynomial: sp.Expr
    exponential_polynomial: sp.Expr
    multiplicity: int
    ramification_index: int
    part_indices: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return self.multiplicity


@dataclass(frozen=True)
class FormalBlockPartition:
    """Partition of completed scalar branches into exponential blocks."""

    blocks: tuple[FormalExponentialBlockMetadata, ...]
    ramification_index: int
    total_dimension: int

    @property
    def block_dimensions(self) -> tuple[int, ...]:
        return tuple(block.dimension for block in self.blocks)


def formal_block_partition(
    parts: Sequence[CompleteFormalExponentialPart],
) -> FormalBlockPartition:
    """Group completed scalar exponential parts into system blocks.

    Equality is certified by simplifying the difference of the *local*
    completed exponential polynomials.  Multiplicity is preserved, including
    unresolved multiplicity surviving the Riccati/Newton--Puiseux refinement.
    """

    groups: list[dict[str, object]] = []
    common_ramification = 1
    total_dimension = 0
    for index, part in enumerate(parts):
        common_ramification = lcm(common_ramification, int(part.ramification_index))
        multiplicity = int(part.multiplicity)
        total_dimension += multiplicity
        local_q = sp.simplify(part.local_exponential_polynomial)
        matched: dict[str, object] | None = None
        for group in groups:
            if sp.simplify(local_q - sp.sympify(group["local_q"])) == 0:
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "local_q": local_q,
                    "q": sp.simplify(part.exponential_polynomial),
                    "multiplicity": multiplicity,
                    "ramification": int(part.ramification_index),
                    "indices": [index],
                }
            )
        else:
            matched["multiplicity"] = int(matched["multiplicity"]) + multiplicity
            matched["ramification"] = lcm(
                int(matched["ramification"]), int(part.ramification_index)
            )
            indices = matched["indices"]
            if not isinstance(indices, list):
                raise RuntimeError("formal block index storage is not mutable")
            indices.append(index)

    blocks = tuple(
        FormalExponentialBlockMetadata(
            local_exponential_polynomial=sp.sympify(group["local_q"]),
            exponential_polynomial=sp.sympify(group["q"]),
            multiplicity=int(group["multiplicity"]),
            ramification_index=int(group["ramification"]),
            part_indices=tuple(group["indices"]),
        )
        for group in groups
    )
    return FormalBlockPartition(
        blocks=blocks,
        ramification_index=common_ramification,
        total_dimension=total_dimension,
    )
