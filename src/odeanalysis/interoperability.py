"""Certified correspondence between scalar equations and first-order systems."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .operator import LinearDifferentialOperator
from .system import FirstOrderSystem, companion_system


@dataclass(frozen=True)
class CyclicScalarization:
    """A scalar equation obtained from a certified cyclic output row."""

    system: FirstOrderSystem
    output_row: sp.ImmutableMatrix
    cyclic_matrix: sp.ImmutableMatrix
    operator: LinearDifferentialOperator | None
    complete: bool
    limitation: str | None = None

    def verify(self) -> bool:
        if not self.complete or self.operator is None:
            return False
        x = self.system.variable
        rows = [sp.Matrix(self.output_row)]
        for _ in range(self.system.dimension):
            row = rows[-1]
            rows.append(row.diff(x) + row * sp.Matrix(self.system.matrix))
        cyclic = sp.Matrix.vstack(*rows[:-1])
        if sp.simplify(cyclic.det()) == 0:
            return False
        alpha = sp.simplify(rows[-1] * cyclic.inv())
        expected = (
            *tuple(sp.cancel(-alpha[0, j]) for j in range(self.system.dimension)),
            sp.S.One,
        )
        return all(
            sp.simplify(a - b) == 0
            for a, b in zip(expected, self.operator.coefficients, strict=True)
        )


def scalarize_system(
    system: FirstOrderSystem,
    *,
    output_row: sp.MatrixBase | None = None,
    function: sp.FunctionClass | None = None,
) -> CyclicScalarization:
    """Scalarize a homogeneous system when the chosen output is cyclic.

    The row jets satisfy ``r_(k+1)=r_k' + r_k A``.  If the first ``n`` rows
    form an invertible cyclic matrix, the next row gives the monic scalar
    equation for the output.  Failure to certify cyclicity is explicit.
    """
    if not system.is_homogeneous:
        return CyclicScalarization(
            system,
            sp.ImmutableMatrix([]),
            sp.ImmutableMatrix([]),
            None,
            False,
            "inhomogeneous systems are not scalarized",
        )
    n = system.dimension
    if output_row is None:
        row = sp.zeros(1, n)
        row[0, 0] = 1
    else:
        row = sp.Matrix(output_row)
        if row.shape == (n, 1):
            row = row.T
        if row.shape != (1, n):
            raise ValueError(f"output_row must have shape (1, {n})")
    x = system.variable
    rows = [row]
    for _ in range(n):
        rows.append(rows[-1].diff(x) + rows[-1] * sp.Matrix(system.matrix))
    cyclic = sp.Matrix.vstack(*rows[:-1])
    determinant = sp.factor(cyclic.det())
    if determinant == 0 or determinant.is_zero is True:
        return CyclicScalarization(
            system,
            sp.ImmutableMatrix(row),
            sp.ImmutableMatrix(cyclic),
            None,
            False,
            "chosen output row is not cyclic",
        )
    if determinant.is_zero is None and determinant.free_symbols - {x}:
        return CyclicScalarization(
            system,
            sp.ImmutableMatrix(row),
            sp.ImmutableMatrix(cyclic),
            None,
            False,
            "cyclicity is parameter-dependent and was not certified",
        )
    try:
        alpha = (rows[-1] * cyclic.inv()).applyfunc(lambda e: sp.cancel(sp.together(e)))
    except (ValueError, ZeroDivisionError):
        return CyclicScalarization(
            system,
            sp.ImmutableMatrix(row),
            sp.ImmutableMatrix(cyclic),
            None,
            False,
            "cyclic matrix inversion failed",
        )
    coeffs = (*tuple(sp.cancel(-alpha[0, j]) for j in range(n)), sp.S.One)
    if function is None:
        function = sp.Function("u")
    operator = LinearDifferentialOperator(x, function, coeffs)
    return CyclicScalarization(
        system, sp.ImmutableMatrix(row), sp.ImmutableMatrix(cyclic), operator, True
    )


@dataclass(frozen=True)
class ScalarSystemCorrespondence:
    """Explicit companion/scalarization round-trip evidence."""

    operator: LinearDifferentialOperator
    system: FirstOrderSystem
    scalarization: CyclicScalarization

    def verify(self) -> bool:
        if not self.scalarization.verify() or self.scalarization.operator is None:
            return False
        left = self.operator.normalized().coefficients
        right = self.scalarization.operator.normalized().coefficients
        return all(sp.simplify(a - b) == 0 for a, b in zip(left, right, strict=True))


def scalar_system_correspondence(
    operator: LinearDifferentialOperator,
) -> ScalarSystemCorrespondence:
    """Build and verify the standard scalar -> companion -> scalar round trip."""
    system = companion_system(operator)
    scalarization = scalarize_system(system, function=operator.function)
    return ScalarSystemCorrespondence(operator, system, scalarization)
