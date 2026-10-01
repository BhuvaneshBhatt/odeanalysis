"""Formal gauge transformations and independently verifiable gauge records."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._block_common import BlockDecompositionError
from ._symbolic_errors import SYMBOLIC_FAILURES
from .matrix_series import MatrixLaurentSeries


def constant_series(variable: sp.Symbol, matrix: sp.MatrixBase) -> MatrixLaurentSeries:
    """Represent a constant matrix as a Laurent matrix series."""

    matrix = sp.Matrix(matrix)
    return MatrixLaurentSeries.from_mapping(variable, {0: matrix}, shape=matrix.shape)


def formal_gauge_transform(
    connection: MatrixLaurentSeries,
    gauge: MatrixLaurentSeries,
    *,
    max_power: int,
) -> MatrixLaurentSeries:
    """Apply ``A -> G^-1 A G - G^-1 G'`` through ``max_power``."""

    if connection.rows != connection.cols or gauge.rows != gauge.cols:
        raise ValueError("formal gauge transformation requires square series")
    if connection.shape != gauge.shape:
        raise ValueError("connection and gauge must have the same shape")
    if connection.variable != gauge.variable:
        raise ValueError("connection and gauge use different variables")
    if connection.is_zero:
        connection_min = 0
    else:
        connection_min = connection.min_power
        if connection_min is None:
            raise BlockDecompositionError("nonzero connection has no leading power")

    inverse_order = max(0, max_power - connection_min)
    try:
        inverse = gauge.inverse(max_power=inverse_order)
    except ValueError:
        variable = connection.variable
        g = sp.Matrix(gauge.to_matrix())
        try:
            ginv = g.inv()
        except SYMBOLIC_FAILURES as exc:  # pragma: no cover
            raise BlockDecompositionError("formal gauge is not Laurent-invertible") from exc
        a = sp.Matrix(connection.to_matrix())
        transformed = ginv * a * g - ginv * g.diff(variable)
        expanded = sp.zeros(connection.rows)
        for i in range(connection.rows):
            for j in range(connection.cols):
                try:
                    expanded[i, j] = (
                        sp.series(
                            sp.cancel(sp.together(transformed[i, j])),
                            variable,
                            0,
                            max_power + 1,
                        )
                        .removeO()
                        .expand()
                    )
                except SYMBOLIC_FAILURES as exc:
                    raise BlockDecompositionError(
                        "could not Laurent-expand a general formal gauge transform"
                    ) from exc
        return MatrixLaurentSeries.from_matrix(expanded, variable).truncate(max_power=max_power)
    ag = connection.multiply(gauge, max_power=max_power)
    conjugated = inverse.multiply(ag, max_power=max_power)
    derivative = gauge.derivative(max_power=max_power)
    correction = inverse.multiply(derivative, max_power=max_power)
    return conjugated.add(correction.scale(-1), max_power=max_power)


def multiply_gauges(
    left: MatrixLaurentSeries,
    right: MatrixLaurentSeries,
    *,
    max_power: int,
) -> MatrixLaurentSeries:
    """Multiply two formal gauges through ``max_power``."""

    return left.multiply(right, max_power=max_power)


@dataclass(frozen=True)
class GaugeTransformationVerification:
    """Stored data sufficient to verify one truncated formal gauge transform."""

    source: MatrixLaurentSeries
    gauge: MatrixLaurentSeries
    transformed: MatrixLaurentSeries
    max_power: int

    def verify(self) -> bool:
        """Recompute the stored transform without repeating reduction search."""

        try:
            expected = formal_gauge_transform(
                self.source,
                self.gauge,
                max_power=self.max_power,
            )
        except (ValueError, BlockDecompositionError):
            return False
        return expected == self.transformed
