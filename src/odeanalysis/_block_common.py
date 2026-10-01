"""Shared exact helpers for formal block reduction."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from .matrix_series import MatrixLaurentSeries


class BlockDecompositionError(NotImplementedError):
    """Raised when an exact formal block split cannot be certified."""


def is_zero_matrix(matrix: sp.MatrixBase) -> bool:
    """Return whether every matrix entry simplifies exactly to zero."""

    return all(sp.simplify(entry) == 0 for entry in matrix)


def is_scalar_matrix(matrix: sp.MatrixBase) -> tuple[bool, sp.Expr]:
    """Return whether ``matrix`` is scalar and, when so, its scalar value."""

    matrix = sp.Matrix(matrix)
    if matrix.rows != matrix.cols:
        return False, sp.S.Zero
    if matrix.rows == 0:
        return True, sp.S.Zero
    scalar = sp.simplify(matrix[0, 0])
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            expected = scalar if i == j else sp.S.Zero
            if sp.simplify(matrix[i, j] - expected) != 0:
                return False, sp.S.Zero
    return True, scalar


def matrix_block(
    matrix: sp.MatrixBase,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> sp.Matrix:
    """Extract a dense matrix block."""

    return sp.Matrix(matrix)[row_start:row_stop, col_start:col_stop]


def series_block(
    series: MatrixLaurentSeries,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> MatrixLaurentSeries:
    """Extract a block from every coefficient of a Laurent matrix series."""

    rows = row_stop - row_start
    cols = col_stop - col_start
    coefficients: dict[int, sp.Matrix] = {}
    for power, coefficient in series.terms:
        block = matrix_block(coefficient, row_start, row_stop, col_start, col_stop)
        if not is_zero_matrix(block):
            coefficients[power] = block
    return MatrixLaurentSeries.from_mapping(
        series.variable,
        coefficients,
        shape=(rows, cols),
    )


def block_diag_series(
    variable: sp.Symbol,
    series_list: Sequence[MatrixLaurentSeries],
) -> MatrixLaurentSeries:
    """Assemble Laurent matrix series as a block diagonal series."""

    if not series_list:
        return MatrixLaurentSeries.zero(variable, 0, 0)
    if any(series.variable != variable for series in series_list):
        raise ValueError("all block series must use the same variable")
    total_rows = sum(series.rows for series in series_list)
    total_cols = sum(series.cols for series in series_list)
    powers = sorted({power for series in series_list for power, _ in series.terms})
    coefficients: dict[int, sp.Matrix] = {}
    for power in powers:
        blocks = [sp.Matrix(series.coefficient(power)) for series in series_list]
        coefficient = sp.diag(*blocks)
        if not is_zero_matrix(coefficient):
            coefficients[power] = coefficient
    return MatrixLaurentSeries.from_mapping(
        variable,
        coefficients,
        shape=(total_rows, total_cols),
    )


def partition_offsets(dimensions: Sequence[int]) -> tuple[int, ...]:
    """Return cumulative block offsets for ``dimensions``."""

    offsets = [0]
    for dimension in dimensions:
        offsets.append(offsets[-1] + dimension)
    return tuple(offsets)
