"""Moser/Newton integer shearing for unresolved irregular blocks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import sympy as sp

from ._block_common import BlockDecompositionError
from ._formal_gauge import (
    GaugeTransformationVerification,
    constant_series,
    formal_gauge_transform,
    multiply_gauges,
)
from ._spectral import first_irregular_spectral_data
from ._symbolic_errors import SYMBOLIC_FAILURES
from .diagnostics import ReductionDiagnostic
from .matrix_series import MatrixLaurentSeries


class MoserReductionError(BlockDecompositionError):
    """Raised when a requested Moser/shearing reduction cannot be certified."""


@dataclass(frozen=True)
class MoserShearingStep:
    """One exact integer shearing used to lower an irregular obstruction."""

    pivot_power: int
    eigenvalue: sp.Expr
    nilpotent_rank_before: int
    exponents: tuple[int, ...]
    pivot_power_after: int | None
    nilpotent_rank_after: int


@dataclass(frozen=True)
class MoserReduction:
    """Result of recursive Moser/Newton integer shearing."""

    original_connection: MatrixLaurentSeries
    transformed_connection: MatrixLaurentSeries
    gauge: MatrixLaurentSeries
    steps: tuple[MoserShearingStep, ...]
    complete: bool
    limitation: str | None = None
    diagnostics: tuple[ReductionDiagnostic, ...] = ()
    gauge_verification: GaugeTransformationVerification | None = None

    def verify(self) -> bool:
        """Verify the retained total gauge transformation."""

        return self.gauge_verification is not None and self.gauge_verification.verify()


def _single_irregular_data(
    connection: MatrixLaurentSeries,
) -> tuple[int, sp.Matrix, sp.Expr, int] | None:
    data = first_irregular_spectral_data(connection)
    if data is None or data.distinct:
        return None
    eigenvalue = data.single_eigenvalue
    if eigenvalue is None or data.nilpotent_rank is None:
        return None
    return (
        data.power,
        sp.Matrix(data.coefficient),
        eigenvalue,
        data.nilpotent_rank,
    )


def _candidate_measure(connection: MatrixLaurentSeries) -> tuple[int, int]:
    """Return the Moser search measure using one eigenspectrum computation."""

    data = first_irregular_spectral_data(connection)
    if data is None:
        return (0, 0)
    if data.distinct:
        return (0, 1)
    if data.nilpotent_rank is None:
        return (0, 0)
    return (data.power, -data.nilpotent_rank)


def _monomial_shear_transform(
    connection: MatrixLaurentSeries,
    exponents: tuple[int, ...],
    *,
    max_power: int,
) -> MatrixLaurentSeries:
    r"""Apply ``G=diag(t**s_i)`` without requiring a unit leading matrix."""

    if len(exponents) != connection.rows:
        raise ValueError("shearing exponent count must equal the system dimension")
    variable = connection.variable
    coefficients: dict[int, sp.Matrix] = {}
    for power, coefficient_imm in connection.terms:
        coefficient = sp.Matrix(coefficient_imm)
        for i in range(connection.rows):
            for j in range(connection.cols):
                entry = coefficient[i, j]
                if entry == 0:
                    continue
                shifted_power = power + exponents[j] - exponents[i]
                if shifted_power > max_power:
                    continue
                target = coefficients.setdefault(
                    shifted_power, sp.zeros(connection.rows, connection.cols)
                )
                target[i, j] += entry
    residue = coefficients.setdefault(-1, sp.zeros(connection.rows, connection.cols))
    for i, exponent in enumerate(exponents):
        residue[i, i] -= exponent
    return MatrixLaurentSeries.from_mapping(
        variable, coefficients, shape=connection.shape
    ).truncate(max_power=max_power)


def _single_step(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
    max_shear: int,
) -> tuple[MatrixLaurentSeries, MatrixLaurentSeries, MoserShearingStep] | None:
    """Find one exact Jordan-basis integer shearing that improves Moser measure."""

    data = _single_irregular_data(connection)
    if data is None:
        return None
    pivot_power, coefficient, eigenvalue, rank_before = data
    nilpotent = coefficient - eigenvalue * sp.eye(connection.rows)
    if nilpotent.is_zero_matrix:
        return None
    try:
        change, _jordan = nilpotent.jordan_form()
    except SYMBOLIC_FAILURES as exc:  # pragma: no cover
        raise MoserReductionError(
            "could not construct a Jordan basis for Moser reduction"
        ) from exc
    if sp.simplify(change.det()) == 0:
        raise MoserReductionError("Jordan basis for Moser reduction is singular")

    constant = constant_series(connection.variable, change)
    in_jordan_basis = formal_gauge_transform(connection, constant, max_power=max_power)
    before = _candidate_measure(in_jordan_basis)
    best: tuple[tuple[int, int], tuple[int, ...], MatrixLaurentSeries] | None = None

    for weights_tail in combinations_with_replacement(
        range(max_shear + 1), connection.rows - 1
    ):
        weights = (0, *tuple(int(weight) for weight in weights_tail))
        if all(weight == 0 for weight in weights):
            continue
        candidate = _monomial_shear_transform(
            in_jordan_basis, weights, max_power=max_power
        )
        measure = _candidate_measure(candidate)
        if measure <= before:
            continue
        if (
            best is None
            or measure > best[0]
            or (measure == best[0] and sum(weights) < sum(best[1]))
        ):
            best = (measure, weights, candidate)

    if best is None:
        return None
    _, weights, transformed = best
    after_data = _single_irregular_data(transformed)
    if after_data is None:
        pivot_after = None
        rank_after = 0
    else:
        pivot_after, _, _, rank_after = after_data
    shear_matrix = sp.diag(*(connection.variable**weight for weight in weights))
    shear_series = MatrixLaurentSeries.from_matrix(shear_matrix, connection.variable)
    gauge = multiply_gauges(
        constant,
        shear_series,
        max_power=max_power - (connection.min_power or 0) + max(weights),
    )
    step = MoserShearingStep(
        pivot_power=pivot_power,
        eigenvalue=eigenvalue,
        nilpotent_rank_before=rank_before,
        exponents=weights,
        pivot_power_after=pivot_after,
        nilpotent_rank_after=rank_after,
    )
    return transformed, gauge, step


def moser_reduce(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
    max_steps: int = 8,
    max_shear: int | None = None,
) -> MoserReduction:
    """Reduce single-eigenvalue irregular blocks by bounded integer shearing."""

    if connection.rows != connection.cols:
        raise ValueError("Moser reduction requires a square Laurent connection")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if max_shear is None:
        max_shear = max(1, connection.rows)
    if max_shear < 1:
        raise ValueError("max_shear must be positive")

    original = connection
    current = connection
    total_gauge = MatrixLaurentSeries.identity(connection.variable, connection.rows)
    steps: list[MoserShearingStep] = []
    limitation = None
    for _ in range(max_steps):
        data = first_irregular_spectral_data(current)
        if data is None or data.distinct:
            break
        step_result = _single_step(current, max_power=max_power, max_shear=max_shear)
        if step_result is None:
            limitation = (
                "no certified integer Moser/Newton shearing improved the leading "
                "single-eigenvalue irregular block"
            )
            break
        current, gauge, step = step_result
        total_gauge = multiply_gauges(
            total_gauge,
            gauge,
            max_power=max_power - (original.min_power or 0) + max_shear,
        )
        steps.append(step)
    else:
        limitation = "Moser reduction reached the configured step limit"

    final_data = first_irregular_spectral_data(current)
    complete = final_data is None or final_data.distinct
    diagnostics: tuple[ReductionDiagnostic, ...] = ()
    if complete:
        limitation = None
    else:
        diagnostics = (
            ReductionDiagnostic(
                stage="moser",
                code="unresolved-irregular-block",
                message=limitation or "Moser reduction remained incomplete",
                power=final_data.power,
                rank=final_data.nilpotent_rank,
            ),
        )
    verification = GaugeTransformationVerification(
        source=original,
        gauge=total_gauge,
        transformed=current,
        max_power=max_power,
    )
    return MoserReduction(
        original_connection=original,
        transformed_connection=current,
        gauge=total_gauge,
        steps=tuple(steps),
        complete=complete,
        limitation=limitation,
        diagnostics=diagnostics,
        gauge_verification=verification,
    )
