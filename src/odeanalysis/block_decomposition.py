"""Formal exponential-block decomposition for first-order systems.

This module implements the system-level splitting stage that sits between the
scalar Newton--Riccati analysis and Levelt reduction.  The basic mechanism is
formal spectral separation.  After a constant generalized-eigenvector change
of basis exposes distinct spectral groups at an irregular Laurent order, the
off-diagonal connection blocks are removed recursively by near-identity gauges
whose coefficients solve exact Sylvester equations.

For scalar equations, the completed exponential parts found by :mod:`odeanalysis.formal`
are used to choose a Newton shearing of the ramified companion system.  This
usually turns the leading Riccati characteristic roots into ordinary matrix
eigenvalues and lets the system splitter isolate repeated exponential blocks.

The implementation is restricted to cases it can certify.  A nonscalar coefficient
with only one eigenvalue at the first unresolved irregular order signals that a
further Moser/Newton shearing is needed; such a block is returned as unresolved
rather than guessed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ._block_common import (
    BlockDecompositionError,
)
from ._block_common import (
    block_diag_series as _block_diag_series,
)
from ._block_common import (
    is_scalar_matrix as _is_scalar_matrix,
)
from ._block_common import (
    is_zero_matrix as _is_zero_matrix,
)
from ._block_common import (
    matrix_block as _matrix_block,
)
from ._block_common import (
    partition_offsets as _partition_offsets,
)
from ._block_common import (
    series_block as _series_block,
)
from ._formal_gauge import (
    GaugeTransformationVerification,
)
from ._formal_gauge import (
    constant_series as _constant_series,
)
from ._formal_gauge import (
    formal_gauge_transform as _formal_gauge_transform,
)
from ._formal_gauge import (
    multiply_gauges as _multiply_gauges,
)
from ._moser import MoserShearingStep, moser_reduce
from ._power_simplify import analytic_powsimp
from ._spectral import (
    FormalSpectralSplit,
    SpectralSplitVerification,
    first_irregular_spectral_data,
)
from ._spectral import (
    generalized_eigenbasis as _generalized_eigenbasis,
)
from ._symbolic_errors import SYMBOLIC_FAILURES
from .formal import complete_formal_exponential_parts
from .matrix_series import MatrixLaurentSeries
from .newton import localize_operator
from .operator import LinearDifferentialOperator
from .system import (
    FirstOrderSystem,
    FormalBlockPartition,
    FormalExponentialBlockMetadata,
    companion_system,
    formal_block_partition,
)

if TYPE_CHECKING:
    from .formal import CompleteFormalExponentialPart


def _solve_sylvester(
    left: sp.MatrixBase,
    right: sp.MatrixBase,
    target: sp.MatrixBase,
) -> sp.Matrix:
    """Solve ``left*X - X*right = target`` exactly and uniquely."""

    left = sp.Matrix(left)
    right = sp.Matrix(right)
    target = sp.Matrix(target)
    rows, cols = target.shape
    symbols = sp.symbols(f"_x0:{rows * cols}")
    x = sp.Matrix(rows, cols, symbols)
    equations = list(left * x - x * right - target)
    coefficient_matrix, rhs = sp.linear_eq_to_matrix(equations, symbols)
    try:
        solution_set = sp.linsolve((coefficient_matrix, rhs), symbols)
    except SYMBOLIC_FAILURES as exc:  # pragma: no cover - SymPy backend variation
        raise BlockDecompositionError("could not solve the Sylvester equation") from exc
    solutions = list(solution_set)
    if len(solutions) != 1:
        raise BlockDecompositionError("Sylvester equation did not have a unique solution")
    solution = solutions[0]
    # Parameters from the original coefficient field are allowed.  Detect only
    # linsolve-generated tau symbols by checking whether a solution still
    # contains one of the unknown symbols or a Dummy-like free parameter.
    if any(sp.sympify(item).has(*symbols) for item in solution):
        raise BlockDecompositionError("Sylvester equation left unresolved matrix entries")
    generated = set().union(*(sp.sympify(item).free_symbols for item in solution))
    original = set().union(
        *(entry.free_symbols for entry in list(left) + list(right) + list(target))
    )
    if generated - original:
        raise BlockDecompositionError("Sylvester equation introduced free parameters")
    return sp.Matrix(rows, cols, tuple(sp.simplify(item) for item in solution))


def _off_block_part(matrix: sp.MatrixBase, dimensions: Sequence[int]) -> sp.Matrix:
    matrix = sp.Matrix(matrix)
    offsets = _partition_offsets(dimensions)
    result = sp.zeros(*matrix.shape)
    for i in range(len(dimensions)):
        for j in range(len(dimensions)):
            if i == j:
                continue
            result[
                offsets[i] : offsets[i + 1],
                offsets[j] : offsets[j + 1],
            ] = matrix[
                offsets[i] : offsets[i + 1],
                offsets[j] : offsets[j + 1],
            ]
    return result


def _diagonalize_partition(
    connection: MatrixLaurentSeries,
    dimensions: tuple[int, ...],
    *,
    pivot_power: int,
    max_power: int,
) -> tuple[MatrixLaurentSeries, MatrixLaurentSeries]:
    """Eliminate off-block terms recursively through ``max_power``."""

    variable = connection.variable
    size = connection.rows
    current = connection
    gauge = MatrixLaurentSeries.identity(variable, size)
    offsets = _partition_offsets(dimensions)
    pivot = sp.Matrix(current.coefficient(pivot_power))
    pivot_blocks = [
        pivot[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]]
        for i in range(len(dimensions))
    ]

    # At target order pivot_power+n, X_n first appears through [A_p, X_n].
    # Since exponential splitting is only performed for p < -1, G_n' occurs
    # at the later order n-1 and is automatically handled in subsequent steps.
    for n in range(1, max_power - pivot_power + 1):
        target_power = pivot_power + n
        coefficient = sp.Matrix(current.coefficient(target_power))
        x = sp.zeros(size)
        nonzero = False
        for i in range(len(dimensions)):
            for j in range(len(dimensions)):
                if i == j:
                    continue
                row_slice = slice(offsets[i], offsets[i + 1])
                col_slice = slice(offsets[j], offsets[j + 1])
                coupling = coefficient[row_slice, col_slice]
                if _is_zero_matrix(coupling):
                    continue
                # C + A_i X - X A_j = 0.
                correction = _solve_sylvester(
                    pivot_blocks[i],
                    pivot_blocks[j],
                    -coupling,
                )
                x[row_slice, col_slice] = correction
                nonzero = True
        if not nonzero:
            continue
        step = MatrixLaurentSeries.from_mapping(
            variable,
            {0: sp.eye(size), n: x},
            shape=(size, size),
        )
        current = _formal_gauge_transform(current, step, max_power=max_power)
        gauge = _multiply_gauges(gauge, step, max_power=max_power - pivot_power)

    return current, gauge


@dataclass(frozen=True)
class RamificationStep:
    """One exact cover change ``t = u**index`` used in formal reduction."""

    index: int
    source: MatrixLaurentSeries
    transformed: MatrixLaurentSeries

    def verify(self) -> bool:
        """Verify the recorded cover pullback from its stored source data."""

        if self.index < 1 or self.source.variable == self.transformed.variable:
            return False
        expected = ramified_pullback(self.source, self.transformed.variable, self.index)
        return expected == self.transformed


@dataclass(frozen=True)
class LeveltTurrittinReduction:
    """Recursive formal reduction on a finite ramified cover.

    The result records every ramification and formal reduction stage.  The
    stored data can be verified independently without repeating the search for
    a successful cover.
    """

    original_connection: MatrixLaurentSeries
    transformed_connection: MatrixLaurentSeries
    ramification_index: int
    ramifications: tuple[RamificationStep, ...]
    formal_stages: tuple[FormalBlockDiagonalization, ...]
    complete: bool
    limitation: str | None = None

    @property
    def final_diagonalization(self) -> FormalBlockDiagonalization:
        """Return the final formal block-diagonalization stage."""

        return self.formal_stages[-1]

    def verify(self) -> bool:
        """Verify cover chaining, total ramification, and final-stage metadata."""

        if not self.formal_stages:
            return False
        if self.ramification_index != _ramification_product(self.ramifications):
            return False
        if any(not step.verify() for step in self.ramifications):
            return False
        if any(not stage.verify() for stage in self.formal_stages):
            return False
        if len(self.formal_stages) != len(self.ramifications) + 1:
            return False
        if self.formal_stages[0].original_connection != self.original_connection:
            return False
        for index, step in enumerate(self.ramifications):
            before = self.formal_stages[index]
            after = self.formal_stages[index + 1]
            if step.source != before.transformed_connection:
                return False
            if step.transformed != after.original_connection:
                return False
        final = self.formal_stages[-1]
        if self.transformed_connection != final.transformed_connection:
            return False
        if self.complete != final.complete:
            return False
        return not (self.complete and self.limitation is not None)


def _ramification_product(steps: tuple[RamificationStep, ...]) -> int:
    """Return the total cover index represented by ``steps``."""

    product = 1
    for step in steps:
        product *= step.index
    return product


@dataclass(frozen=True)
class FormalBlockDiagonalization:
    """Truncated formal block diagonalization of a Laurent connection."""

    original_connection: MatrixLaurentSeries
    transformed_connection: MatrixLaurentSeries
    gauge: MatrixLaurentSeries
    block_dimensions: tuple[int, ...]
    block_slices: tuple[tuple[int, int], ...]
    spectral_splits: tuple[FormalSpectralSplit, ...]
    moser_steps: tuple[MoserShearingStep, ...]
    max_power: int
    complete: bool
    limitation: str | None = None
    gauge_verification: GaugeTransformationVerification | None = None

    def verify(self) -> bool:
        """Verify spectral evidence, gauge action, residual, and completion claim.

        ``complete`` is a mathematical claim, not merely search metadata.  A
        completed Levelt--Turrittin block may have scalar irregular terms
        (the common exponential part), but no nonscalar coefficient may remain
        below the residue order.
        """

        if self.gauge_verification is None or not self.gauge_verification.verify():
            return False
        if any(dimension <= 0 for dimension in self.block_dimensions):
            return False
        if sum(self.block_dimensions) != self.transformed_connection.rows:
            return False
        offsets = _partition_offsets(self.block_dimensions)
        expected_slices = tuple(
            (offsets[i], offsets[i + 1]) for i in range(len(self.block_dimensions))
        )
        if self.block_slices != expected_slices:
            return False
        if any(not split.verify() for split in self.spectral_splits):
            return False
        residual = self.off_block_residual().truncate(max_power=self.max_power)
        if self.complete and not residual.is_zero:
            return False
        if self.complete and not self._completion_criterion():
            return False
        return True

    def _completion_criterion(self) -> bool:
        """Independently check the reduced-block Levelt--Turrittin criterion."""

        for block in self.blocks:
            for power, coefficient in block.terms:
                if power >= -1:
                    continue
                scalar, _ = _is_scalar_matrix(sp.Matrix(coefficient))
                if not scalar:
                    return False
        return True

    @property
    def blocks(self) -> tuple[MatrixLaurentSeries, ...]:
        return tuple(
            _series_block(self.transformed_connection, start, stop, start, stop)
            for start, stop in self.block_slices
        )

    def off_block_residual(self) -> MatrixLaurentSeries:
        coefficients: dict[int, sp.Matrix] = {}
        for power, coefficient in self.transformed_connection.terms:
            off = _off_block_part(coefficient, self.block_dimensions)
            if not _is_zero_matrix(off):
                coefficients[power] = off
        return MatrixLaurentSeries.from_mapping(
            self.transformed_connection.variable,
            coefficients,
            shape=self.transformed_connection.shape,
        )


@dataclass(frozen=True)
class _RecursiveResult:
    transformed: MatrixLaurentSeries
    gauge: MatrixLaurentSeries
    leaf_dimensions: tuple[int, ...]
    splits: tuple[FormalSpectralSplit, ...]
    moser_steps: tuple[MoserShearingStep, ...]
    complete: bool
    limitation: str | None


def _first_splitting_coefficient(
    connection: MatrixLaurentSeries,
) -> tuple[int, sp.Matrix, tuple[sp.Expr, ...]] | None:
    """Find the first irregular coefficient that can spectrally split a block."""

    data = first_irregular_spectral_data(connection)
    if data is None or not data.distinct:
        return None
    return (
        data.power,
        sp.Matrix(data.coefficient),
        tuple(value for value, _ in data.eigenvalues),
    )


def _recursive_diagonalize(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
) -> _RecursiveResult:
    size = connection.rows
    variable = connection.variable
    split_data = _first_splitting_coefficient(connection)
    if split_data is None:
        irregular = first_irregular_spectral_data(connection)
        unresolved = irregular is not None and not irregular.distinct
        limitation = None
        if unresolved:
            limitation = (
                "an unresolved irregular coefficient has only one eigenvalue; "
                "an additional Moser/Newton shearing is required"
            )
        if unresolved:
            reduction = moser_reduce(connection, max_power=max_power)
            if reduction.steps:
                child = _recursive_diagonalize(
                    reduction.transformed_connection, max_power=max_power
                )
                total_gauge = _multiply_gauges(
                    reduction.gauge,
                    child.gauge,
                    max_power=max_power - (connection.min_power or 0) + connection.rows,
                )
                return _RecursiveResult(
                    transformed=child.transformed,
                    gauge=total_gauge,
                    leaf_dimensions=child.leaf_dimensions,
                    splits=child.splits,
                    moser_steps=reduction.steps + child.moser_steps,
                    complete=child.complete,
                    limitation=child.limitation,
                )
            limitation = reduction.limitation or limitation
        return _RecursiveResult(
            transformed=connection,
            gauge=MatrixLaurentSeries.identity(variable, size),
            leaf_dimensions=(size,),
            splits=(),
            moser_steps=(),
            complete=not unresolved,
            limitation=limitation,
        )

    pivot_power, pivot_coefficient, _ = split_data
    change, dimensions, eigenvalues, projectors = _generalized_eigenbasis(pivot_coefficient)
    constant_gauge = _constant_series(variable, change)
    transformed = _formal_gauge_transform(connection, constant_gauge, max_power=max_power)
    transformed, near_identity = _diagonalize_partition(
        transformed,
        dimensions,
        pivot_power=pivot_power,
        max_power=max_power,
    )
    stage_gauge = _multiply_gauges(
        constant_gauge,
        near_identity,
        max_power=max_power - (connection.min_power or 0),
    )
    split = FormalSpectralSplit(
        pivot_power=pivot_power,
        eigenvalues=eigenvalues,
        dimensions=dimensions,
        projectors=projectors,
        verification=SpectralSplitVerification(
            coefficient=sp.ImmutableMatrix(pivot_coefficient),
            eigenvalues=eigenvalues,
            dimensions=dimensions,
            projectors=projectors,
        ),
    )

    offsets = _partition_offsets(dimensions)
    child_results: list[_RecursiveResult] = []
    for index, _dimension in enumerate(dimensions):
        start, stop = offsets[index], offsets[index + 1]
        child = _series_block(transformed, start, stop, start, stop)
        child_results.append(_recursive_diagonalize(child, max_power=max_power))

    if any(
        not _is_zero_matrix(
            _matrix_block(
                transformed.coefficient(power),
                offsets[i],
                offsets[i + 1],
                offsets[j],
                offsets[j + 1],
            )
        )
        for power, _ in transformed.terms
        for i in range(len(dimensions))
        for j in range(len(dimensions))
        if i != j
    ):
        return _RecursiveResult(
            transformed=transformed,
            gauge=stage_gauge,
            leaf_dimensions=dimensions,
            splits=(split,),
            moser_steps=(),
            complete=False,
            limitation="off-block terms remained after formal Sylvester reduction",
        )

    child_gauge = _block_diag_series(variable, [child.gauge for child in child_results])
    if child_gauge.terms:
        transformed = _formal_gauge_transform(transformed, child_gauge, max_power=max_power)
        total_gauge = _multiply_gauges(
            stage_gauge,
            child_gauge,
            max_power=max_power - (connection.min_power or 0),
        )
    else:
        total_gauge = stage_gauge

    leaf_dimensions = tuple(
        dimension for child in child_results for dimension in child.leaf_dimensions
    )
    splits = (
        split,
        *tuple(nested for child in child_results for nested in child.splits),
    )
    complete = all(child.complete for child in child_results)
    limitation = next(
        (child.limitation for child in child_results if child.limitation),
        None,
    )
    return _RecursiveResult(
        transformed=transformed,
        gauge=total_gauge,
        leaf_dimensions=leaf_dimensions,
        splits=splits,
        moser_steps=tuple(step for child in child_results for step in child.moser_steps),
        complete=complete,
        limitation=limitation,
    )


def ramified_pullback(
    connection: MatrixLaurentSeries,
    cover_variable: sp.Symbol,
    index: int,
) -> MatrixLaurentSeries:
    r"""Pull a connection back by ``t = u**index``.

    If ``Y_t = A(t)Y`` and ``t=u**r``, then
    ``Y_u = r*u**(r-1)*A(u**r)Y``.  The sparse Laurent representation makes
    this transformation exact and avoids generic substitution or series calls.
    """

    if index < 1:
        raise ValueError("ramification index must be positive")
    if cover_variable == connection.variable:
        raise ValueError("cover variable must differ from the source variable")
    coefficients: dict[int, sp.Matrix] = {}
    for power, coefficient in connection.terms:
        cover_power = index * power + index - 1
        coefficients[cover_power] = index * sp.Matrix(coefficient)
    return MatrixLaurentSeries.from_mapping(cover_variable, coefficients, shape=connection.shape)


def levelt_turrittin_reduce(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
    max_depth: int = 2,
    max_cover_index: int | None = None,
) -> LeveltTurrittinReduction:
    """Recursively reduce repeated irregular blocks on finite covers.

    Each level first performs the ordinary exact Moser/spectral reduction.
    Ramification is attempted only if that stage ends at a nonscalar irregular
    coefficient with one eigenvalue of full multiplicity.  The next cover is
    applied to the *transformed* unresolved connection, so successful Moser
    work is never discarded.  All stages and cover pullbacks are retained for
    independent verification.
    """

    if max_power < -1:
        raise ValueError("max_power must include at least the residue order -1")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    if connection.rows != connection.cols:
        raise ValueError("Levelt-Turrittin reduction requires a square connection")
    if max_cover_index is None:
        max_cover_index = max(2, connection.rows)
    if max_cover_index < 2 and max_depth:
        raise ValueError("max_cover_index must be at least 2 when covers are enabled")

    def descend(
        current: MatrixLaurentSeries,
        current_max: int,
        depth: int,
        total_index: int,
        steps: tuple[RamificationStep, ...],
        stages: tuple[FormalBlockDiagonalization, ...],
        diagonal: FormalBlockDiagonalization | None = None,
    ) -> LeveltTurrittinReduction:
        if diagonal is None:
            diagonal = formal_block_diagonalize(current, max_power=current_max)
        all_stages = (*stages, diagonal)
        if diagonal.complete or depth >= max_depth:
            return LeveltTurrittinReduction(
                original_connection=connection,
                transformed_connection=diagonal.transformed_connection,
                ramification_index=total_index,
                ramifications=steps,
                formal_stages=all_stages,
                complete=diagonal.complete,
                limitation=diagonal.limitation,
            )

        unresolved = diagonal.transformed_connection
        if first_irregular_spectral_data(unresolved) is None:
            return LeveltTurrittinReduction(
                original_connection=connection,
                transformed_connection=unresolved,
                ramification_index=total_index,
                ramifications=steps,
                formal_stages=all_stages,
                complete=False,
                limitation=diagonal.limitation,
            )

        for index in range(2, max_cover_index + 1):
            cover = sp.Dummy(f"{current.variable.name}_cover", positive=True)
            pulled = ramified_pullback(unresolved, cover, index)
            cover_max = index * current_max + index - 1
            candidate = formal_block_diagonalize(pulled, max_power=cover_max)
            step = RamificationStep(index=index, source=unresolved, transformed=pulled)
            if candidate.complete:
                return LeveltTurrittinReduction(
                    original_connection=connection,
                    transformed_connection=candidate.transformed_connection,
                    ramification_index=total_index * index,
                    ramifications=(*steps, step),
                    formal_stages=(*all_stages, candidate),
                    complete=True,
                    limitation=None,
                )
            if depth + 1 < max_depth:
                nested = descend(
                    pulled,
                    cover_max,
                    depth + 1,
                    total_index * index,
                    (*steps, step),
                    all_stages,
                    candidate,
                )
                if nested.complete:
                    return nested

        return LeveltTurrittinReduction(
            original_connection=connection,
            transformed_connection=unresolved,
            ramification_index=total_index,
            ramifications=steps,
            formal_stages=all_stages,
            complete=False,
            limitation=(
                diagonal.limitation
                or "bounded ramified reduction did not resolve the irregular block"
            ),
        )

    return descend(connection, max_power, 0, 1, (), ())


def formal_block_diagonalize(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
) -> FormalBlockDiagonalization:
    """Recursively split and block-diagonalize an irregular Laurent system.

    Scalar irregular coefficients are skipped because they are common
    exponential factors.  At the first nonscalar irregular order with two or
    more distinct eigenvalues, generalized eigenspaces give a constant block
    basis and exact Sylvester equations remove off-block terms order by order.
    The procedure then recurses inside each diagonal block.
    """

    if connection.rows != connection.cols:
        raise ValueError("formal block diagonalization requires a square connection")
    if max_power < -1:
        raise ValueError("max_power must include at least the residue order -1")
    result = _recursive_diagonalize(connection, max_power=max_power)
    offsets = _partition_offsets(result.leaf_dimensions)
    slices = tuple((offsets[i], offsets[i + 1]) for i in range(len(result.leaf_dimensions)))
    decomposition = FormalBlockDiagonalization(
        original_connection=connection,
        transformed_connection=result.transformed,
        gauge=result.gauge,
        block_dimensions=result.leaf_dimensions,
        block_slices=slices,
        spectral_splits=result.splits,
        moser_steps=result.moser_steps,
        max_power=max_power,
        complete=result.complete,
        limitation=result.limitation,
        gauge_verification=GaugeTransformationVerification(
            source=connection,
            gauge=result.gauge,
            transformed=result.transformed,
            max_power=max_power,
        ),
    )
    residual = decomposition.off_block_residual().truncate(max_power=max_power)
    if not residual.is_zero:
        return FormalBlockDiagonalization(
            original_connection=decomposition.original_connection,
            transformed_connection=decomposition.transformed_connection,
            gauge=decomposition.gauge,
            block_dimensions=decomposition.block_dimensions,
            block_slices=decomposition.block_slices,
            spectral_splits=decomposition.spectral_splits,
            moser_steps=decomposition.moser_steps,
            max_power=decomposition.max_power,
            complete=False,
            limitation="off-block residual is nonzero at the requested truncation order",
            gauge_verification=decomposition.gauge_verification,
        )
    return decomposition


def _expand_matrix_laurent(
    matrix: sp.MatrixBase,
    variable: sp.Symbol,
    *,
    max_power: int,
) -> MatrixLaurentSeries:
    """Laurent-expand a meromorphic matrix at zero through ``max_power``."""

    matrix = sp.Matrix(matrix)
    expanded = sp.zeros(matrix.rows, matrix.cols)
    # series(..., n) retains all principal-part terms and terms below n.
    order = max_power + 1
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            entry = matrix[i, j]
            try:
                truncated = sp.series(entry, variable, 0, order).removeO()
            except SYMBOLIC_FAILURES as exc:
                raise BlockDecompositionError(
                    f"could not Laurent-expand system entry {entry!s}"
                ) from exc
            expanded[i, j] = sp.expand(truncated)
    return MatrixLaurentSeries.from_matrix(expanded, variable).truncate(max_power=max_power)


def _uniformized_q(
    part: CompleteFormalExponentialPart,
    local_coordinate: sp.Symbol,
    parameter: sp.Symbol,
    ramification: int,
) -> sp.Expr:
    q = part.local_exponential_polynomial.subs(part.local_coordinate, local_coordinate)
    return analytic_powsimp(sp.expand(q.subs(local_coordinate, parameter**ramification)))


def _uniformized_log_derivative_h(
    q_t: sp.Expr,
    parameter: sp.Symbol,
    ramification: int,
) -> sp.Expr:
    return sp.cancel(
        sp.together(sp.diff(q_t, parameter) / (ramification * parameter ** (ramification - 1)))
    )


def _leading_integral_power(expr: sp.Expr, variable: sp.Symbol) -> int:
    expr = sp.expand(expr)
    powers: list[int] = []
    for term in sp.Add.make_args(expr):
        coefficient, power = term.as_coeff_exponent(variable)
        power = sp.sympify(power)
        if coefficient.has(variable) or power.is_Integer is not True:
            raise BlockDecompositionError(
                f"expected an integral Laurent polynomial in {variable!s}, got {expr!s}"
            )
        powers.append(int(power))
    if not powers:
        raise BlockDecompositionError("cannot determine the leading power of zero")
    return min(powers)


def _newton_shearing_exponent(
    parts: Sequence[CompleteFormalExponentialPart],
    local_coordinate: sp.Symbol,
    parameter: sp.Symbol,
    ramification: int,
) -> int:
    pole_orders: list[int] = []
    for part in parts:
        q_t = _uniformized_q(part, local_coordinate, parameter, ramification)
        if q_t == 0:
            continue
        w_h = _uniformized_log_derivative_h(q_t, parameter, ramification)
        power = _leading_integral_power(w_h, parameter)
        pole_orders.append(max(0, -power))
    return max(pole_orders, default=0)


def _shearing_matrix(parameter: sp.Symbol, dimension: int, exponent: int) -> sp.ImmutableMatrix:
    if exponent < 0:
        raise ValueError("shearing exponent must be nonnegative")
    return sp.ImmutableMatrix(
        sp.diag(*(parameter ** (-index * exponent) for index in range(dimension)))
    )


def _block_exponential_polynomial(block: MatrixLaurentSeries) -> tuple[sp.Expr, bool]:
    """Extract the common scalar exponential polynomial of an isolated block.

    A repeated exponential block may still carry a nilpotent irregular part in
    the current system presentation.  If an irregular coefficient has a
    single eigenvalue of full algebraic multiplicity, that eigenvalue is the
    common scalar exponential derivative; the nonscalar remainder is recorded
    by returning ``regular_after_exp=False`` but does not prevent the
    block itself from being identified.
    """

    variable = block.variable
    derivative = sp.S.Zero
    regular_after_exp = True
    for power, coefficient_immutable in block.terms:
        if power >= -1:
            break
        coefficient = sp.Matrix(coefficient_immutable)
        scalar, value = _is_scalar_matrix(coefficient)
        if scalar:
            derivative += value * variable**power
            continue
        eigenvalues = coefficient.eigenvals()
        if len(eigenvalues) != 1:
            return sp.S.Zero, False
        value, multiplicity = next(iter(eigenvalues.items()))
        if int(multiplicity) != coefficient.rows:
            return sp.S.Zero, False
        derivative += sp.simplify(value) * variable**power
        regular_after_exp = False
    if derivative == 0:
        return sp.S.Zero, regular_after_exp
    return sp.expand(sp.integrate(derivative, variable)), regular_after_exp


def _same_q(left: sp.Expr, right: sp.Expr) -> bool:
    difference = sp.expand(left - right)
    # Exponential polynomials are defined only up to an additive constant.
    variable_candidates = tuple(difference.free_symbols)
    if not variable_candidates:
        return True
    variable = variable_candidates[0]
    return sp.simplify(sp.diff(difference, variable)) == 0


def _match_block_metadata(
    q: sp.Expr,
    dimension: int,
    metadata: FormalBlockPartition,
    parts: Sequence[CompleteFormalExponentialPart],
    local_coordinate: sp.Symbol,
    parameter: sp.Symbol,
) -> FormalExponentialBlockMetadata | None:
    for block in metadata.blocks:
        if block.dimension != dimension:
            continue
        representative = parts[block.part_indices[0]]
        expected = _uniformized_q(
            representative,
            local_coordinate,
            parameter,
            metadata.ramification_index,
        )
        if _same_q(q, expected):
            return block
    return None


def _row_series_from_matrix(
    row: sp.MatrixBase,
    variable: sp.Symbol,
    *,
    max_power: int,
) -> MatrixLaurentSeries:
    return _expand_matrix_laurent(sp.Matrix(row), variable, max_power=max_power)


@dataclass(frozen=True)
class FormalExponentialSystemBlock:
    """One isolated exponential block on the common uniformizing cover."""

    index: int
    start: int
    stop: int
    connection: MatrixLaurentSeries
    output_row: MatrixLaurentSeries
    parameter_exponential_polynomial: sp.Expr
    metadata: FormalExponentialBlockMetadata | None
    regular_after_exp: bool

    @property
    def dimension(self) -> int:
        return self.stop - self.start


def cyclic_scalar_operator(
    connection: MatrixLaurentSeries | sp.MatrixBase,
    output_row: MatrixLaurentSeries | sp.MatrixBase,
    *,
    function_name: str = "_U",
) -> LinearDifferentialOperator:
    """Scalarize an isolated system block through a cyclic output row.

    If ``Z' = B Z`` and ``u = c Z``, define row jets recursively by
    ``r_0=c`` and ``r_{k+1}=r_k' + r_k B``.  When the first ``m`` rows form
    an invertible matrix, ``r_m`` is expressed in that basis and yields the
    monic order-``m`` scalar equation annihilating the physical output ``u``.
    """

    if isinstance(connection, MatrixLaurentSeries):
        variable = connection.variable
        b = sp.Matrix(connection.to_matrix())
    else:
        b = sp.Matrix(connection)
        symbols = sorted(set().union(*(entry.free_symbols for entry in b)), key=sp.default_sort_key)
        if len(symbols) != 1:
            raise ValueError("matrix input must involve exactly one independent variable")
        variable = symbols[0]
    if b.rows != b.cols:
        raise ValueError("connection must be square")
    if isinstance(output_row, MatrixLaurentSeries):
        if output_row.variable != variable:
            raise ValueError("connection and output row use different variables")
        c = sp.Matrix(output_row.to_matrix())
    else:
        c = sp.Matrix(output_row)
    if c.shape != (1, b.rows):
        raise ValueError(f"output row must have shape (1, {b.rows})")

    rows = [c]
    for _ in range(b.rows):
        previous = rows[-1]
        rows.append(
            (previous.diff(variable) + previous * b).applyfunc(
                lambda entry: sp.cancel(sp.together(entry))
            )
        )
    cyclic = sp.Matrix.vstack(*rows[:-1])
    if sp.simplify(cyclic.det()) == 0:
        raise BlockDecompositionError("chosen block output is not a cyclic vector")
    coefficients = (rows[-1] * cyclic.inv()).applyfunc(lambda entry: sp.cancel(sp.together(entry)))
    # u^(m) = sum_j coefficients[j] u^(j).
    operator_coefficients = (
        *tuple(-coefficients[0, j] for j in range(b.rows)),
        sp.S.One,
    )
    function = sp.Function(function_name)
    return LinearDifferentialOperator(
        variable=variable,
        function=function,
        coefficients=operator_coefficients,
    )


@dataclass(frozen=True)
class ExponentialBlockDecomposition:
    """Formal exponential-block decomposition of a scalar linear ODE."""

    point: sp.Expr
    local_coordinate: sp.Symbol
    parameter: sp.Symbol
    ramification_index: int
    shearing_exponent: int
    partition: FormalBlockPartition
    ramified_system: FirstOrderSystem
    shearing_gauge: sp.ImmutableMatrix
    diagonalization: FormalBlockDiagonalization
    total_gauge: sp.ImmutableMatrix
    blocks: tuple[FormalExponentialSystemBlock, ...]
    max_power: int
    complete: bool
    limitation: str | None = None

    @property
    def block_dimensions(self) -> tuple[int, ...]:
        return tuple(block.dimension for block in self.blocks)


def exponential_block_decomposition(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    max_power: int = 8,
    max_branches: int = 64,
) -> ExponentialBlockDecomposition:
    """Isolate completed exponential blocks of a scalar linear ODE.

    The scalar Riccati analysis supplies completed exponential parts and a
    common ramification.  The ramified companion system is Newton-sheared so
    the most singular logarithmic-derivative roots become matrix eigenvalues;
    recursive spectral/Sylvester reduction then eliminates couplings between
    the resulting invariant formal submodules through ``max_power``.
    """

    localized = localize_operator(ode, function, variable, point=point)
    parts = complete_formal_exponential_parts(
        ode,
        function,
        variable,
        point=point,
        max_branches=max_branches,
    )
    partition = formal_block_partition(parts)
    if partition.total_dimension != localized.operator.order:
        raise BlockDecompositionError(
            "completed Riccati branches do not account for the scalar operator order"
        )
    ramification = partition.ramification_index
    parameter = sp.Dummy("t", positive=True)
    local_system = companion_system(localized.operator)
    ramified = local_system.ramify(parameter, ramification)
    shear_exponent = _newton_shearing_exponent(
        parts,
        localized.local_variable,
        parameter,
        ramification,
    )
    shear = _shearing_matrix(parameter, ramified.dimension, shear_exponent)
    sheared_system = ramified.gauge_transform(shear)
    connection = _expand_matrix_laurent(
        sheared_system.matrix,
        parameter,
        max_power=max_power,
    )
    diagonalization = formal_block_diagonalize(connection, max_power=max_power)

    formal_gauge_matrix = sp.Matrix(diagonalization.gauge.to_matrix())
    total_gauge = sp.ImmutableMatrix((sp.Matrix(shear) * formal_gauge_matrix).applyfunc(sp.expand))
    output = sp.Matrix(total_gauge)[0:1, :]
    offsets = _partition_offsets(diagonalization.block_dimensions)
    blocks: list[FormalExponentialSystemBlock] = []
    matched_metadata: set[tuple[int, ...]] = set()
    all_regular = True
    for index, (start, stop) in enumerate(
        (offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)
    ):
        block_connection = _series_block(
            diagonalization.transformed_connection,
            start,
            stop,
            start,
            stop,
        )
        q_parameter, regular_after_q = _block_exponential_polynomial(block_connection)
        all_regular = all_regular and regular_after_q
        metadata = _match_block_metadata(
            q_parameter,
            stop - start,
            partition,
            parts,
            localized.local_variable,
            parameter,
        )
        if metadata is not None:
            matched_metadata.add(metadata.part_indices)
        row = output[:, start:stop]
        row_series = _row_series_from_matrix(row, parameter, max_power=max_power)
        blocks.append(
            FormalExponentialSystemBlock(
                index=index,
                start=start,
                stop=stop,
                connection=block_connection,
                output_row=row_series,
                parameter_exponential_polynomial=sp.simplify(q_parameter),
                metadata=metadata,
                regular_after_exp=regular_after_q,
            )
        )

    expected_metadata = {block.part_indices for block in partition.blocks}
    metadata_complete = matched_metadata == expected_metadata
    dimension_match = sorted(block.dimension for block in blocks) == sorted(
        partition.block_dimensions
    )
    # At this layer ``complete`` means that the distinct completed exponential
    # factors have been isolated as invariant formal submodules.  A repeated
    # block may still need an internal Moser/Levelt reduction; that belongs to
    # the next stage and does not invalidate the exponential splitting itself.
    complete = (
        metadata_complete and dimension_match and diagonalization.off_block_residual().is_zero
    )
    limitation = None
    if not dimension_match:
        limitation = "spectral block dimensions do not match completed Riccati multiplicities"
    elif not metadata_complete:
        limitation = "could not match every system block to a completed exponential part"

    return ExponentialBlockDecomposition(
        point=sp.sympify(point),
        local_coordinate=localized.local_variable,
        parameter=parameter,
        ramification_index=ramification,
        shearing_exponent=shear_exponent,
        partition=partition,
        ramified_system=ramified,
        shearing_gauge=shear,
        diagonalization=diagonalization,
        total_gauge=total_gauge,
        blocks=tuple(blocks),
        max_power=max_power,
        complete=complete,
        limitation=limitation,
    )
