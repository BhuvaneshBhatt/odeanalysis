"""Exact spectral classification and verification for formal block reduction."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._block_common import BlockDecompositionError, is_scalar_matrix
from ._symbolic_errors import SYMBOLIC_FAILURES
from .matrix_series import MatrixLaurentSeries


@dataclass(frozen=True)
class IrregularSpectralData:
    """First nonscalar irregular coefficient and its exact spectral data."""

    power: int
    coefficient: sp.ImmutableMatrix
    eigenvalues: tuple[tuple[sp.Expr, int], ...]
    nilpotent_rank: int | None

    @property
    def distinct(self) -> bool:
        return len(self.eigenvalues) >= 2

    @property
    def single_eigenvalue(self) -> sp.Expr | None:
        if len(self.eigenvalues) != 1:
            return None
        return self.eigenvalues[0][0]


@dataclass(frozen=True)
class SpectralSplitVerification:
    """Evidence for an exact decomposition into generalized eigenspaces."""

    coefficient: sp.ImmutableMatrix
    eigenvalues: tuple[sp.Expr, ...]
    dimensions: tuple[int, ...]
    projectors: tuple[sp.ImmutableMatrix, ...]

    def verify(self) -> bool:
        """Verify projector partition, dimensions, and spectral invariance."""

        matrix = sp.Matrix(self.coefficient)
        if matrix.rows != matrix.cols:
            return False
        if len(self.eigenvalues) != len(self.dimensions):
            return False
        if len(self.projectors) != len(self.dimensions):
            return False
        if sum(self.dimensions) != matrix.rows:
            return False
        identity = sp.eye(matrix.rows)
        total = sp.zeros(matrix.rows)
        for i, (value, dimension, projector_imm) in enumerate(
            zip(self.eigenvalues, self.dimensions, self.projectors, strict=True)
        ):
            projector = sp.Matrix(projector_imm)
            if projector.shape != matrix.shape or int(projector.rank()) != dimension:
                return False
            if any(
                sp.simplify(entry) != 0 for entry in projector * projector - projector
            ):
                return False
            if any(
                sp.simplify(entry) != 0
                for entry in matrix * projector - projector * matrix
            ):
                return False
            generalized = (matrix - value * identity) ** matrix.rows
            if any(sp.simplify(entry) != 0 for entry in generalized * projector):
                return False
            for other in self.projectors[i + 1 :]:
                if any(
                    sp.simplify(entry) != 0 for entry in projector * sp.Matrix(other)
                ):
                    return False
            total += projector
        return all(sp.simplify(entry) == 0 for entry in total - identity)


@dataclass(frozen=True)
class FormalSpectralSplit:
    """One exact spectral split encountered during recursive reduction."""

    pivot_power: int
    eigenvalues: tuple[sp.Expr, ...]
    dimensions: tuple[int, ...]
    projectors: tuple[sp.ImmutableMatrix, ...]
    verification: SpectralSplitVerification

    def verify(self) -> bool:
        """Verify the retained generalized-eigenspace decomposition."""

        return self.verification.verify()


def first_irregular_spectral_data(
    connection: MatrixLaurentSeries,
) -> IrregularSpectralData | None:
    """Classify the first nonscalar irregular coefficient with one eigen solve."""

    for power, coefficient_imm in connection.terms:
        if power >= -1:
            break
        coefficient = sp.Matrix(coefficient_imm)
        scalar, _ = is_scalar_matrix(coefficient)
        if scalar:
            continue
        try:
            eigenvalues = coefficient.eigenvals()
        except SYMBOLIC_FAILURES as exc:
            raise BlockDecompositionError(
                f"could not determine eigenvalues at Laurent order {power}"
            ) from exc
        if sum(int(mult) for mult in eigenvalues.values()) != coefficient.rows:
            raise BlockDecompositionError(
                f"characteristic polynomial did not split at Laurent order {power}"
            )
        ordered = tuple(
            (value, int(mult))
            for value, mult in sorted(
                eigenvalues.items(), key=lambda item: sp.default_sort_key(item[0])
            )
        )
        rank: int | None = None
        if len(ordered) == 1:
            raw_value, multiplicity = ordered[0]
            value = sp.simplify(raw_value)
            ordered = ((value, multiplicity),)
            if multiplicity != coefficient.rows:
                raise BlockDecompositionError(
                    f"single-eigenvalue multiplicity is incomplete at Laurent order {power}"
                )
            rank = int((coefficient - value * sp.eye(coefficient.rows)).rank())
        return IrregularSpectralData(
            power=power,
            coefficient=sp.ImmutableMatrix(coefficient),
            eigenvalues=ordered,
            nilpotent_rank=rank,
        )
    return None


def generalized_eigenbasis(
    matrix: sp.MatrixBase,
) -> tuple[
    sp.ImmutableMatrix,
    tuple[int, ...],
    tuple[sp.Expr, ...],
    tuple[sp.ImmutableMatrix, ...],
]:
    """Return a generalized-eigenspace basis and exact spectral projectors."""

    matrix = sp.Matrix(matrix)
    if matrix.rows != matrix.cols:
        raise BlockDecompositionError(
            "spectral splitting requires a square coefficient"
        )
    n = matrix.rows
    try:
        eigenvalues = matrix.eigenvals()
    except SYMBOLIC_FAILURES as exc:  # pragma: no cover
        raise BlockDecompositionError(
            "could not compute exact leading eigenvalues"
        ) from exc
    if sum(int(mult) for mult in eigenvalues.values()) != n:
        raise BlockDecompositionError(
            "leading characteristic polynomial did not split completely"
        )
    ordered = sorted(eigenvalues.items(), key=lambda item: sp.default_sort_key(item[0]))
    if len(ordered) < 2:
        raise BlockDecompositionError("coefficient has no distinct spectral groups")

    columns: list[sp.Matrix] = []
    dimensions: list[int] = []
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity_expr in ordered:
        multiplicity = int(multiplicity_expr)
        generalized = (matrix - eigenvalue * sp.eye(n)) ** n
        basis = generalized.nullspace()
        if len(basis) != multiplicity:
            raise BlockDecompositionError(
                "could not construct the complete generalized eigenspace "
                f"for eigenvalue {eigenvalue!s}"
            )
        columns.extend(basis)
        dimensions.append(multiplicity)
        values.append(sp.simplify(eigenvalue))

    change = sp.Matrix.hstack(*columns)
    if change.rank() != n:
        raise BlockDecompositionError(
            "generalized eigenspaces did not form a full basis"
        )
    inverse = change.inv()
    projectors: list[sp.ImmutableMatrix] = []
    offset = 0
    for dimension in dimensions:
        selector = sp.zeros(n)
        for index in range(offset, offset + dimension):
            selector[index, index] = 1
        projector = change * selector * inverse
        projectors.append(
            sp.ImmutableMatrix(projector.applyfunc(lambda entry: sp.simplify(entry)))
        )
        offset += dimension
    return (
        sp.ImmutableMatrix(change),
        tuple(dimensions),
        tuple(values),
        tuple(projectors),
    )
