"""Levelt structure for regular-singular and exponential formal blocks.

The public scalar entry point combines the exponential-block decomposition
with the logarithmic formal basis.  Each isolated exponential block is then
represented in Levelt form by a semisimple exponent matrix, a commuting
nilpotent logarithmic part, exponent classes modulo integers, and formal
monodromy on the common uniformizing cover.

For matrix systems, :func:`levelt_reduce_regular_singular` performs a full
truncated Levelt reduction.  It first removes nonresonant positive-degree
terms by the homological equations

    H -> R H - H R - n H.

Integer-difference resonances are then absorbed by a diagonal integer Levelt
transformation ``diag(t**k_i)`` that shifts every exponent in a congruence
class to a common representative.  Resonant terms thereby move into the
residue, where they become the nilpotent logarithmic part.  A final
nonresonant cleanup produces a connection with constant Levelt residue modulo
the requested truncation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

import sympy as sp

from ._formal_gauge import constant_series as _constant_series
from ._formal_gauge import formal_gauge_transform as _formal_gauge_transform
from ._moser import MoserReduction, moser_reduce
from ._symbolic_errors import SYMBOLIC_FAILURES
from .diagnostics import ReductionDiagnostic
from .formal_basis import (
    FormalBasisError,
    FormalLogarithmicBasis,
    FormalMonodromy,
    FormalSolutionBlock,
    formal_logarithmic_basis,
    formal_monodromy,
)
from .matrix_series import MatrixLaurentSeries
from .operator import LinearDifferentialOperator


class LeveltReductionError(NotImplementedError):
    """Raised when an exact Levelt structure cannot be certified."""


@dataclass(frozen=True)
class LeveltExponentClass:
    """Indices whose exponents are congruent modulo the integers."""

    representative: sp.Expr
    exponents: tuple[sp.Expr, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class LeveltResonantTerm:
    """One positive-degree term retained by the Levelt homological equation."""

    power: int
    coefficient: sp.ImmutableMatrix
    homological_rank: int
    nullity: int


@dataclass(frozen=True)
class RegularSingularLeveltReduction:
    """Truncated exact Levelt reduction of a Fuchsian system.

    ``jordan_residue`` is the residue before integer exponent shifts.
    ``levelt_residue`` is the final normalized residue after resonant terms
    have been absorbed.  The latter splits as ``semisimple_residue`` plus
    ``nilpotent_residue`` and directly determines formal cover monodromy.
    """

    original_connection: MatrixLaurentSeries
    transformed_connection: MatrixLaurentSeries
    gauge: MatrixLaurentSeries
    residue: sp.ImmutableMatrix
    jordan_basis: sp.ImmutableMatrix
    jordan_residue: sp.ImmutableMatrix
    levelt_residue: sp.ImmutableMatrix
    semisimple_residue: sp.ImmutableMatrix
    nilpotent_residue: sp.ImmutableMatrix
    exponent_classes: tuple[LeveltExponentClass, ...]
    integer_shifts: tuple[int, ...]
    integer_gauge: MatrixLaurentSeries
    absorbed_resonant_terms: tuple[LeveltResonantTerm, ...]
    resonant_terms: tuple[LeveltResonantTerm, ...]
    max_power: int
    complete: bool
    limitation: str | None = None
    diagnostics: tuple[ReductionDiagnostic, ...] = ()

    @property
    def cover_monodromy(self) -> sp.ImmutableMatrix:
        """Formal monodromy on one full turn of the uniformizing variable."""

        return sp.ImmutableMatrix((2 * sp.pi * sp.I * sp.Matrix(self.levelt_residue)).exp())

    @property
    def cover_monodromy_if_nonresonant(self) -> sp.ImmutableMatrix | None:
        """Backward-compatible alias once no positive-degree terms remain."""

        if self.resonant_terms:
            return None
        return self.cover_monodromy


@dataclass(frozen=True)
class LeveltExponentialBlock:
    """One exponential block in formal Levelt-Turrittin form.

    On its cover the block has the structural form

    ``H(t) * exp(Q(t^-1)) * t**Lambda * exp(N*log(t))``.
    """

    index: int
    exponential_polynomial: sp.Expr
    local_exponential_polynomial: sp.Expr
    ramification_index: int
    dimension: int
    semisimple_exponent: sp.ImmutableMatrix
    nilpotent_exponent: sp.ImmutableMatrix
    formal_exponent_matrix: sp.ImmutableMatrix
    exponent_classes: tuple[LeveltExponentClass, ...]
    cover_monodromy: sp.ImmutableMatrix
    has_logarithms: bool
    basis_block: FormalSolutionBlock

    @property
    def normal_form_factorization(
        self,
    ) -> tuple[sp.Expr, sp.ImmutableMatrix, sp.ImmutableMatrix]:
        return (
            self.exponential_polynomial,
            self.semisimple_exponent,
            self.nilpotent_exponent,
        )


@dataclass(frozen=True)
class LeveltStructure:
    """Full scalar formal exponential/Levelt block structure."""

    point: sp.Expr
    local_coordinate: sp.Symbol
    blocks: tuple[LeveltExponentialBlock, ...]
    basis: FormalLogarithmicBasis
    monodromy: FormalMonodromy
    ramification_index: int
    complete: bool
    limitation: str | None = None

    @property
    def dimension(self) -> int:
        return sum(block.dimension for block in self.blocks)

    @property
    def cover_monodromy(self) -> sp.Matrix:
        return self.monodromy.cover_matrix

    @property
    def local_monodromy(self) -> sp.Matrix | None:
        return self.monodromy.local_matrix


def _same_mod_integer(left: sp.Expr, right: sp.Expr) -> bool:
    diff = sp.simplify(left - right)
    return bool(diff.is_integer is True)


def _exponent_classes(
    exponents: tuple[sp.Expr, ...],
) -> tuple[LeveltExponentClass, ...]:
    groups: list[list[tuple[int, sp.Expr]]] = []
    for index, exponent in enumerate(exponents):
        for group in groups:
            if _same_mod_integer(exponent, group[0][1]):
                group.append((index, exponent))
                break
        else:
            groups.append([(index, exponent)])
    return tuple(
        LeveltExponentClass(
            representative=sp.simplify(group[0][1]),
            exponents=tuple(sp.simplify(value) for _, value in group),
            indices=tuple(index for index, _ in group),
        )
        for group in groups
    )


def _vec(matrix: sp.MatrixBase) -> sp.Matrix:
    matrix = sp.Matrix(matrix)
    return sp.Matrix(list(matrix))


def _unvec(vector: sp.MatrixBase, rows: int, cols: int) -> sp.Matrix:
    values = list(sp.Matrix(vector))
    return sp.Matrix(rows, cols, values)


def _homological_matrix(residue: sp.MatrixBase, n: int) -> sp.Matrix:
    residue = sp.Matrix(residue)
    size = residue.rows
    columns: list[sp.Matrix] = []
    for index in range(size * size):
        basis = sp.zeros(size)
        row, col = divmod(index, size)
        basis[row, col] = 1
        image = residue * basis - basis * residue - n * basis
        columns.append(_vec(image))
    return sp.Matrix.hstack(*columns)


def _image_complement(matrix: sp.MatrixBase) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return independent image-column indices and coordinate complement indices."""

    matrix = sp.Matrix(matrix)
    _, pivots = matrix.rref()
    image_pivots = tuple(int(i) for i in pivots)
    basis = [matrix[:, i] for i in image_pivots]
    current = sp.Matrix.hstack(*basis) if basis else sp.zeros(matrix.rows, 0)
    rank = current.rank()
    complement: list[int] = []
    for i in range(matrix.rows):
        e = sp.eye(matrix.rows)[:, i]
        candidate = current.row_join(e)
        new_rank = candidate.rank()
        if new_rank > rank:
            complement.append(i)
            current = candidate
            rank = new_rank
        if rank == matrix.rows:
            break
    if rank != matrix.rows:
        raise LeveltReductionError("could not construct a complement to the homological image")
    return image_pivots, tuple(complement)


def _homological_reduce_coefficient(
    residue: sp.MatrixBase,
    coefficient: sp.MatrixBase,
    n: int,
) -> tuple[sp.Matrix, sp.Matrix, int, int]:
    """Solve C + L_n(H) = K with K in a deterministic exact complement."""

    residue = sp.Matrix(residue)
    coefficient = sp.Matrix(coefficient)
    size = residue.rows
    operator = _homological_matrix(residue, n)
    image_pivots, complement_indices = _image_complement(operator)
    image_basis = [operator[:, i] for i in image_pivots]
    complement_basis = [sp.eye(size * size)[:, i] for i in complement_indices]
    system = sp.Matrix.hstack(*(image_basis + [(-v) for v in complement_basis]))
    rhs = -_vec(coefficient)
    try:
        solution = system.inv() * rhs
    except SYMBOLIC_FAILURES as exc:
        raise LeveltReductionError("could not solve the Levelt homological decomposition") from exc
    h_vec = sp.zeros(size * size, 1)
    for position, domain_index in enumerate(image_pivots):
        h_vec[domain_index] = sp.simplify(solution[position])
    k_vec = sp.zeros(size * size, 1)
    offset = len(image_pivots)
    for position, coordinate_index in enumerate(complement_indices):
        k_vec[coordinate_index] = sp.simplify(solution[offset + position])
    h = _unvec(h_vec, size, size)
    k = _unvec(k_vec, size, size)
    rank = int(operator.rank())
    return h, k, rank, size * size - rank


def _homological_normalize(
    connection: MatrixLaurentSeries,
    residue: sp.MatrixBase,
    *,
    max_power: int,
) -> tuple[MatrixLaurentSeries, MatrixLaurentSeries, tuple[LeveltResonantTerm, ...]]:
    """Remove homological-image terms and retain exact resonant complements."""

    variable = connection.variable
    current = connection
    gauge = MatrixLaurentSeries.identity(variable, connection.rows)
    resonant: list[LeveltResonantTerm] = []
    residue = sp.Matrix(residue)
    for n in range(1, max_power + 2):
        power = n - 1
        coefficient = sp.Matrix(current.coefficient(power))
        if coefficient.is_zero_matrix:
            continue
        h, _k, rank, nullity = _homological_reduce_coefficient(residue, coefficient, n)
        if not h.is_zero_matrix:
            step = MatrixLaurentSeries.from_mapping(
                variable,
                {0: sp.eye(connection.rows), n: h},
                shape=connection.shape,
            )
            current = _formal_gauge_transform(current, step, max_power=max_power)
            gauge = gauge.multiply(step, max_power=max_power + n)
        retained = sp.Matrix(current.coefficient(power))
        if not retained.is_zero_matrix:
            resonant.append(
                LeveltResonantTerm(
                    power=power,
                    coefficient=sp.ImmutableMatrix(retained),
                    homological_rank=rank,
                    nullity=nullity,
                )
            )
    return current, gauge, tuple(resonant)


def _integer_levelt_shifts(
    exponents: tuple[sp.Expr, ...],
    classes: tuple[LeveltExponentClass, ...],
) -> tuple[int, ...]:
    """Choose concrete integer shifts sending each class to its representative."""

    shifts = [0] * len(exponents)
    for cls in classes:
        representative = sp.sympify(cls.representative)
        for index in cls.indices:
            difference = sp.simplify(exponents[index] - representative)
            if difference.is_Integer:
                shifts[index] = int(difference)
                continue
            if difference.is_integer is True:
                raise LeveltReductionError(
                    "symbolic integer exponent differences are not representable "
                    "by MatrixLaurentSeries"
                )
            if difference != 0:
                raise LeveltReductionError("could not certify an integer Levelt exponent shift")
    return tuple(shifts)


def _diagonal_integer_gauge(variable: sp.Symbol, shifts: tuple[int, ...]) -> MatrixLaurentSeries:
    matrix = sp.diag(*(variable**shift for shift in shifts))
    return MatrixLaurentSeries.from_matrix(matrix, variable)


def _is_scalar_coefficient(matrix: sp.MatrixBase) -> bool:
    matrix = sp.Matrix(matrix)
    if matrix.rows != matrix.cols:
        return False
    scalar = matrix[0, 0] if matrix.rows else 0
    return all(
        sp.simplify(matrix[i, j] - (scalar if i == j else 0)) == 0
        for i in range(matrix.rows)
        for j in range(matrix.cols)
    )


def levelt_reduce_regular_singular(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
) -> RegularSingularLeveltReduction:
    """Return a truncated full Levelt reduction of a Fuchsian connection.

    Positive-degree resonances are first exposed by the usual homological
    equations.  For each exponent class modulo the integers, a diagonal
    meromorphic gauge ``diag(t**k_i)`` then shifts all exponents to the same
    representative.  A resonant coefficient in position ``(i,j)`` at
    ``t**(n-1)`` satisfies ``lambda_i-lambda_j=n``; the integer gauge changes
    its power to ``-1``, so it becomes part of the Levelt residue instead of
    remaining as an external positive-degree normal-form term.
    """

    if max_power < 0:
        raise ValueError("max_power must be nonnegative")
    if connection.rows != connection.cols:
        raise ValueError("Levelt reduction requires a square connection")
    for power, coefficient in connection.terms:
        if power < -1 and not _is_scalar_coefficient(coefficient):
            raise LeveltReductionError(
                "connection is still irregular; run Moser/exponential reduction first"
            )

    variable = connection.variable
    residue = sp.Matrix(connection.coefficient(-1))
    try:
        jordan_basis, _jordan = residue.jordan_form()
    except SYMBOLIC_FAILURES as exc:
        raise LeveltReductionError("could not construct the exact residue Jordan form") from exc
    if sp.simplify(jordan_basis.det()) == 0:
        raise LeveltReductionError("residue Jordan basis is singular")

    constant = _constant_series(variable, jordan_basis)
    jordan_connection = _formal_gauge_transform(connection, constant, max_power=max_power)
    jordan_residue = sp.Matrix(jordan_connection.coefficient(-1))
    exponents = tuple(sp.simplify(jordan_residue[i, i]) for i in range(jordan_residue.rows))
    classes = _exponent_classes(exponents)
    shifts = _integer_levelt_shifts(exponents, classes)
    spread = max(shifts, default=0) - min(shifts, default=0)

    # Terms that an integer shear can move down to residue/negative order must
    # be normalized *before* the shear.  The spread is the largest possible
    # power displacement between two matrix entries.
    pre_order = max_power + spread
    if pre_order != max_power:
        jordan_connection = _formal_gauge_transform(connection, constant, max_power=pre_order)
    pre_current, pre_gauge, absorbed = _homological_normalize(
        jordan_connection, jordan_residue, max_power=pre_order
    )

    integer_gauge = _diagonal_integer_gauge(variable, shifts)
    shifted = _formal_gauge_transform(pre_current, integer_gauge, max_power=max_power)

    # A correct Levelt integer transform may leave scalar exponential pieces
    # below -1, but no nonscalar irregular term.  In the regular-singular
    # setting even scalar terms should normally be absent; we retain the more
    # invariant nonscalar certification here.
    bad_irregular = [
        power
        for power, coefficient in shifted.terms
        if power < -1 and not _is_scalar_coefficient(coefficient)
    ]
    if bad_irregular:
        limitation = (
            "integer Levelt transformation produced an unresolved nonscalar "
            f"irregular term at power {min(bad_irregular)}"
        )
        return RegularSingularLeveltReduction(
            original_connection=connection,
            transformed_connection=shifted,
            gauge=constant.multiply(pre_gauge, max_power=pre_order + spread).multiply(
                integer_gauge, max_power=pre_order + spread
            ),
            residue=sp.ImmutableMatrix(residue),
            jordan_basis=sp.ImmutableMatrix(jordan_basis),
            jordan_residue=sp.ImmutableMatrix(jordan_residue),
            levelt_residue=sp.ImmutableMatrix(shifted.coefficient(-1)),
            semisimple_residue=sp.ImmutableMatrix(sp.zeros(connection.rows)),
            nilpotent_residue=sp.ImmutableMatrix(sp.zeros(connection.rows)),
            exponent_classes=classes,
            integer_shifts=shifts,
            integer_gauge=integer_gauge,
            absorbed_resonant_terms=absorbed,
            resonant_terms=(),
            max_power=max_power,
            complete=False,
            limitation=limitation,
            diagnostics=(
                ReductionDiagnostic(
                    stage="levelt",
                    code="irregular-term-after-shift",
                    message=limitation,
                    power=min(bad_irregular),
                ),
            ),
        )

    levelt_residue = sp.Matrix(shifted.coefficient(-1))
    post_current, post_gauge, remaining = _homological_normalize(
        shifted, levelt_residue, max_power=max_power
    )

    # After integer normalization all semisimple exponent differences inside
    # a class are zero and differences between classes are nonintegral.  Thus
    # no positive-degree Levelt resonance should remain.
    complete = not remaining
    limitation = None
    if remaining:
        limitation = (
            "positive-degree resonance remains after integer Levelt "
            "normalization; the exponent-class split is incomplete"
        )

    levelt_residue = sp.Matrix(post_current.coefficient(-1))
    normalized_exponents = tuple(
        sp.simplify(exponents[i] - shifts[i]) for i in range(len(exponents))
    )
    semisimple = sp.diag(*normalized_exponents)
    nilpotent = (levelt_residue - semisimple).applyfunc(sp.simplify)
    if not (semisimple * nilpotent - nilpotent * semisimple).is_zero_matrix:
        complete = False
        limitation = (
            "the normalized residue did not split into commuting semisimple "
            "and nilpotent Levelt parts"
        )

    gauge_order = pre_order + spread + max_power + 2
    gauge = constant.multiply(pre_gauge, max_power=gauge_order)
    gauge = gauge.multiply(integer_gauge, max_power=gauge_order)
    gauge = gauge.multiply(post_gauge, max_power=gauge_order)

    diagnostics: list[ReductionDiagnostic] = []
    for term in remaining:
        diagnostics.append(
            ReductionDiagnostic(
                stage="levelt",
                code="positive-degree-resonance",
                message="homological normalization retained a resonant coefficient",
                power=term.power,
                rank=term.homological_rank,
            )
        )
    if not complete and not diagnostics and limitation:
        diagnostics.append(
            ReductionDiagnostic(
                stage="levelt",
                code="noncommuting-residue-split",
                message=limitation,
            )
        )

    return RegularSingularLeveltReduction(
        original_connection=connection,
        transformed_connection=post_current,
        gauge=gauge,
        residue=sp.ImmutableMatrix(residue),
        jordan_basis=sp.ImmutableMatrix(jordan_basis),
        jordan_residue=sp.ImmutableMatrix(jordan_residue),
        levelt_residue=sp.ImmutableMatrix(levelt_residue),
        semisimple_residue=sp.ImmutableMatrix(semisimple),
        nilpotent_residue=sp.ImmutableMatrix(nilpotent),
        exponent_classes=_exponent_classes(normalized_exponents),
        integer_shifts=shifts,
        integer_gauge=integer_gauge,
        absorbed_resonant_terms=absorbed,
        resonant_terms=remaining,
        max_power=max_power,
        complete=complete,
        limitation=limitation,
        diagnostics=tuple(diagnostics),
    )


def reduce_to_fuchsian(
    connection: MatrixLaurentSeries,
    *,
    max_power: int,
    max_steps: int = 8,
) -> MoserReduction:
    """Public convenience wrapper for the Moser/shearing stage."""

    return moser_reduce(connection, max_power=max_power, max_steps=max_steps)


def _nilpotent_logarithm(unipotent: sp.MatrixBase) -> sp.Matrix:
    unipotent = sp.Matrix(unipotent)
    x = unipotent - sp.eye(unipotent.rows)
    result = sp.zeros(unipotent.rows)
    power = sp.eye(unipotent.rows)
    for k in range(1, unipotent.rows + 1):
        power = power * x
        if power.is_zero_matrix:
            break
        result += sp.Rational((-1) ** (k + 1), k) * power
    if not (x**unipotent.rows).is_zero_matrix:
        raise LeveltReductionError("monodromy quotient is not certifiably unipotent")
    return result.applyfunc(sp.simplify)


def _block_levelt_data(
    block: FormalSolutionBlock,
    cover_monodromy: sp.MatrixBase,
    index: int,
) -> LeveltExponentialBlock:
    exponents = tuple(sp.simplify(v.ramified_exponent) for v in block.basis_vectors)
    semisimple = sp.diag(*exponents) if exponents else sp.zeros(0)
    classes = _exponent_classes(exponents)
    cover = sp.Matrix(cover_monodromy)

    # On a Levelt class all exp(2*pi*i*lambda) coincide.  Remove the
    # semisimple factor class-by-class, then take the finite nilpotent log.
    nilpotent = sp.zeros(block.dimension)
    for cls in classes:
        indices = cls.indices
        if not indices:
            continue
        scalar = sp.exp(2 * sp.pi * sp.I * cls.representative)
        sub = cover.extract(indices, indices) / scalar
        nsub = _nilpotent_logarithm(sub) / (2 * sp.pi * sp.I)
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                nilpotent[i, j] = sp.simplify(nsub[a, b])
    formal_exponent = semisimple + nilpotent
    return LeveltExponentialBlock(
        index=index,
        exponential_polynomial=block.exponential_polynomial,
        local_exponential_polynomial=block.local_exponential_polynomial,
        ramification_index=block.ramification_index,
        dimension=block.dimension,
        semisimple_exponent=sp.ImmutableMatrix(semisimple),
        nilpotent_exponent=sp.ImmutableMatrix(nilpotent),
        formal_exponent_matrix=sp.ImmutableMatrix(formal_exponent),
        exponent_classes=classes,
        cover_monodromy=sp.ImmutableMatrix(cover),
        has_logarithms=block.has_logarithms,
        basis_block=block,
    )


def levelt_structure(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 8,
    max_branches: int = 64,
) -> LeveltStructure:
    """Return the completed scalar exponential/Levelt block structure.

    The existing formal basis constructor supplies certified logarithmic
    companions inside every isolated exponential block.  The corresponding
    cover monodromy determines the commuting nilpotent Levelt part exactly,
    including integer-difference resonances that are invisible in the residue
    alone.
    """

    try:
        basis = formal_logarithmic_basis(
            ode,
            function,
            variable,
            point=point,
            terms=terms,
            max_branches=max_branches,
        )
    except FormalBasisError as exc:
        raise LeveltReductionError(str(exc)) from exc
    if not basis.complete:
        return LeveltStructure(
            point=sp.sympify(point),
            local_coordinate=basis.local_coordinate,
            blocks=(),
            basis=basis,
            monodromy=FormalMonodromy(basis, sp.zeros(0), None),
            ramification_index=1,
            complete=False,
            limitation=basis.limitation,
        )
    monodromy = formal_monodromy(basis)
    blocks: list[LeveltExponentialBlock] = []
    offset = 0
    common = 1
    for index, block in enumerate(basis.blocks):
        stop = offset + block.dimension
        cover = monodromy.cover_matrix[offset:stop, offset:stop]
        blocks.append(_block_levelt_data(block, cover, index))
        common = lcm(common, int(block.ramification_index))
        offset = stop
    return LeveltStructure(
        point=sp.sympify(point),
        local_coordinate=basis.local_coordinate,
        blocks=tuple(blocks),
        basis=basis,
        monodromy=monodromy,
        ramification_index=common,
        complete=True,
    )
