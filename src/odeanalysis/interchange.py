"""Formal ODE interchange objects for asymptotic consumers.

The module exposes mathematical results needed by asymptotic analysis without
exposing companion-system, Moser, block-diagonalization, or Levelt-reduction
implementation objects.

The central object is :class:`FormalODEData`.  A consumer can inspect completed
exponential polynomials, ramification, semisimple and nilpotent exponent data,
truncated scalar amplitudes, formal monodromy, and Stokes-sector geometry while
remaining independent of the internal differential-module implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_compare import expressions_equal
from .levelt import LeveltStructure, levelt_structure
from .operator import LinearDifferentialOperator
from .stokes import (
    StokesGeometry,
    StokesGeometryError,
    _project_ray,
    stokes_geometry,
)


@dataclass(frozen=True)
class FormalODEExponentClass:
    """One congruence class of formal exponents modulo the integers."""

    representative: sp.Expr
    exponents: tuple[sp.Expr, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class FormalODEBasisVector:
    """One scalar formal basis vector in consumer-facing form.

    ``amplitude_parameter`` omits the completed exponential factor and is
    written in ``local_parameter``.  It may contain powers and logarithms of
    that parameter.  ``local_expression`` includes the exponential factor in
    the local coordinate, while ``expression`` is mapped back to the original
    independent variable.
    """

    source_exponent: sp.Expr
    exponent: sp.Expr
    ramified_exponent: sp.Expr
    logarithmic_degree: int
    local_parameter: sp.Symbol
    amplitude_parameter: sp.Expr
    local_expression: sp.Expr
    expression: sp.Expr


@dataclass(frozen=True)
class FormalODEBlock:
    """One completed exponential block for use by :mod:`asymptotic`.

    On its uniformizing cover ``h=t**r`` the formal structure is

    ``exp(Q(t^-1)) * t**Lambda * exp(N*log(t)) * H(t)``.

    The scalar columns of the truncated ``H``-part are represented by
    ``basis_vectors``.  The matrices ``Lambda`` and ``N`` are the semisimple
    and commuting nilpotent exponent data respectively.
    """

    index: int
    point: sp.Expr
    local_coordinate: sp.Symbol
    local_parameter: sp.Symbol
    ramification_index: int
    dimension: int
    exponential_polynomial: sp.Expr
    local_exponential_polynomial: sp.Expr
    semisimple_exponent: sp.ImmutableMatrix
    nilpotent_exponent: sp.ImmutableMatrix
    formal_exponent_matrix: sp.ImmutableMatrix
    exponent_classes: tuple[FormalODEExponentClass, ...]
    basis_vectors: tuple[FormalODEBasisVector, ...]
    cover_monodromy: sp.ImmutableMatrix
    has_logarithms: bool

    @property
    def amplitudes(self) -> tuple[sp.Expr, ...]:
        """Truncated scalar amplitudes in the block uniformizer."""

        return tuple(vector.amplitude_parameter for vector in self.basis_vectors)

    @property
    def expressions(self) -> tuple[sp.Expr, ...]:
        """Truncated formal scalar solutions in the original variable."""

        return tuple(vector.expression for vector in self.basis_vectors)


@dataclass(frozen=True)
class FormalODEStokesPair:
    """Consumer-facing pairwise completed-exponential Stokes data."""

    block_indices: tuple[int, int]
    difference_local_exponential_polynomial: sp.Expr
    common_ramification: int
    leading_parameter_power: int
    leading_coefficient: sp.Expr
    exponential_order: sp.Rational
    equal_magnitude_cover_angles: tuple[sp.Expr, ...]
    phase_alignment_cover_angles: tuple[sp.Expr, ...]
    equal_magnitude_local_angles: tuple[sp.Expr, ...] = ()
    equal_magnitude_original_angles: tuple[sp.Expr, ...] = ()
    equal_magnitude_sheets: tuple[int | None, ...] = ()
    phase_alignment_local_angles: tuple[sp.Expr, ...] = ()
    phase_alignment_original_angles: tuple[sp.Expr, ...] = ()
    phase_alignment_sheets: tuple[int | None, ...] = ()
    source_branch_pairs: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class FormalODEStokesSector:
    """One open sector on the common ramified cover."""

    index: int
    start_angle: sp.Expr
    end_angle: sp.Expr
    representative_angle: sp.Expr
    width: sp.Expr
    dominance_levels: tuple[tuple[int, ...], ...]
    local_start_angle: sp.Expr | None = None
    local_end_angle: sp.Expr | None = None
    local_representative_angle: sp.Expr | None = None
    original_start_angle: sp.Expr | None = None
    original_end_angle: sp.Expr | None = None
    original_representative_angle: sp.Expr | None = None
    sheet: int | None = None


@dataclass(frozen=True)
class FormalODEStokesData:
    """Stokes geometry expressed only in terms needed by consumers."""

    common_parameter: sp.Symbol
    common_ramification: int
    pairs: tuple[FormalODEStokesPair, ...]
    equal_magnitude_cover_angles: tuple[sp.Expr, ...]
    phase_alignment_cover_angles: tuple[sp.Expr, ...]
    sector_boundaries: tuple[sp.Expr, ...]
    sectors: tuple[FormalODEStokesSector, ...]
    sector_geometry_complete: bool


@dataclass(frozen=True)
class FormalODEProvenance:
    """Reduction provenance retained without exposing internal gauge objects."""

    reduction_path: tuple[str, ...]
    structure_complete: bool
    structure_limitation: str | None
    source_block_count: int
    stokes_requested: bool
    stokes_computed: bool
    stokes_complete: bool | None


@dataclass(frozen=True)
class FormalODEData:
    """Stable interchange representation of local formal ODE asymptotics.

    This is the intended boundary object between ``odeanalysis`` and
    ``asymptotic``.  It contains mathematical output rather than discovery
    machinery.  In particular, no companion system, Moser transformation, or
    block-diagonalizing gauge appears in this schema.
    """

    point: sp.Expr
    local_coordinate: sp.Symbol
    ramification_index: int
    blocks: tuple[FormalODEBlock, ...]
    terms: int
    operator_order: int
    cover_monodromy: sp.ImmutableMatrix
    local_monodromy: sp.ImmutableMatrix | None
    stokes: FormalODEStokesData | None
    provenance: FormalODEProvenance
    complete: bool
    limitation: str | None = None

    @property
    def dimension(self) -> int:
        return sum(block.dimension for block in self.blocks)

    @property
    def has_logarithms(self) -> bool:
        return any(block.has_logarithms for block in self.blocks)

    @property
    def expressions(self) -> tuple[sp.Expr, ...]:
        return tuple(expr for block in self.blocks for expr in block.expressions)


@dataclass(frozen=True)
class FormalODEGreenOperatorData:
    """Stable scalar-operator data for certified asymptotic Green inverses.

    The object contains the differential operator and its characteristic
    polynomial. It does not depend on :mod:`asymptotic`;
    consumers can verify the characteristic-polynomial construction before
    using the data in a Green/exponential-dichotomy theorem.

    ``coefficients[k]`` multiplies the ``k``-th derivative.  A rigorous
    constant-coefficient dichotomy is available only when
    ``constant_coefficients`` is true, but variable-coefficient operators are
    represented faithfully so a downstream theorem can return ``UNKNOWN``
    without changing the problem.
    """

    variable: sp.Symbol
    point: sp.Expr
    coefficients: tuple[sp.Expr, ...]
    order: int
    characteristic_parameter: sp.Symbol
    characteristic_polynomial: sp.Expr
    constant_coefficients: bool
    leading_coefficient_nonzero: bool | None

    def verify(self) -> bool:
        """Recompute and verify the stored operator invariants exactly."""

        lam = self.characteristic_parameter
        expected = sp.expand(sum(self.coefficients[k] * lam**k for k in range(self.order + 1)))
        return (
            len(self.coefficients) == self.order + 1
            and expressions_equal(expected, self.characteristic_polynomial)
            and self.constant_coefficients
            == all(
                self.variable not in coefficient.free_symbols for coefficient in self.coefficients
            )
            and self.leading_coefficient_nonzero == _nonzero_status(self.coefficients[-1])
        )


def green_operator_data(
    operator: LinearDifferentialOperator,
    *,
    point: sp.Expr = sp.oo,
) -> FormalODEGreenOperatorData:
    """Project a scalar linear operator to the Green-certificate schema.

    This is a dependency-free handoff to asymptotic consumers.  It performs no
    spectral classification itself: that classification depends on the chosen
    asymptotic end and belongs to the Green/exponential-dichotomy theorem.
    """

    if not isinstance(operator, LinearDifferentialOperator):
        raise TypeError("operator must be a LinearDifferentialOperator")
    coefficients = tuple(sp.simplify(c) for c in operator.coefficients)
    lam = sp.Symbol("__lambda")
    characteristic = sp.expand(sum(coefficients[k] * lam**k for k in range(operator.order + 1)))
    return FormalODEGreenOperatorData(
        variable=operator.variable,
        point=sp.sympify(point),
        coefficients=coefficients,
        order=operator.order,
        characteristic_parameter=lam,
        characteristic_polynomial=characteristic,
        constant_coefficients=all(
            operator.variable not in coefficient.free_symbols for coefficient in coefficients
        ),
        leading_coefficient_nonzero=_nonzero_status(coefficients[-1]),
    )


def _nonzero_status(expression: sp.Expr) -> bool | None:
    """Return whether an expression is proved nonzero, zero, or unresolved."""

    simplified = sp.simplify(expression)
    if simplified.is_zero is True:
        return False
    if simplified.is_zero is False:
        return True
    return None


def _convert_block(structure: LeveltStructure, index: int) -> FormalODEBlock:
    block = structure.blocks[index]
    basis_block = block.basis_block
    classes = tuple(
        FormalODEExponentClass(
            representative=cls.representative,
            exponents=cls.exponents,
            indices=cls.indices,
        )
        for cls in block.exponent_classes
    )
    vectors = tuple(
        FormalODEBasisVector(
            source_exponent=vector.source_exponent,
            exponent=vector.exponent,
            ramified_exponent=vector.ramified_exponent,
            logarithmic_degree=vector.logarithmic_degree,
            local_parameter=vector.local_parameter,
            amplitude_parameter=vector.parameter_expression,
            local_expression=vector.local_expression,
            expression=vector.expression,
        )
        for vector in basis_block.basis_vectors
    )
    return FormalODEBlock(
        index=index,
        point=structure.point,
        local_coordinate=structure.local_coordinate,
        local_parameter=basis_block.local_parameter,
        ramification_index=block.ramification_index,
        dimension=block.dimension,
        exponential_polynomial=block.exponential_polynomial,
        local_exponential_polynomial=block.local_exponential_polynomial,
        semisimple_exponent=block.semisimple_exponent,
        nilpotent_exponent=block.nilpotent_exponent,
        formal_exponent_matrix=block.formal_exponent_matrix,
        exponent_classes=classes,
        basis_vectors=vectors,
        cover_monodromy=block.cover_monodromy,
        has_logarithms=block.has_logarithms,
    )


def _unique_expressions(values: tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    result: list[sp.Expr] = []
    for value in values:
        if not any(expressions_equal(value, previous) for previous in result):
            result.append(value)
    return tuple(result)


def _convert_stokes(
    geometry: StokesGeometry,
    blocks: tuple[FormalODEBlock, ...],
) -> FormalODEStokesData:
    """Collapse branch-level Stokes geometry to exponential-block indices."""

    branch_to_block: dict[int, int] = {}
    target_h = blocks[0].local_coordinate if blocks else None
    for branch_index, part in enumerate(geometry.exponential_parts):
        part_q = part.local_exponential_polynomial
        if target_h is not None and part.local_coordinate != target_h:
            part_q = part_q.xreplace({part.local_coordinate: target_h})
        matches = [
            block.index
            for block in blocks
            if expressions_equal(part_q, block.local_exponential_polynomial)
        ]
        if len(matches) != 1:
            raise StokesGeometryError(
                "could not uniquely match a completed exponential branch to an interchange block"
            )
        branch_to_block[branch_index] = matches[0]

    pair_groups: dict[tuple[int, int], list] = {}
    for pair in geometry.pairs:
        mapped = tuple(sorted(branch_to_block[i] for i in pair.branch_indices))
        if mapped[0] == mapped[1]:
            continue
        pair_groups.setdefault(mapped, []).append(pair)

    converted_pairs: list[FormalODEStokesPair] = []
    for mapped, source_pairs in sorted(pair_groups.items()):
        first = source_pairs[0]
        first_difference = first.difference_local_exponential_polynomial
        first_h = geometry.exponential_parts[first.branch_indices[0]].local_coordinate
        for other in source_pairs[1:]:
            other_difference = other.difference_local_exponential_polynomial
            other_h = geometry.exponential_parts[other.branch_indices[0]].local_coordinate
            if other_h != first_h:
                other_difference = other_difference.xreplace({other_h: first_h})
            if not expressions_equal(first_difference, other_difference) and not expressions_equal(
                first_difference, -other_difference
            ):
                raise StokesGeometryError(
                    "branch-level Stokes pairs in one exponential block pair disagree"
                )
        difference_q = first.difference_local_exponential_polynomial
        source_h = geometry.exponential_parts[first.branch_indices[0]].local_coordinate
        if target_h is not None and source_h != target_h:
            difference_q = difference_q.xreplace({source_h: target_h})
        converted_pairs.append(
            FormalODEStokesPair(
                block_indices=mapped,
                difference_local_exponential_polynomial=difference_q,
                common_ramification=first.common_ramification,
                leading_parameter_power=first.leading_parameter_power,
                leading_coefficient=first.leading_coefficient,
                exponential_order=first.exponential_order,
                equal_magnitude_cover_angles=_unique_expressions(
                    tuple(
                        ray.cover_angle
                        for pair in source_pairs
                        for ray in pair.equal_magnitude_rays
                    )
                ),
                phase_alignment_cover_angles=_unique_expressions(
                    tuple(
                        ray.cover_angle
                        for pair in source_pairs
                        for ray in pair.phase_alignment_rays
                    )
                ),
                equal_magnitude_local_angles=_unique_expressions(
                    tuple(
                        ray.local_angle
                        for pair in source_pairs
                        for ray in pair.equal_magnitude_rays
                    )
                ),
                equal_magnitude_original_angles=_unique_expressions(
                    tuple(
                        ray.original_angle
                        for pair in source_pairs
                        for ray in pair.equal_magnitude_rays
                    )
                ),
                equal_magnitude_sheets=tuple(
                    ray.sheet for pair in source_pairs for ray in pair.equal_magnitude_rays
                ),
                phase_alignment_local_angles=_unique_expressions(
                    tuple(
                        ray.local_angle
                        for pair in source_pairs
                        for ray in pair.phase_alignment_rays
                    )
                ),
                phase_alignment_original_angles=_unique_expressions(
                    tuple(
                        ray.original_angle
                        for pair in source_pairs
                        for ray in pair.phase_alignment_rays
                    )
                ),
                phase_alignment_sheets=tuple(
                    ray.sheet for pair in source_pairs for ray in pair.phase_alignment_rays
                ),
                source_branch_pairs=tuple(pair.branch_indices for pair in source_pairs),
            )
        )

    sectors: list[FormalODEStokesSector] = []
    for sector in geometry.sectors:
        levels: list[tuple[int, ...]] = []
        for level in sector.dominance_levels:
            mapped_level = tuple(dict.fromkeys(branch_to_block[i] for i in level))
            if mapped_level:
                levels.append(mapped_level)
        start_projection = _project_ray(
            pair=(-1, -1),
            kind="sector-boundary",
            cover_angle=sector.start_angle,
            ramification=geometry.common_ramification,
            point=geometry.point,
        )
        end_projection = _project_ray(
            pair=(-1, -1),
            kind="sector-boundary",
            cover_angle=sector.end_angle,
            ramification=geometry.common_ramification,
            point=geometry.point,
        )
        rep_projection = _project_ray(
            pair=(-1, -1),
            kind="sector-representative",
            cover_angle=sector.representative_angle,
            ramification=geometry.common_ramification,
            point=geometry.point,
        )
        sectors.append(
            FormalODEStokesSector(
                index=sector.index,
                start_angle=sector.start_angle,
                end_angle=sector.end_angle,
                representative_angle=sector.representative_angle,
                width=sector.width,
                dominance_levels=tuple(levels),
                local_start_angle=start_projection.local_angle,
                local_end_angle=end_projection.local_angle,
                local_representative_angle=rep_projection.local_angle,
                original_start_angle=start_projection.original_angle,
                original_end_angle=end_projection.original_angle,
                original_representative_angle=rep_projection.original_angle,
                sheet=rep_projection.sheet,
            )
        )

    return FormalODEStokesData(
        common_parameter=geometry.common_parameter,
        common_ramification=geometry.common_ramification,
        pairs=tuple(converted_pairs),
        equal_magnitude_cover_angles=_unique_expressions(
            tuple(ray.cover_angle for ray in geometry.equal_magnitude_rays)
        ),
        phase_alignment_cover_angles=_unique_expressions(
            tuple(ray.cover_angle for ray in geometry.phase_alignment_rays)
        ),
        sector_boundaries=geometry.sector_boundaries,
        sectors=tuple(sectors),
        sector_geometry_complete=geometry.sector_geometry_complete,
    )


def formal_ode_data(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 8,
    max_branches: int = 64,
    include_stokes: bool = True,
) -> FormalODEData:
    """Return the downstream-facing local formal data for a linear ODE.

    Stokes geometry is computed only when requested and when at least two
    distinct completed exponential blocks exist.  Failure to resolve sector
    geometry does not invalidate the already-certified Levelt data: in that
    case ``stokes`` is ``None`` and the limitation records the Stokes issue.
    """

    structure = levelt_structure(
        ode,
        function,
        variable,
        point=point,
        terms=terms,
        max_branches=max_branches,
    )
    blocks = tuple(_convert_block(structure, i) for i in range(len(structure.blocks)))

    limitation = structure.limitation
    stokes_data: FormalODEStokesData | None = None
    distinct_q = {sp.srepr(sp.expand(block.local_exponential_polynomial)) for block in blocks}
    if include_stokes and structure.complete and len(distinct_q) >= 2:
        try:
            geometry = stokes_geometry(
                ode,
                function,
                variable,
                point=point,
                max_branches=max_branches,
            )
            stokes_data = _convert_stokes(geometry, blocks)
        except (StokesGeometryError, NotImplementedError, ValueError) as exc:
            stokes_note = f"Stokes geometry unavailable: {exc}"
            limitation = f"{limitation}; {stokes_note}" if limitation else stokes_note

    stokes_complete = stokes_data.sector_geometry_complete if stokes_data is not None else None
    provenance = FormalODEProvenance(
        reduction_path=(
            "scalar-operator",
            "newton-puiseux",
            "formal-block-reduction",
            "levelt-normalization",
        ),
        structure_complete=structure.complete,
        structure_limitation=structure.limitation,
        source_block_count=len(structure.blocks),
        stokes_requested=include_stokes,
        stokes_computed=stokes_data is not None,
        stokes_complete=stokes_complete,
    )

    cover = sp.ImmutableMatrix(structure.cover_monodromy)
    local = structure.local_monodromy
    return FormalODEData(
        point=structure.point,
        local_coordinate=structure.local_coordinate,
        ramification_index=structure.ramification_index,
        blocks=blocks,
        terms=structure.basis.terms,
        operator_order=structure.basis.operator_order,
        cover_monodromy=cover,
        local_monodromy=(sp.ImmutableMatrix(local) if local is not None else None),
        stokes=stokes_data,
        provenance=provenance,
        complete=structure.complete,
        limitation=limitation,
    )
