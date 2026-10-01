"""Stokes/anti-Stokes ray geometry and exponential sector dominance.

The formal exponential factors returned by :mod:`odeanalysis.formal` have the
shape ``exp(Q_i(h))`` near a singular point, where ``h`` is the local
coordinate (``h=x-x0`` at a finite point and ``h=1/x`` at infinity).  For a
pair of branches, put ``Delta Q_ij = Q_i-Q_j``.  The most singular nonzero term
of the *completed* difference determines the tangent ray geometry:

``Re(Delta Q_ij) = 0``
    equal-magnitude rays, called Stokes rays by the convention used here;

``Im(Delta Q_ij) = 0``
    phase-alignment rays, called anti-Stokes rays by the convention used here.

The names Stokes/anti-Stokes are reversed in part of the literature, so the
invariant names ``equal_magnitude_rays`` and ``phase_alignment_rays`` are the
primary API.

Ramified branches are handled on a common uniformizing cover ``h=t**R``.  Ray
angles and sectors are therefore represented first in the ``t``-plane.  Each
ray also records its projection to the local ``h``-plane and to the original
independent-variable plane.  Sector dominance is computed on this cover,
which keeps branch labels well-defined around ramified singularities.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from math import lcm

import sympy as sp

from ._symbolic_errors import NUMERIC_CONVERSION_FAILURES
from .formal import CompleteFormalExponentialPart, complete_formal_exponential_parts
from .operator import LinearDifferentialOperator


class StokesGeometryError(NotImplementedError):
    """Raised when exact Stokes-sector geometry cannot be resolved."""


@dataclass(frozen=True)
class StokesRay:
    """One lifted ray on the common ramified cover.

    ``cover_angle`` is the argument of the common parameter ``t`` with
    ``h=t**common_ramification``.  ``local_angle`` is its projected argument in
    the local coordinate ``h``.  ``original_angle`` is the corresponding angle
    in the original independent-variable plane; at infinity this reverses the
    local angle because ``h=1/x``.
    """

    pair: tuple[int, int]
    kind: str
    cover_angle: sp.Expr
    local_angle: sp.Expr
    original_angle: sp.Expr
    sheet: int | None


@dataclass(frozen=True)
class StokesPairGeometry:
    """Pairwise geometry derived from ``Q_i-Q_j``."""

    branch_indices: tuple[int, int]
    difference_local_exponential_polynomial: sp.Expr
    common_parameter: sp.Symbol
    common_ramification: int
    difference_parameter_polynomial: sp.Expr
    leading_parameter_power: int
    leading_coefficient: sp.Expr
    exponential_order: sp.Rational
    equal_magnitude_rays: tuple[StokesRay, ...]
    phase_alignment_rays: tuple[StokesRay, ...]

    @property
    def stokes_rays(self) -> tuple[StokesRay, ...]:
        """Alias for equal-magnitude rays under this package's convention."""

        return self.equal_magnitude_rays

    @property
    def anti_stokes_rays(self) -> tuple[StokesRay, ...]:
        """Alias for phase-alignment rays under this package's convention."""

        return self.phase_alignment_rays


@dataclass(frozen=True)
class StokesSector:
    """Open sector on the common ramified cover.

    ``dominance_levels`` lists branch indices from exponentially largest to
    exponentially smallest.  A level may contain several branches when their
    completed exponential polynomials are identical, so exponential data alone
    does not separate them.
    """

    index: int
    start_angle: sp.Expr
    end_angle: sp.Expr
    representative_angle: sp.Expr
    width: sp.Expr
    dominance_levels: tuple[tuple[int, ...], ...]

    @property
    def dominance_order(self) -> tuple[int, ...]:
        return tuple(index for level in self.dominance_levels for index in level)

    @property
    def dominant_branches(self) -> tuple[int, ...]:
        return self.dominance_levels[0] if self.dominance_levels else ()

    @property
    def subdominant_branches(self) -> tuple[int, ...]:
        return self.dominance_levels[-1] if self.dominance_levels else ()


@dataclass(frozen=True)
class StokesConnectionPattern:
    """Allowed support of a Stokes connection factor at one boundary ray.

    The pattern does not invent connection constants.  It records exactly
    which exponential branch pairs become equal in magnitude at the boundary.
    Off-diagonal entries outside those pairs are forbidden; diagonal entries
    are normalized to one.  Both orientations are retained as a support envelope,
    so the symbolic matrix is not itself asserted to be a Stokes factor.
    Determining the actual orientation and constants still requires
    sectorial normalization or analytic continuation data.
    """

    boundary_angle: sp.Expr
    dimension: int
    active_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("Stokes connection dimension must be positive")
        if len(set(self.active_pairs)) != len(self.active_pairs):
            raise ValueError("Stokes connection pairs must be unique")
        for left, right in self.active_pairs:
            if left == right:
                raise ValueError(
                    "Stokes connection pairs must contain distinct branches"
                )
            if not (0 <= left < self.dimension and 0 <= right < self.dimension):
                raise ValueError("Stokes connection pair index is out of range")

    @property
    def allowed_entries(self) -> tuple[tuple[int, int], ...]:
        entries = []
        for left, right in self.active_pairs:
            entries.extend(((left, right), (right, left)))
        return tuple(entries)

    def symbolic_matrix(self, prefix: str = "s") -> sp.ImmutableMatrix:
        """Return a unit-diagonal symbolic support envelope for the boundary."""

        matrix = sp.eye(self.dimension)
        for row, col in self.allowed_entries:
            matrix[row, col] = sp.Symbol(f"{prefix}_{row}_{col}", commutative=True)
        return sp.ImmutableMatrix(matrix)

    def validate_matrix(self, matrix: sp.MatrixBase) -> None:
        """Validate dimension, unit diagonal, and Stokes-ray sparsity."""

        matrix = sp.Matrix(matrix)
        if matrix.shape != (self.dimension, self.dimension):
            raise StokesGeometryError(
                "Stokes connection matrix has the wrong dimension"
            )
        allowed = set(self.allowed_entries)
        for row in range(self.dimension):
            for col in range(self.dimension):
                entry = sp.simplify(matrix[row, col])
                if row == col:
                    if entry != 1:
                        raise StokesGeometryError(
                            "Stokes connection matrix must have unit diagonal"
                        )
                elif (row, col) not in allowed and entry != 0:
                    raise StokesGeometryError(
                        "Stokes connection matrix couples branches that do not "
                        "share the boundary ray"
                    )


@dataclass(frozen=True)
class StokesGeometry:
    """Completed pairwise Stokes geometry and sector dominance data."""

    point: sp.Expr
    exponential_parts: tuple[CompleteFormalExponentialPart, ...]
    common_parameter: sp.Symbol
    common_ramification: int
    pairs: tuple[StokesPairGeometry, ...]
    equal_magnitude_rays: tuple[StokesRay, ...]
    phase_alignment_rays: tuple[StokesRay, ...]
    sector_boundaries: tuple[sp.Expr, ...]
    sectors: tuple[StokesSector, ...]
    sector_geometry_complete: bool

    @property
    def stokes_rays(self) -> tuple[StokesRay, ...]:
        """Alias for equal-magnitude rays under this package's convention."""

        return self.equal_magnitude_rays

    @property
    def anti_stokes_rays(self) -> tuple[StokesRay, ...]:
        """Alias for phase-alignment rays under this package's convention."""

        return self.phase_alignment_rays

    def validate(self) -> None:
        """Validate exact cover projection, sector partition, and dominance invariants.

        The check is structural and exact: it does not numerically guess symbolic
        angle orderings.  A :class:`StokesGeometryError` identifies inconsistent
        geometry rather than allowing malformed interchange data downstream.
        """

        if self.common_ramification < 1:
            raise StokesGeometryError("common ramification must be positive")
        branch_ids = set(range(len(self.exponential_parts)))
        pair_equal_rays = tuple(
            ray for pair in self.pairs for ray in pair.equal_magnitude_rays
        )
        pair_phase_rays = tuple(
            ray for pair in self.pairs for ray in pair.phase_alignment_rays
        )
        if set(pair_equal_rays) != set(self.equal_magnitude_rays):
            raise StokesGeometryError(
                "global equal-magnitude rays disagree with pairwise geometry"
            )
        if set(pair_phase_rays) != set(self.phase_alignment_rays):
            raise StokesGeometryError(
                "global phase-alignment rays disagree with pairwise geometry"
            )
        for pair in self.pairs:
            expected_pair = pair.branch_indices
            if pair.common_ramification != self.common_ramification:
                raise StokesGeometryError(
                    "pairwise and global Stokes ramifications disagree"
                )
            if len(set(expected_pair)) != 2 or any(
                index not in branch_ids for index in expected_pair
            ):
                raise StokesGeometryError(
                    "Stokes pair contains an invalid branch index"
                )
            for ray in (*pair.equal_magnitude_rays, *pair.phase_alignment_rays):
                if ray.pair != expected_pair:
                    raise StokesGeometryError(
                        "Stokes ray is attached to the wrong branch pair"
                    )
                expected_local = _normalize_angle(
                    self.common_ramification * ray.cover_angle
                )
                if sp.simplify(_normalize_angle(ray.local_angle) - expected_local) != 0:
                    raise StokesGeometryError(
                        "Stokes ray has an inconsistent local projection"
                    )
                expected_original = (
                    _normalize_angle(-expected_local)
                    if sp.sympify(self.point) == sp.oo
                    else expected_local
                )
                if (
                    sp.simplify(
                        _normalize_angle(ray.original_angle) - expected_original
                    )
                    != 0
                ):
                    raise StokesGeometryError(
                        "Stokes ray has an inconsistent original-plane projection"
                    )
        if self.sectors:
            if len(self.sector_boundaries) != len(self.sectors):
                raise StokesGeometryError(
                    "Stokes boundary and sector counts are inconsistent"
                )
            total_width = sp.simplify(
                sum((sector.width for sector in self.sectors), sp.S.Zero)
            )
            if sp.simplify(total_width - _TWO_PI) != 0:
                raise StokesGeometryError(
                    "Stokes sectors do not partition one full cover turn"
                )
            for index, sector in enumerate(self.sectors):
                start = self.sector_boundaries[index]
                end = self.sector_boundaries[(index + 1) % len(self.sectors)]
                expected_width = sp.simplify(_normalize_angle(end - start))
                if expected_width == 0:
                    expected_width = _TWO_PI
                if sector.index != index:
                    raise StokesGeometryError(
                        "Stokes sector indices are not contiguous"
                    )
                if sp.simplify(_normalize_angle(sector.start_angle) - start) != 0:
                    raise StokesGeometryError(
                        "Stokes sector has the wrong start boundary"
                    )
                if sp.simplify(sector.width - expected_width) != 0:
                    raise StokesGeometryError(
                        "Stokes sector width disagrees with its boundaries"
                    )
                rep_offset = sp.simplify(
                    _normalize_angle(sector.representative_angle - start)
                )
                if (
                    rep_offset.is_positive is not True
                    or sp.simplify(sector.width - rep_offset).is_positive is not True
                ):
                    raise StokesGeometryError(
                        "Stokes representative angle must lie inside its sector"
                    )
                order = sector.dominance_order
                if len(order) != len(set(order)) or set(order) != branch_ids:
                    raise StokesGeometryError(
                        "sector dominance levels do not partition the branches"
                    )
                width = sp.simplify(sector.width)
                if width.is_positive is not True:
                    raise StokesGeometryError(
                        "Stokes sector width must be provably positive"
                    )


_TWO_PI = 2 * sp.pi


def _normalize_angle(angle: sp.Expr) -> sp.Expr:
    angle = sp.simplify(angle)
    # Exact rational multiples of pi are common and simplify more reliably by
    # reducing the coefficient rather than leaving an unevaluated Mod.
    quotient = sp.simplify(angle / sp.pi)
    if quotient.is_Rational:
        numerator = int(quotient.p) % (2 * int(quotient.q))
        return sp.Rational(numerator, int(quotient.q)) * sp.pi
    return sp.simplify(sp.Mod(angle, _TWO_PI))


def _angle_float(angle: sp.Expr) -> float | None:
    if angle.free_symbols:
        return None
    try:
        value = complex(sp.N(angle, 40))
    except NUMERIC_CONVERSION_FAILURES:
        return None
    if abs(value.imag) > 1e-25:
        return None
    return float(value.real % float(2 * sp.pi.evalf(40)))


def _formal_terms_in_common_parameter(
    expression: sp.Expr,
    local_coordinate: sp.Symbol,
    parameter: sp.Symbol,
    ramification: int,
) -> sp.Expr:
    """Formally replace ``h**q`` by ``t**(R*q)`` for rational q.

    This treats the completed exponential polynomial as formal
    Puiseux data rather than asking SymPy to simplify ``(t**R)**q`` using
    principal-branch identities.
    """

    expression = sp.expand(expression)
    if expression == 0:
        return sp.S.Zero
    result = sp.S.Zero
    for term in sp.Add.make_args(expression):
        powers = term.as_powers_dict()
        exponent = sp.sympify(powers.get(local_coordinate, sp.S.Zero))
        if not exponent.is_Rational:
            raise StokesGeometryError(
                "completed exponential polynomial contains a non-rational local power"
            )
        parameter_power = sp.simplify(ramification * exponent)
        if not parameter_power.is_Integer:
            raise StokesGeometryError(
                "common ramification did not integralize a completed exponential power"
            )
        coefficient = sp.simplify(term / local_coordinate**exponent)
        if coefficient.has(local_coordinate):
            raise StokesGeometryError(
                "could not separate a completed exponential term into coefficient and local power"
            )
        result += coefficient * parameter ** int(parameter_power)
    return sp.expand(result)


def _leading_negative_term(
    expression: sp.Expr, parameter: sp.Symbol
) -> tuple[int, sp.Expr]:
    expression = sp.expand(expression)
    terms: dict[int, sp.Expr] = {}
    for term in sp.Add.make_args(expression):
        powers = term.as_powers_dict()
        exponent = sp.sympify(powers.get(parameter, sp.S.Zero))
        if not exponent.is_Integer:
            raise StokesGeometryError(
                "uniformized exponential difference has a nonintegral power"
            )
        exponent_int = int(exponent)
        coefficient = sp.simplify(term / parameter**exponent_int)
        terms[exponent_int] = sp.simplify(
            terms.get(exponent_int, sp.S.Zero) + coefficient
        )
    nonzero = [
        (power, coeff) for power, coeff in terms.items() if sp.simplify(coeff) != 0
    ]
    if not nonzero:
        raise ValueError("zero exponential difference has no Stokes rays")
    power, coefficient = min(nonzero, key=lambda item: item[0])
    if power >= 0:
        raise StokesGeometryError(
            "exponential difference has no negative-power term and does not define irregular Stokes rays"
        )
    return power, sp.simplify(coefficient)


def _project_ray(
    *,
    pair: tuple[int, int],
    kind: str,
    cover_angle: sp.Expr,
    ramification: int,
    point: sp.Expr,
) -> StokesRay:
    cover_angle = _normalize_angle(cover_angle)
    local_unwrapped = sp.simplify(ramification * cover_angle)
    local_angle = _normalize_angle(local_unwrapped)
    original_angle = (
        _normalize_angle(-local_angle) if sp.sympify(point) == sp.oo else local_angle
    )

    sheet: int | None = None
    q = sp.simplify(local_unwrapped / _TWO_PI)
    if not q.free_symbols:
        try:
            sheet = int(sp.floor(q)) % ramification
        except NUMERIC_CONVERSION_FAILURES:
            sheet = None
    return StokesRay(
        pair=pair,
        kind=kind,
        cover_angle=cover_angle,
        local_angle=local_angle,
        original_angle=original_angle,
        sheet=sheet,
    )


def _pair_rays(
    *,
    pair: tuple[int, int],
    leading_coefficient: sp.Expr,
    positive_order: int,
    ramification: int,
    point: sp.Expr,
    phase_alignment: bool,
) -> tuple[StokesRay, ...]:
    arg_c = sp.arg(leading_coefficient)
    kind = "phase_alignment" if phase_alignment else "equal_magnitude"
    offset = sp.S.Zero if phase_alignment else sp.pi / 2
    rays: list[StokesRay] = []
    # arg(c) - m*theta = offset + k*pi.  Taking k in 0..2m-1 gives every
    # lifted ray exactly once modulo 2*pi.
    for k in range(2 * positive_order):
        theta = (arg_c - offset - k * sp.pi) / positive_order
        rays.append(
            _project_ray(
                pair=pair,
                kind=kind,
                cover_angle=theta,
                ramification=ramification,
                point=point,
            )
        )
    # Normalize/deduplicate exact coincidences; repeated algebraic forms can
    # otherwise arise after SymPy simplifies arg(c).
    unique: dict[str, StokesRay] = {}
    for ray in rays:
        unique[sp.srepr(ray.cover_angle)] = ray
    values = list(unique.values())
    sortable = [(_angle_float(ray.cover_angle), ray) for ray in values]
    if all(value is not None for value, _ in sortable):
        values = [ray for _, ray in sorted(sortable, key=lambda item: item[0])]
    return tuple(values)


def _real_leading_sign(pair: StokesPairGeometry, angle: sp.Expr) -> int | None:
    m = -pair.leading_parameter_power
    value = sp.simplify(sp.re(pair.leading_coefficient * sp.exp(-sp.I * m * angle)))
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    if value.is_zero:
        return 0
    if not value.free_symbols:
        try:
            numeric = complex(sp.N(value, 50))
        except NUMERIC_CONVERSION_FAILURES:
            return None
        if abs(numeric.imag) > 1e-30:
            return None
        tolerance = 1e-25
        if numeric.real > tolerance:
            return 1
        if numeric.real < -tolerance:
            return -1
        return 0
    return None


def _dominance_levels(
    branch_count: int,
    pairs: tuple[StokesPairGeometry, ...],
    angle: sp.Expr,
) -> tuple[tuple[int, ...], ...] | None:
    pair_lookup = {pair.branch_indices: pair for pair in pairs}

    def compare(i: int, j: int) -> int:
        if i == j:
            return 0
        pair_key = (i, j) if i < j else (j, i)
        pair = pair_lookup.get(pair_key)
        if pair is None:
            # No pair means the completed exponential polynomials are equal.
            return 0
        sign = _real_leading_sign(pair, angle)
        if sign is None:
            raise StokesGeometryError("could not determine symbolic sector dominance")
        if i > j:
            sign = -sign
        # cmp convention: negative means i comes first.  Positive Delta Q_ij
        # means branch i is exponentially larger.
        return -sign

    indices = list(range(branch_count))
    try:
        ordered = sorted(indices, key=cmp_to_key(compare))
    except StokesGeometryError:
        return None

    levels: list[list[int]] = []
    try:
        for index in ordered:
            if not levels:
                levels.append([index])
                continue
            if compare(levels[-1][0], index) == 0:
                levels[-1].append(index)
            else:
                levels.append([index])
    except StokesGeometryError:
        return None
    return tuple(tuple(level) for level in levels)


def stokes_connection_patterns(
    geometry: StokesGeometry,
) -> tuple[StokesConnectionPattern, ...]:
    """Return formal connection-matrix support patterns for sector boundaries.

    One pattern is produced for each distinct equal-magnitude boundary on the
    common cover.  Multiple exponential pairs may be active on the same ray.
    The output describes support only; it leaves Stokes constants
    symbolic because formal local data does not determine them.
    """

    geometry.validate()
    patterns: list[StokesConnectionPattern] = []
    for boundary in geometry.sector_boundaries:
        active: list[tuple[int, int]] = []
        for ray in geometry.equal_magnitude_rays:
            same_boundary = (
                sp.simplify(
                    _normalize_angle(ray.cover_angle) - _normalize_angle(boundary)
                )
                == 0
            )
            if same_boundary and ray.pair not in active:
                active.append(ray.pair)
        if active:
            patterns.append(
                StokesConnectionPattern(
                    boundary_angle=_normalize_angle(boundary),
                    dimension=len(geometry.exponential_parts),
                    active_pairs=tuple(sorted(active)),
                )
            )
    return tuple(patterns)


def stokes_geometry_from_exponential_parts(
    exponential_parts: tuple[CompleteFormalExponentialPart, ...]
    | list[CompleteFormalExponentialPart],
    *,
    point: sp.Expr | None = None,
) -> StokesGeometry:
    """Build Stokes geometry from completed formal exponential parts.

    Pairwise differences use the *completed* exponential polynomials.  Thus if
    the highest terms of two branches cancel, the first surviving lower term
    correctly determines their Stokes rays.

    Sector boundaries are equal-magnitude rays on the common ramified cover.
    If symbolic parameters prevent those angles from being ordered or their
    dominance signs from being determined, pairwise ray formulas are still
    returned but ``sector_geometry_complete`` is false and ``sectors`` is
    empty.
    """

    parts = tuple(exponential_parts)
    if len(parts) < 2:
        raise ValueError(
            "Stokes geometry requires at least two formal exponential branches"
        )

    inferred_point = parts[0].point
    if point is None:
        point = inferred_point
    point = sp.sympify(point)
    if any(sp.simplify(part.point - point) != 0 for part in parts if point != sp.oo):
        raise ValueError(
            "all formal exponential parts must belong to the same singular point"
        )
    if point == sp.oo and any(part.point != sp.oo for part in parts):
        raise ValueError(
            "all formal exponential parts must belong to the same singular point"
        )

    h = parts[0].local_coordinate
    if any(part.local_coordinate != h for part in parts):
        raise ValueError("formal exponential parts use different local coordinates")

    common_ramification = 1
    for part in parts:
        common_ramification = lcm(common_ramification, int(part.ramification_index))
    parameter = sp.Symbol("_stokes_t", positive=True)

    parameter_q = tuple(
        _formal_terms_in_common_parameter(
            part.local_exponential_polynomial,
            h,
            parameter,
            common_ramification,
        )
        for part in parts
    )

    pair_geometries: list[StokesPairGeometry] = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            local_difference = sp.expand(
                parts[i].local_exponential_polynomial
                - parts[j].local_exponential_polynomial
            )
            parameter_difference = sp.expand(parameter_q[i] - parameter_q[j])
            if sp.simplify(parameter_difference) == 0:
                # Exponentially equivalent branches have no Stokes rays at this
                # level.  Algebraic/logarithmic data may still distinguish them.
                continue
            power, coefficient = _leading_negative_term(parameter_difference, parameter)
            m = -power
            pair = (i, j)
            equal_rays = _pair_rays(
                pair=pair,
                leading_coefficient=coefficient,
                positive_order=m,
                ramification=common_ramification,
                point=point,
                phase_alignment=False,
            )
            phase_rays = _pair_rays(
                pair=pair,
                leading_coefficient=coefficient,
                positive_order=m,
                ramification=common_ramification,
                point=point,
                phase_alignment=True,
            )
            pair_geometries.append(
                StokesPairGeometry(
                    branch_indices=pair,
                    difference_local_exponential_polynomial=local_difference,
                    common_parameter=parameter,
                    common_ramification=common_ramification,
                    difference_parameter_polynomial=parameter_difference,
                    leading_parameter_power=power,
                    leading_coefficient=coefficient,
                    exponential_order=sp.Rational(m, common_ramification),
                    equal_magnitude_rays=equal_rays,
                    phase_alignment_rays=phase_rays,
                )
            )

    pairs = tuple(pair_geometries)
    equal_rays = tuple(ray for pair in pairs for ray in pair.equal_magnitude_rays)
    phase_rays = tuple(ray for pair in pairs for ray in pair.phase_alignment_rays)

    # Sector boundaries are the union of distinct lifted equal-magnitude rays.
    by_repr: dict[str, sp.Expr] = {}
    for ray in equal_rays:
        by_repr[sp.srepr(ray.cover_angle)] = ray.cover_angle
    boundary_values = list(by_repr.values())
    numeric_boundaries = [(_angle_float(angle), angle) for angle in boundary_values]
    sector_geometry_complete = bool(boundary_values) and all(
        value is not None for value, _ in numeric_boundaries
    )

    sectors: list[StokesSector] = []
    boundaries: tuple[sp.Expr, ...]
    if sector_geometry_complete:
        ordered = [
            angle for _, angle in sorted(numeric_boundaries, key=lambda item: item[0])
        ]
        boundaries = tuple(ordered)
        for index, start in enumerate(ordered):
            if index + 1 < len(ordered):
                end = ordered[index + 1]
            else:
                end = sp.simplify(ordered[0] + _TWO_PI)
            representative = sp.simplify((start + end) / 2)
            levels = _dominance_levels(len(parts), pairs, representative)
            if levels is None:
                sector_geometry_complete = False
                sectors = []
                break
            sectors.append(
                StokesSector(
                    index=index,
                    start_angle=start,
                    end_angle=end,
                    representative_angle=_normalize_angle(representative),
                    width=sp.simplify(end - start),
                    dominance_levels=levels,
                )
            )
    else:
        boundaries = ()

    if not sector_geometry_complete:
        sectors = []

    return StokesGeometry(
        point=point,
        exponential_parts=parts,
        common_parameter=parameter,
        common_ramification=common_ramification,
        pairs=pairs,
        equal_magnitude_rays=equal_rays,
        phase_alignment_rays=phase_rays,
        sector_boundaries=boundaries,
        sectors=tuple(sectors),
        sector_geometry_complete=sector_geometry_complete,
    )


def stokes_geometry(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    max_branches: int = 64,
) -> StokesGeometry:
    """Compute completed Stokes geometry and sector dominance at ``point``."""

    parts = complete_formal_exponential_parts(
        ode,
        function,
        variable,
        point=point,
        max_branches=max_branches,
    )
    return stokes_geometry_from_exponential_parts(parts, point=point)
