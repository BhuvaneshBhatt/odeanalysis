"""Logarithmic formal bases and formal monodromy.

The regular-singular construction uses a parameterized Frobenius family.
If ``r`` is a free indicial parameter, the coefficients ``a_m(r)`` are
chosen so that all positive-order Frobenius equations vanish.  At an
indicial root ``r0`` the coefficient family can have poles when another root
is resonant.  Multiplying by the minimal power ``(r-r0)**p`` regularizes the
family; derivatives of orders ``p, ..., p+m-1`` at a root of multiplicity
``m`` give the logarithmic companions.  This treats repeated roots and
integer-difference resonance in one construction.

For irregular problems, resolved simple exponential branches are exposed as
one-dimensional formal blocks.  Repeated completed exponential factors are
isolated through the first-order formal block decomposition, scalarized by a
cyclic physical output, and then passed through the same logarithmic Frobenius
construction after their common exponential has been removed.

Formal monodromy is computed from the resulting truncated formal basis.  On a
ramified cover ``h=t**r`` the always-defined ``cover_matrix`` corresponds to a
full turn in ``t`` (hence ``r`` turns in ``h``).  ``local_matrix`` is also
reported when every block is unramified; physical one-turn monodromy of a
ramified problem additionally permutes cover sheets and is not
guessed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, lcm

import sympy as sp

from ._local import pole_order as _shared_pole_order
from ._power_simplify import analytic_powsimp
from ._symbolic_compare import expressions_equal
from ._symbolic_errors import SYMBOLIC_FAILURES
from .formal import (
    CompleteFormalExponentialPart,
    complete_formal_exponential_parts,
    formal_amplitude_series,
)
from .frobenius import (
    _indicial_polynomial,
    _regularized_coefficients,
    _root_data,
    _taylor_coeff,
)
from .newton import (
    LocalizedOperator,
    localize_operator,
)
from .operator import LinearDifferentialOperator
from .singularities import ODESingularityKind, classify_ode_point


class FormalBasisError(NotImplementedError):
    """Raised when a complete logarithmic formal basis cannot be certified."""


@dataclass(frozen=True)
class LogarithmicBasisVector:
    """One vector in a truncated logarithmic formal basis.

    ``parameter_expression`` omits the exponential polynomial and is written
    in the block uniformizer.  ``local_expression`` includes the exponential
    factor and is written in the local coordinate ``h``.
    """

    source_exponent: sp.Expr
    exponent: sp.Expr
    ramified_exponent: sp.Expr
    root_multiplicity: int
    pole_order: int
    derivative_order: int
    logarithmic_degree: int
    local_parameter: sp.Symbol
    parameter_expression: sp.Expr
    local_expression: sp.Expr
    expression: sp.Expr


@dataclass(frozen=True)
class FormalSolutionBlock:
    """A formal block sharing one exponential polynomial."""

    local_exponential_polynomial: sp.Expr
    exponential_polynomial: sp.Expr
    ramification_index: int
    local_parameter: sp.Symbol
    basis_vectors: tuple[LogarithmicBasisVector, ...]

    @property
    def dimension(self) -> int:
        return len(self.basis_vectors)

    @property
    def has_logarithms(self) -> bool:
        return any(vector.logarithmic_degree > 0 for vector in self.basis_vectors)


@dataclass(frozen=True)
class FormalLogarithmicBasis:
    """A block-decomposed truncated formal basis at a local point."""

    point: sp.Expr
    local_coordinate: sp.Symbol
    blocks: tuple[FormalSolutionBlock, ...]
    terms: int
    operator_order: int
    complete: bool = True
    limitation: str | None = None

    @property
    def vectors(self) -> tuple[LogarithmicBasisVector, ...]:
        return tuple(vector for block in self.blocks for vector in block.basis_vectors)

    @property
    def dimension(self) -> int:
        return len(self.vectors)

    @property
    def has_logarithms(self) -> bool:
        return any(block.has_logarithms for block in self.blocks)


@dataclass(frozen=True)
class FormalMonodromy:
    """Formal monodromy in the ordering of ``basis.vectors``.

    Matrix columns are images of basis vectors under positive analytic
    continuation.  ``cover_matrix`` is block diagonal and corresponds to one
    full turn of each block uniformizer.  When all blocks are unramified this
    is also the one-turn local monodromy and is returned as ``local_matrix``.
    """

    basis: FormalLogarithmicBasis
    cover_matrix: sp.Matrix
    local_matrix: sp.Matrix | None

    @property
    def matrix(self) -> sp.Matrix:
        """Prefer physical local monodromy when available, otherwise the cover matrix."""

        return self.local_matrix if self.local_matrix is not None else self.cover_matrix

    @property
    def eigenvalues(self) -> tuple[sp.Expr, ...]:
        values: list[sp.Expr] = []
        for value, multiplicity in self.matrix.eigenvals().items():
            values.extend([sp.simplify(value)] * int(multiplicity))
        return tuple(sorted(values, key=sp.default_sort_key))


@dataclass(frozen=True)
class _ParameterizedFrobeniusFamily:
    operator: LinearDifferentialOperator
    exponent_variable: sp.Symbol
    indicial_polynomial: sp.Expr
    roots: tuple[tuple[sp.Expr, int], ...]
    coefficients: tuple[sp.Expr, ...]


def _parameterized_frobenius_family(
    operator: LinearDifferentialOperator,
    *,
    terms: int,
) -> _ParameterizedFrobeniusFamily:
    if terms < 1:
        raise ValueError("terms must be at least one")
    if not operator.is_homogeneous:
        raise ValueError("formal basis construction requires a homogeneous operator")

    kind = classify_ode_point(operator, point=0).kind
    if kind not in (ODESingularityKind.ORDINARY, ODESingularityKind.REGULAR):
        raise FormalBasisError(
            "parameterized Frobenius construction requires an ordinary or regular singular operator"
        )

    x = operator.variable
    r = sp.Dummy("r")
    b = _regularized_coefficients(operator, sp.S.Zero)
    indicial = _indicial_polynomial(b, x, sp.S.Zero, r)
    roots = _root_data(indicial, r)
    if sum(int(mult) for _, mult in roots) != operator.order:
        raise FormalBasisError("could not resolve the complete indicial root multiset")

    bcoeff = [
        tuple(_taylor_coeff(bj, x, sp.S.Zero, q) for q in range(terms)) for bj in b
    ]
    coefficients: list[sp.Expr] = [sp.S.One]
    for m in range(1, terms):
        numerator = sp.S.Zero
        for q in range(1, m + 1):
            inner = sum(bcoeff[j][q] * sp.ff(r + m - q, j) for j in range(len(b)))
            numerator += coefficients[m - q] * inner
        denominator = sp.simplify(indicial.subs(r, r + m))
        coefficients.append(sp.cancel(-numerator / denominator))

    return _ParameterizedFrobeniusFamily(
        operator=operator,
        exponent_variable=r,
        indicial_polynomial=indicial,
        roots=roots,
        coefficients=tuple(coefficients),
    )


def _pole_order_at(expr: sp.Expr, variable: sp.Symbol, point: sp.Expr) -> int:
    """Return a certified meromorphic pole order using shared local valuation logic."""
    order = _shared_pole_order(expr, variable, point)
    if order is None:
        raise ValueError(f"could not determine pole order at {point}")
    return order


def _logarithmic_degree(expr: sp.Expr, variable: sp.Symbol) -> int:
    logx = sp.log(variable)
    try:
        poly = sp.Poly(sp.expand(expr), logx)
    except sp.PolynomialError:
        return 0
    return max(0, int(poly.degree()))


def _regularized_family_derivative(
    family: _ParameterizedFrobeniusFamily,
    root: sp.Expr,
    pole_order: int,
    derivative_order: int,
) -> sp.Expr:
    x = family.operator.variable
    r = family.exponent_variable
    result = sp.S.Zero
    for m, coefficient in enumerate(family.coefficients):
        term = (r - root) ** pole_order * coefficient * x ** (r + m)
        differentiated = sp.diff(term, r, derivative_order) / factorial(
            derivative_order
        )
        try:
            value = sp.limit(differentiated, r, root)
        except SYMBOLIC_FAILURES as exc:
            raise FormalBasisError(
                f"could not regularize Frobenius family at indicial root {root!s}"
            ) from exc
        result += value
    return sp.expand(sp.simplify(result))


def _basis_for_operator_at_zero(
    operator: LinearDifferentialOperator,
    *,
    terms: int,
) -> tuple[LogarithmicBasisVector, ...]:
    family = _parameterized_frobenius_family(operator, terms=terms)
    x = operator.variable
    vectors: list[LogarithmicBasisVector] = []
    for root, multiplicity in family.roots:
        pole_order = max(
            (
                _pole_order_at(c, family.exponent_variable, root)
                for c in family.coefficients
            ),
            default=0,
        )
        for offset in range(int(multiplicity)):
            derivative_order = pole_order + offset
            expression = _regularized_family_derivative(
                family,
                root,
                pole_order,
                derivative_order,
            )
            vectors.append(
                LogarithmicBasisVector(
                    source_exponent=sp.simplify(root),
                    exponent=sp.simplify(root),
                    ramified_exponent=sp.simplify(root),
                    root_multiplicity=int(multiplicity),
                    pole_order=pole_order,
                    derivative_order=derivative_order,
                    logarithmic_degree=_logarithmic_degree(expression, x),
                    local_parameter=x,
                    parameter_expression=expression,
                    local_expression=expression,
                    expression=expression,
                )
            )
    return tuple(vectors)


def logarithmic_frobenius_basis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 8,
) -> FormalLogarithmicBasis:
    """Return a logarithmic Frobenius basis at an ordinary/regular singular point.

    Repeated indicial roots and integer-difference resonances are handled by
    regularizing a parameterized Frobenius family and differentiating with
    respect to its exponent.  This directly produces powers of ``log(h)``
    instead of merely flagging that logarithms may be required.
    """

    localized = localize_operator(ode, function, variable, point=point)
    vectors0 = _basis_for_operator_at_zero(localized.operator, terms=terms)
    h = localized.local_variable
    vectors = tuple(
        LogarithmicBasisVector(
            source_exponent=v.source_exponent,
            exponent=v.exponent,
            ramified_exponent=v.ramified_exponent,
            root_multiplicity=v.root_multiplicity,
            pole_order=v.pole_order,
            derivative_order=v.derivative_order,
            logarithmic_degree=v.logarithmic_degree,
            local_parameter=h,
            parameter_expression=v.parameter_expression,
            local_expression=v.local_expression,
            expression=localized.to_original(v.local_expression),
        )
        for v in vectors0
    )
    block = FormalSolutionBlock(
        local_exponential_polynomial=sp.S.Zero,
        exponential_polynomial=sp.S.Zero,
        ramification_index=1,
        local_parameter=h,
        basis_vectors=vectors,
    )
    return FormalLogarithmicBasis(
        point=sp.sympify(point),
        local_coordinate=h,
        blocks=(block,),
        terms=terms,
        operator_order=localized.operator.order,
        complete=len(vectors) == localized.operator.order,
    )


def _group_completed_parts(
    parts: tuple[CompleteFormalExponentialPart, ...],
) -> tuple[tuple[CompleteFormalExponentialPart, ...], ...]:
    groups: list[list[CompleteFormalExponentialPart]] = []
    for part in parts:
        for group in groups:
            if expressions_equal(
                part.local_exponential_polynomial,
                group[0].local_exponential_polynomial,
            ):
                group.append(part)
                break
        else:
            groups.append([part])
    return tuple(tuple(group) for group in groups)


def _common_ramification(parts: tuple[CompleteFormalExponentialPart, ...]) -> int:
    result = 1
    for part in parts:
        result = lcm(result, int(part.ramification_index))
    return result


def _q_conjugated_cover_operator(
    localized: LocalizedOperator,
    local_q: sp.Expr,
    ramification: int,
) -> tuple[LinearDifferentialOperator, sp.Symbol, sp.Expr]:
    h = localized.local_variable
    t = sp.Dummy("t", positive=True)
    q_t = analytic_powsimp(sp.expand(local_q.subs(h, t**ramification)))
    coefficients = tuple(
        sp.cancel(sp.together(c.subs(h, t**ramification)))
        for c in localized.operator.coefficients
    )
    amplitude = sp.Function("_V")
    v = amplitude(t)
    factor = sp.exp(q_t)

    def deriv(expr: sp.Expr) -> sp.Expr:
        return sp.diff(expr, t) / (ramification * t ** (ramification - 1))

    derivatives = [factor * v]
    for _ in range(localized.operator.order):
        derivatives.append(sp.expand(deriv(derivatives[-1])))
    expression = sp.S.Zero
    for j, coefficient in enumerate(coefficients):
        expression += coefficient * derivatives[j] / factor
    expression = sp.cancel(sp.together(sp.expand(expression)))
    try:
        operator = LinearDifferentialOperator.from_ode(expression, amplitude, t)
    except SYMBOLIC_FAILURES as exc:
        raise FormalBasisError(
            "could not construct the exponential-conjugated cover operator"
        ) from exc
    return operator, t, q_t


def _conjugate_scalar_operator_by_exponential(
    operator: LinearDifferentialOperator,
    exponent: sp.Expr,
) -> LinearDifferentialOperator:
    """Return ``exp(-Q) L exp(Q)`` as a scalar differential operator."""

    variable = operator.variable
    amplitude = sp.Function("_V")
    v = amplitude(variable)
    factor = sp.exp(sp.sympify(exponent))
    expression = sp.S.Zero
    for order, coefficient in enumerate(operator.coefficients):
        expression += coefficient * sp.diff(factor * v, variable, order) / factor
    expression = sp.cancel(sp.together(sp.expand(expression)))
    try:
        return LinearDifferentialOperator.from_ode(expression, amplitude, variable)
    except SYMBOLIC_FAILURES as exc:
        raise FormalBasisError(
            "could not construct the exponential-conjugated isolated block operator"
        ) from exc


def _cover_to_local(
    expr: sp.Expr, t: sp.Symbol, h: sp.Symbol, ramification: int
) -> sp.Expr:
    result = sp.expand(expr).subs(sp.log(t), sp.log(h) / ramification)
    result = result.subs(t, h ** sp.Rational(1, ramification))
    return analytic_powsimp(sp.expand(result))


def _wrap_cover_vectors(
    vectors: tuple[LogarithmicBasisVector, ...],
    localized: LocalizedOperator,
    local_q: sp.Expr,
    q_t: sp.Expr,
    ramification: int,
) -> tuple[LogarithmicBasisVector, ...]:
    h = localized.local_variable
    result: list[LogarithmicBasisVector] = []
    for vector in vectors:
        t = vector.local_parameter
        reduced_local = _cover_to_local(vector.parameter_expression, t, h, ramification)
        local_expression = sp.exp(local_q) * reduced_local
        result.append(
            LogarithmicBasisVector(
                source_exponent=sp.simplify(vector.source_exponent / ramification),
                exponent=sp.simplify(vector.exponent / ramification),
                ramified_exponent=vector.ramified_exponent,
                root_multiplicity=vector.root_multiplicity,
                pole_order=vector.pole_order,
                derivative_order=vector.derivative_order,
                logarithmic_degree=vector.logarithmic_degree,
                local_parameter=t,
                parameter_expression=vector.parameter_expression,
                local_expression=analytic_powsimp(sp.expand(local_expression)),
                expression=localized.to_original(
                    analytic_powsimp(sp.expand(local_expression))
                ),
            )
        )
    return tuple(result)


def _simple_irregular_blocks(
    localized: LocalizedOperator,
    parts: tuple[CompleteFormalExponentialPart, ...],
    *,
    terms: int,
    ode,
    function,
    variable,
    point,
    max_branches: int,
) -> tuple[FormalSolutionBlock, ...]:
    amplitudes = formal_amplitude_series(
        ode,
        function,
        variable,
        point=point,
        terms=terms,
        max_branches=max_branches,
    )
    by_part = {id(amplitude.exponential_part): amplitude for amplitude in amplitudes}
    blocks: list[FormalSolutionBlock] = []
    for part in parts:
        if part.multiplicity != 1:
            continue
        amplitude = by_part.get(id(part))
        if amplitude is None:
            # complete_formal_exponential_parts() was called independently, so
            # object identity need not survive. Match by formal data instead.
            amplitude = next(
                a
                for a in amplitudes
                if expressions_equal(
                    a.exponential_part.exponential_polynomial,
                    part.exponential_polynomial,
                )
                and expressions_equal(
                    a.exponential_part.algebraic_power, part.algebraic_power
                )
            )
        t = amplitude.local_parameter
        beta = sp.simplify(part.ramification_index * part.algebraic_power)
        reduced = sp.expand(
            t**beta
            * sum(
                amplitude.coefficients[k] * t**k
                for k in range(len(amplitude.coefficients))
            )
        )
        h = localized.local_variable
        local_q = part.local_exponential_polynomial.subs(part.local_coordinate, h)
        amplitude_h = amplitude.local_series.subs(
            amplitude.exponential_part.local_coordinate, h
        )
        local_expr = sp.exp(local_q) * h**part.algebraic_power * amplitude_h
        vector = LogarithmicBasisVector(
            source_exponent=part.algebraic_power,
            exponent=part.algebraic_power,
            ramified_exponent=beta,
            root_multiplicity=1,
            pole_order=0,
            derivative_order=0,
            logarithmic_degree=0,
            local_parameter=t,
            parameter_expression=reduced,
            local_expression=analytic_powsimp(sp.expand(local_expr)),
            expression=sp.simplify(
                sp.exp(part.exponential_polynomial)
                * part.algebraic_prefactor
                * amplitude.series
            ),
        )
        blocks.append(
            FormalSolutionBlock(
                local_exponential_polynomial=local_q,
                exponential_polynomial=part.exponential_polynomial,
                ramification_index=part.ramification_index,
                local_parameter=t,
                basis_vectors=(vector,),
            )
        )
    merged: list[FormalSolutionBlock] = []
    for block in blocks:
        for index, existing in enumerate(merged):
            if (
                existing.ramification_index == block.ramification_index
                and expressions_equal(
                    existing.local_exponential_polynomial,
                    block.local_exponential_polynomial,
                )
            ):
                merged[index] = FormalSolutionBlock(
                    local_exponential_polynomial=existing.local_exponential_polynomial,
                    exponential_polynomial=existing.exponential_polynomial,
                    ramification_index=existing.ramification_index,
                    local_parameter=existing.local_parameter,
                    basis_vectors=existing.basis_vectors + block.basis_vectors,
                )
                break
        else:
            merged.append(block)
    return tuple(merged)


def _repeated_irregular_blocks_from_system_decomposition(
    localized: LocalizedOperator,
    parts: tuple[CompleteFormalExponentialPart, ...],
    *,
    ode,
    function,
    variable,
    point,
    terms: int,
    max_branches: int,
) -> tuple[tuple[FormalSolutionBlock, ...], bool, str | None]:
    """Isolate repeated exponential blocks with the system decomposition layer."""

    # Delayed import avoids a module cycle: block_decomposition consumes the
    # completed scalar parts from formal.py, while formal_basis is a client of
    # the resulting system blocks.
    from .block_decomposition import (
        BlockDecompositionError,
        cyclic_scalar_operator,
        exponential_block_decomposition,
    )

    try:
        decomposition = exponential_block_decomposition(
            ode,
            function,
            variable,
            point=point,
            max_power=max(8, terms + 4),
            max_branches=max_branches,
        )
    except (BlockDecompositionError, ValueError) as exc:
        return (), False, f"formal exponential-block decomposition failed: {exc}"
    if not decomposition.complete:
        return (
            (),
            False,
            decomposition.limitation
            or "formal exponential-block decomposition is incomplete",
        )

    blocks: list[FormalSolutionBlock] = []
    for system_block in decomposition.blocks:
        if system_block.dimension <= 1:
            continue
        metadata = system_block.metadata
        if metadata is None:
            return (
                tuple(blocks),
                False,
                "an isolated repeated block could not be matched to scalar exponential metadata",
            )
        representative = parts[metadata.part_indices[0]]
        local_q = representative.local_exponential_polynomial.subs(
            representative.local_coordinate, localized.local_variable
        )
        try:
            scalar_operator = cyclic_scalar_operator(
                system_block.connection,
                system_block.output_row,
                function_name=f"_U{system_block.index}",
            )
            reduced_operator = _conjugate_scalar_operator_by_exponential(
                scalar_operator,
                system_block.parameter_exponential_polynomial,
            )
            cover_vectors = _basis_for_operator_at_zero(reduced_operator, terms=terms)
        except (FormalBasisError, BlockDecompositionError, ValueError) as exc:
            return (
                tuple(blocks),
                False,
                (
                    "isolated repeated exponential block could not be reduced to a "
                    f"regular scalar block: {exc}"
                ),
            )
        if len(cover_vectors) != system_block.dimension:
            return (
                tuple(blocks),
                False,
                "isolated repeated block did not produce its full logarithmic basis",
            )
        wrapped = _wrap_cover_vectors(
            cover_vectors,
            localized,
            local_q,
            system_block.parameter_exponential_polynomial,
            decomposition.ramification_index,
        )
        blocks.append(
            FormalSolutionBlock(
                local_exponential_polynomial=local_q,
                exponential_polynomial=localized.to_original(local_q),
                ramification_index=decomposition.ramification_index,
                local_parameter=decomposition.parameter,
                basis_vectors=wrapped,
            )
        )
    return tuple(blocks), True, None


def formal_logarithmic_basis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 8,
    max_branches: int = 64,
) -> FormalLogarithmicBasis:
    """Return a formal basis with explicit logarithmic companions when possible.

    Ordinary and regular-singular points use the parameterized Frobenius
    construction directly.  At an irregular point, simple completed branches
    are returned as one-dimensional blocks.  If every branch shares one
    completed exponential polynomial, the full exponential is removed on a
    common cover and repeated/resonant amplitude roots are resolved there by
    the same logarithmic construction.

    For a mixed irregular problem, repeated completed exponential factors are
    isolated as formal system blocks, scalarized through the physical output
    row, and reduced independently.  This avoids applying Frobenius analysis
    to an exponential-conjugated *full* scalar operator that still contains
    other irregular modes.
    """

    localized = localize_operator(ode, function, variable, point=point)
    kind = classify_ode_point(localized.operator, point=0).kind
    if kind in (ODESingularityKind.ORDINARY, ODESingularityKind.REGULAR):
        return logarithmic_frobenius_basis(
            ode,
            function,
            variable,
            point=point,
            terms=terms,
        )
    if kind is not ODESingularityKind.IRREGULAR:
        raise FormalBasisError(f"cannot construct a formal basis at point {point!s}")

    parts = complete_formal_exponential_parts(
        ode,
        function,
        variable,
        point=point,
        max_branches=max_branches,
    )
    groups = _group_completed_parts(parts)
    order = localized.operator.order

    if len(groups) == 1 and sum(part.multiplicity for part in groups[0]) == order:
        group = groups[0]
        ramification = _common_ramification(group)
        local_q = group[0].local_exponential_polynomial.subs(
            group[0].local_coordinate, localized.local_variable
        )
        cover_operator, t, q_t = _q_conjugated_cover_operator(
            localized,
            local_q,
            ramification,
        )
        cover_vectors = _basis_for_operator_at_zero(cover_operator, terms=terms)
        wrapped = _wrap_cover_vectors(
            cover_vectors,
            localized,
            local_q,
            q_t,
            ramification,
        )
        block = FormalSolutionBlock(
            local_exponential_polynomial=local_q,
            exponential_polynomial=localized.to_original(local_q),
            ramification_index=ramification,
            local_parameter=t,
            basis_vectors=wrapped,
        )
        return FormalLogarithmicBasis(
            point=sp.sympify(point),
            local_coordinate=localized.local_variable,
            blocks=(block,),
            terms=terms,
            operator_order=order,
            complete=len(wrapped) == order,
        )

    simple_blocks = _simple_irregular_blocks(
        localized,
        parts,
        terms=terms,
        ode=ode,
        function=function,
        variable=variable,
        point=point,
        max_branches=max_branches,
    )
    repeated_parts_present = any(part.multiplicity > 1 for part in parts)
    repeated_blocks: tuple[FormalSolutionBlock, ...] = ()
    decomposition_complete = True
    limitation = None
    if repeated_parts_present:
        repeated_blocks, decomposition_complete, limitation = (
            _repeated_irregular_blocks_from_system_decomposition(
                localized,
                parts,
                ode=ode,
                function=function,
                variable=variable,
                point=point,
                terms=terms,
                max_branches=max_branches,
            )
        )

    blocks = simple_blocks + repeated_blocks

    # Preserve the scalar exponential-part order so branch permutation and
    # monodromy remain deterministic.
    def block_order(block: FormalSolutionBlock) -> int:
        for index, part in enumerate(parts):
            local_q = part.local_exponential_polynomial.subs(
                part.local_coordinate, localized.local_variable
            )
            if expressions_equal(block.local_exponential_polynomial, local_q):
                return index
        return len(parts)

    blocks = tuple(sorted(blocks, key=block_order))
    dimension = sum(block.dimension for block in blocks)
    return FormalLogarithmicBasis(
        point=sp.sympify(point),
        local_coordinate=localized.local_variable,
        blocks=blocks,
        terms=terms,
        operator_order=order,
        complete=dimension == order and decomposition_complete,
        limitation=limitation
        if dimension != order or not decomposition_complete
        else None,
    )


def _term_signature(
    term: sp.Expr,
    variable: sp.Symbol,
) -> tuple[sp.Expr, int, sp.Expr]:
    term = sp.expand_power_base(term, force=False)
    logv = sp.log(variable)
    powers = term.as_powers_dict()
    power = sp.simplify(powers.get(variable, sp.S.Zero))
    log_degree_expr = powers.get(logv, sp.S.Zero)
    if log_degree_expr.is_Integer is not True:
        raise FormalBasisError(
            "formal monodromy encountered a non-polynomial logarithm"
        )
    log_degree = int(log_degree_expr)
    coefficient = sp.simplify(term / (variable**power * logv**log_degree))
    if coefficient.has(variable, logv):
        raise FormalBasisError(
            f"could not decompose formal monomial {term!s} into power/log form"
        )
    return power, log_degree, coefficient


def _formal_coefficient_dict(
    expr: sp.Expr, variable: sp.Symbol
) -> dict[tuple[sp.Expr, int], sp.Expr]:
    result: dict[tuple[sp.Expr, int], sp.Expr] = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        power, log_degree, coefficient = _term_signature(term, variable)
        key = (power, log_degree)
        result[key] = sp.simplify(result.get(key, sp.S.Zero) + coefficient)
    return {key: value for key, value in result.items() if sp.simplify(value) != 0}


def _continued_expression(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    turn_fraction: sp.Rational = sp.S.One,
) -> sp.Expr:
    result = sp.S.Zero
    turn_fraction = sp.Rational(turn_fraction)
    shift = 2 * sp.pi * sp.I * turn_fraction
    logv = sp.log(variable)
    for term in sp.Add.make_args(sp.expand(expr)):
        power, log_degree, coefficient = _term_signature(term, variable)
        result += (
            coefficient
            * sp.exp(2 * sp.pi * sp.I * turn_fraction * power)
            * variable**power
            * (logv + shift) ** log_degree
        )
    return sp.expand(result)


def _coordinates_in_basis(
    target: sp.Expr,
    basis_expressions: tuple[sp.Expr, ...],
    variable: sp.Symbol,
) -> tuple[sp.Expr, ...]:
    dictionaries = tuple(
        _formal_coefficient_dict(expr, variable) for expr in basis_expressions
    )
    target_dict = _formal_coefficient_dict(target, variable)
    keys = sorted(
        set(target_dict).union(*(set(item) for item in dictionaries)),
        key=lambda item: (sp.default_sort_key(item[0]), item[1]),
    )
    rows: list[list[sp.Expr]] = []
    rhs_values: list[sp.Expr] = []
    rank = 0
    for key in keys:
        candidate = [item.get(key, sp.S.Zero) for item in dictionaries]
        trial = sp.Matrix([*rows, candidate])
        new_rank = int(trial.rank())
        rows.append(candidate)
        rhs_values.append(target_dict.get(key, sp.S.Zero))
        rank = new_rank
        if rank == len(dictionaries):
            break
    if rank != len(dictionaries):
        raise FormalBasisError("formal basis jets do not have full rank")
    matrix = sp.Matrix(rows)
    rhs = sp.Matrix(rhs_values)
    try:
        solution_set = sp.linsolve((matrix, rhs))
    except SYMBOLIC_FAILURES as exc:
        raise FormalBasisError("could not solve formal monodromy coordinates") from exc
    solutions = list(solution_set)
    if len(solutions) != 1:
        raise FormalBasisError(
            "formal monodromy coordinates are not uniquely determined"
        )
    solution = tuple(sp.simplify(value) for value in solutions[0])
    generated = set().union(*(value.free_symbols for value in solution))
    basis_symbols = set().union(
        *(expr.free_symbols for expr in basis_expressions), target.free_symbols
    )
    if generated - basis_symbols:
        raise FormalBasisError("formal monodromy coordinate solve left free parameters")
    return solution


def _block_cover_monodromy(block: FormalSolutionBlock) -> sp.Matrix:
    vectors = block.basis_vectors
    if not vectors:
        return sp.zeros(0, 0)
    variable = block.local_parameter
    expressions = tuple(vector.parameter_expression for vector in vectors)
    columns: list[tuple[sp.Expr, ...]] = []
    for expression in expressions:
        continued = _continued_expression(expression, variable)
        columns.append(_coordinates_in_basis(continued, expressions, variable))
    return sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))


def _continued_local_exponential(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    result = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(expr)):
        term = sp.expand_power_base(term, force=False)
        power = sp.simplify(term.as_powers_dict().get(variable, sp.S.Zero))
        coefficient = sp.simplify(term / variable**power)
        if coefficient.has(variable):
            raise FormalBasisError(f"could not decompose exponential monomial {term!s}")
        result += coefficient * sp.exp(2 * sp.pi * sp.I * power) * variable**power
    return analytic_powsimp(sp.expand(result))


def _local_monodromy_matrix(basis: FormalLogarithmicBasis) -> sp.Matrix | None:
    blocks = basis.blocks
    offsets: list[int] = []
    total = 0
    for block in blocks:
        offsets.append(total)
        total += block.dimension
    matrix = sp.zeros(total, total)
    h = basis.local_coordinate

    for source_index, source in enumerate(blocks):
        continued_q = _continued_local_exponential(
            source.local_exponential_polynomial, h
        )
        targets = [
            (index, block)
            for index, block in enumerate(blocks)
            if block.ramification_index == source.ramification_index
            and expressions_equal(block.local_exponential_polynomial, continued_q)
        ]
        if len(targets) != 1:
            return None
        target_index, target = targets[0]
        target_expressions = tuple(v.parameter_expression for v in target.basis_vectors)
        if not target_expressions:
            return None
        fraction = sp.Rational(1, source.ramification_index)
        for local_column, vector in enumerate(source.basis_vectors):
            continued = _continued_expression(
                vector.parameter_expression,
                source.local_parameter,
                turn_fraction=fraction,
            )
            if source.local_parameter != target.local_parameter:
                continued = continued.subs(
                    source.local_parameter, target.local_parameter
                )
            try:
                coordinates = _coordinates_in_basis(
                    continued, target_expressions, target.local_parameter
                )
            except FormalBasisError:
                return None
            global_column = offsets[source_index] + local_column
            for row, value in enumerate(coordinates):
                matrix[offsets[target_index] + row, global_column] = sp.simplify(value)
    return matrix


def formal_monodromy(basis: FormalLogarithmicBasis) -> FormalMonodromy:
    """Return formal monodromy matrices for a completed formal basis.

    The basis must be complete.  Matrix columns give the continued basis
    vectors.  For ramified blocks, ``cover_matrix`` is always meaningful; a
    one-turn matrix in the original local coordinate is also reconstructed
    when continuation maps every exponential block to a uniquely represented
    target block; otherwise ``local_matrix`` is ``None``.
    """

    if not basis.complete or basis.dimension != basis.operator_order:
        raise FormalBasisError(
            "formal monodromy requires a complete formal basis"
            + (f": {basis.limitation}" if basis.limitation else "")
        )
    block_matrices = [_block_cover_monodromy(block) for block in basis.blocks]
    if not block_matrices:
        matrix = sp.zeros(0, 0)
    else:
        matrix = sp.diag(*block_matrices)
    local_matrix = _local_monodromy_matrix(basis)
    return FormalMonodromy(
        basis=basis,
        cover_matrix=matrix,
        local_matrix=local_matrix,
    )
