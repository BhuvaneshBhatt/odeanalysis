"""ODE-specific algebraic loci where structural analysis can change."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from semialg import is_satisfiable, matrix_rank_stratification

from ._zero import ZeroStatus, exact_zero_status
from .newton import localize_operator
from .operator import LinearDifferentialOperator, _coerce_linear_operator
from .system import FirstOrderSystem
from .turning import liouville_normal_form


def _parameter_factors(
    expression: sp.Expr, excluded: set[sp.Symbol]
) -> tuple[sp.Expr, ...]:
    expression = sp.factor(expression)
    factors = (
        sp.factor_list(expression)[1]
        if exact_zero_status(expression) is not ZeroStatus.ZERO
        else ()
    )
    result = []
    for factor, _ in factors:
        if factor.free_symbols - excluded:
            result.append(sp.factor(factor))
    return tuple(dict.fromkeys(result))


def _polynomial_parameter_loci(
    expression: sp.Expr, variable: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Return loci where the effective degree or finite root multiplicities can change."""
    try:
        polynomial = sp.Poly(sp.expand(expression), variable)
    except sp.PolynomialError:
        return ()
    loci: list[sp.Expr] = []
    # Every coefficient can become the leading coefficient on a lower-degree stratum.
    # Its vanishing therefore represents a projective root-at-infinity transition.
    coefficients = [
        sp.factor(polynomial.nth(k)) for k in range(polynomial.degree() + 1)
    ]
    for coefficient in coefficients:
        loci.extend(_parameter_factors(coefficient, {variable}))
    # On each possible effective-degree stratum, the corresponding truncation's
    # discriminant detects finite multiple-root collisions.
    for degree in range(2, polynomial.degree() + 1):
        truncated = sum(polynomial.nth(k) * variable**k for k in range(degree + 1))
        if polynomial.nth(degree) == 0:
            continue
        loci.extend(
            _parameter_factors(sp.discriminant(truncated, variable), {variable})
        )
    return tuple(dict.fromkeys(loci))


def _effective_support_polynomials(
    expression: sp.Expr, variable: sp.Symbol
) -> tuple[sp.Poly, ...]:
    """Canonical numerator/denominator polynomials for rational support analysis."""
    numerator, denominator = sp.fraction(sp.cancel(expression))
    polynomials: list[sp.Poly] = []
    for part in (numerator, denominator):
        try:
            polynomials.append(sp.Poly(part, variable))
        except sp.PolynomialError:
            continue
    return tuple(polynomials)


def _support_loci(expression: sp.Expr, variable: sp.Symbol) -> tuple[sp.Expr, ...]:
    loci: list[sp.Expr] = []
    for polynomial in _effective_support_polynomials(expression, variable):
        for coefficient in polynomial.all_coeffs():
            loci.extend(_parameter_factors(coefficient, {variable}))
    return tuple(dict.fromkeys(loci))


def _rational_numerator(expression: sp.Expr) -> sp.Expr:
    return sp.fraction(sp.cancel(expression))[0]


def newton_loci(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
) -> tuple[sp.Expr, ...]:
    """Return loci where normalized effective support can change the Newton polygon."""
    localized = localize_operator(ode, function, variable, point=point)
    t = localized.local_variable
    loci: list[sp.Expr] = []
    for coefficient in localized.operator.coefficients:
        if coefficient != 0:
            loci.extend(_support_loci(coefficient, t))
    return tuple(dict.fromkeys(loci))


@dataclass(frozen=True)
class ProjectivePolynomialStratum:
    """A certified effective-degree stratum of a fixed projective polynomial family."""

    condition: sp.Expr
    effective_degree: int
    infinity_multiplicity: int
    certified: bool = True


def _projective_strata_from_polynomial(
    polynomial: sp.Poly, parameters: Sequence[sp.Symbol]
) -> tuple[ProjectivePolynomialStratum, ...]:
    """Interpret nested leading-coefficient losses as cumulative roots at infinity."""
    reference_degree = polynomial.degree()
    strata: list[ProjectivePolynomialStratum] = []
    higher_zero: list[sp.Expr] = []
    for degree in range(reference_degree, -1, -1):
        coefficient = sp.factor(polynomial.nth(degree))
        if coefficient == 0:
            higher_zero.append(sp.true)
            continue
        condition = sp.And(*higher_zero, sp.Ne(coefficient, 0, evaluate=False))
        if is_satisfiable(condition, parameters) is not False:
            strata.append(
                ProjectivePolynomialStratum(
                    condition, degree, reference_degree - degree
                )
            )
        higher_zero.append(sp.Eq(coefficient, 0, evaluate=False))
    zero_condition = sp.And(*higher_zero) if higher_zero else sp.false
    if is_satisfiable(zero_condition, parameters) is not False:
        strata.append(
            ProjectivePolynomialStratum(zero_condition, -1, reference_degree + 1)
        )
    return tuple(strata)


def singularity_loci(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> tuple[sp.Expr, ...]:
    """Return finite-collision and projective degree-loss loci of singular points."""
    operator = _coerce_linear_operator(ode, function, variable)
    x = operator.variable
    numerator = _rational_numerator(operator.coefficients[-1])
    return _polynomial_parameter_loci(numerator, x)


def singularity_projective_strata(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    parameters: Sequence[sp.Symbol],
) -> tuple[ProjectivePolynomialStratum, ...]:
    """Return certified effective-degree/root-at-infinity strata for singular points."""
    operator = _coerce_linear_operator(ode, function, variable)
    x = operator.variable
    polynomial = sp.Poly(sp.expand(_rational_numerator(operator.coefficients[-1])), x)
    return _projective_strata_from_polynomial(polynomial, parameters)


def turning_loci(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> tuple[sp.Expr, ...]:
    """Return finite-collision and projective degree-loss loci of turning points."""
    normal = liouville_normal_form(ode, function, variable)
    x = normal.operator.variable
    numerator = _rational_numerator(normal.potential)
    return _polynomial_parameter_loci(numerator, x)


def turning_point_projective_strata(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    parameters: Sequence[sp.Symbol],
) -> tuple[ProjectivePolynomialStratum, ...]:
    """Return certified effective-degree/root-at-infinity strata for turning points."""
    normal = liouville_normal_form(ode, function, variable)
    x = normal.operator.variable
    polynomial = sp.Poly(sp.expand(_rational_numerator(normal.potential)), x)
    return _projective_strata_from_polynomial(polynomial, parameters)


def stokes_formal_loci(
    exponential_parts: Sequence[sp.Expr], local_variable: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Return loci where leading pairwise exponential differences lose formal degree."""
    loci: list[sp.Expr] = []
    for i, left in enumerate(exponential_parts):
        for right in exponential_parts[i + 1 :]:
            difference = sp.expand(left - right)
            if exact_zero_status(difference) is not ZeroStatus.ZERO:
                loci.extend(_support_loci(difference, local_variable))
    return tuple(dict.fromkeys(loci))


def _require_real_parameters(parameters: Sequence[sp.Symbol]) -> None:
    if any(parameter.is_real is not True for parameter in parameters):
        raise ValueError(
            "certified Stokes-ray geometry requires explicit real parameter coordinates"
        )


def _complex_zero_polynomial(expression: sp.Expr) -> sp.Expr:
    expanded = sp.expand_complex(expression)
    return sp.factor(sp.re(expanded) ** 2 + sp.im(expanded) ** 2)


def _phase_alignment_polynomial(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    left = sp.expand_complex(left)
    right = sp.expand_complex(right)
    return sp.simplify(sp.im(left * sp.conjugate(right)))


def stokes_ray_loci(
    exponential_parts: Sequence[sp.Expr],
    local_variable: sp.Symbol,
    parameters: Sequence[sp.Symbol],
) -> tuple[sp.Expr, ...]:
    """Return real-algebraic phase-alignment loci where Stokes rays can coincide/reorder."""
    leading_data: list[tuple[int, sp.Expr]] = []
    for i, left in enumerate(exponential_parts):
        for right in exponential_parts[i + 1 :]:
            difference = sp.cancel(left - right)
            if exact_zero_status(difference) is ZeroStatus.ZERO:
                continue
            try:
                polynomial = sp.Poly(difference, local_variable)
            except sp.PolynomialError:
                continue
            leading_data.append((polynomial.degree(), sp.factor(polynomial.LC())))
    _require_real_parameters(parameters)
    loci: list[sp.Expr] = []
    for formal_locus in stokes_formal_loci(exponential_parts, local_variable):
        real_zero = _complex_zero_polynomial(formal_locus)
        if exact_zero_status(real_zero) is not ZeroStatus.ZERO:
            loci.append(real_zero)
    for i, (degree, left) in enumerate(leading_data):
        for other_degree, right in leading_data[i + 1 :]:
            if degree != other_degree:
                continue
            alignment = _phase_alignment_polynomial(left, right)
            if exact_zero_status(alignment) is not ZeroStatus.ZERO:
                loci.extend(_parameter_factors(alignment, set()))
    return tuple(dict.fromkeys(loci))


@dataclass(frozen=True)
class LeadingMatrixRankAnalysis:
    """Parameter stratification of the leading Laurent matrix rank."""

    pole_order: int
    leading_matrix: sp.ImmutableMatrix
    strata: object


def leading_rank_analysis(
    system: FirstOrderSystem,
    point: sp.Expr,
    parameters: Sequence[sp.Symbol],
    *,
    parameter_domain: sp.Expr = sp.true,
) -> LeadingMatrixRankAnalysis:
    """Stratify the rank of a system's leading Laurent matrix with semialg."""
    x = system.variable
    h = sp.Symbol("_h")
    entries = [sp.cancel(entry.subs(x, point + h)) for entry in system.matrix]
    valuations = []
    for entry in entries:
        if entry == 0:
            continue
        numerator, denominator = sp.fraction(entry)
        try:
            n = sp.Poly(numerator, h)
            d = sp.Poly(denominator, h)
        except sp.PolynomialError as exc:
            raise NotImplementedError(
                "leading-matrix rank requires rational local entries"
            ) from exc
        nv = min(k for k in range(n.degree() + 1) if n.nth(k) != 0)
        dv = min(k for k in range(d.degree() + 1) if d.nth(k) != 0)
        valuations.append(nv - dv)
    minimum = min(valuations, default=0)
    leading = system.matrix.applyfunc(
        lambda entry: sp.simplify(sp.limit(entry.subs(x, point + h) / h**minimum, h, 0))
    )
    immutable = sp.ImmutableMatrix(leading)
    strata = matrix_rank_stratification(
        immutable, parameters, parameter_domain=parameter_domain
    )
    return LeadingMatrixRankAnalysis(-minimum, immutable, strata)
