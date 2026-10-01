"""Recognition of classical second-order canonical equations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations

import sympy as sp

from .fuchsian import _complete_exponents
from .operator import LinearDifferentialOperator, _coerce_linear_operator
from .singularities import ODESingularityKind, analyze_ode_singularities


class CanonicalEquationFamily(Enum):
    """Classical second-order equation families recognized by ``odeanalysis``."""

    AIRY = "airy"
    EULER = "euler"
    BESSEL = "bessel"
    MODIFIED_BESSEL = "modified_bessel"
    HYPERGEOMETRIC = "hypergeometric"
    CONFLUENT_HYPERGEOMETRIC = "confluent_hypergeometric"


@dataclass(frozen=True)
class CanonicalEquationRecognition:
    """An exact recognition certificate for a classical canonical equation.

    ``variable_transform`` is the projective change ``z=z(x)`` and
    ``dependent_gauge`` records ``y(x)=g(x)u(z(x))``.  The logarithmic
    derivative of the gauge is stored separately so verification does not
    depend on branch-sensitive simplification of symbolic powers.
    """

    family: CanonicalEquationFamily
    operator: LinearDifferentialOperator
    canonical_variable: sp.Symbol
    variable_transform: sp.Expr
    mobius_coefficients: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]
    dependent_gauge: sp.Expr
    gauge_log_derivative: sp.Expr
    parameters: tuple[tuple[str, sp.Expr], ...]
    canonical_expression: sp.Expr

    @property
    def parameter_map(self) -> dict[str, sp.Expr]:
        """Return canonical parameters as a fresh mapping."""

        return dict(self.parameters)

    @property
    def scale(self) -> sp.Expr | None:
        """Return the affine scale when the projective map is affine."""

        a, _, c, d = self.mobius_coefficients
        if sp.simplify(c) != 0:
            return None
        return sp.simplify(a / d)

    @property
    def shift(self) -> sp.Expr | None:
        """Return the affine shift when the projective map is affine."""

        _, b, c, d = self.mobius_coefficients
        if sp.simplify(c) != 0:
            return None
        return sp.simplify(b / d)

    @property
    def is_affine(self) -> bool:
        """Whether the independent-variable transformation is affine."""

        return sp.simplify(self.mobius_coefficients[2]) == 0

    def verify(self) -> bool:
        """Replay the projective pullback and gauge transformation exactly."""

        x = self.operator.variable
        z = self.canonical_variable
        u = sp.Function("_odeanalysis_canonical_u")
        canonical = LinearDifferentialOperator.from_ode(
            self.canonical_expression, u, z
        ).normalized()
        target = self.operator.normalized()
        z_of_x = self.variable_transform
        z_prime = sp.diff(z_of_x, x)
        if sp.simplify(z_prime) == 0:
            return False
        z_second = sp.diff(z_prime, x)
        h = self.gauge_log_derivative
        canonical_p = canonical.coefficients[1].subs(z, z_of_x)
        canonical_q = canonical.coefficients[0].subs(z, z_of_x)
        expected_p = sp.cancel(z_prime * canonical_p - 2 * h - z_second / z_prime)
        expected_q = sp.cancel(
            z_prime**2 * canonical_q - expected_p * h - sp.diff(h, x) - h**2
        )
        expected = (expected_q, expected_p, sp.S.One)
        return all(
            sp.simplify(sp.cancel(a - b)) == 0
            for a, b in zip(expected, target.coefficients, strict=True)
        )


def _canonical_expression(
    family: CanonicalEquationFamily,
    z: sp.Symbol,
    params: dict[str, sp.Expr],
) -> sp.Expr:
    u = sp.Function("_odeanalysis_canonical_u")
    uz = u(z)
    if family is CanonicalEquationFamily.AIRY:
        return sp.diff(uz, z, 2) - z * uz
    if family is CanonicalEquationFamily.EULER:
        alpha, beta = params["alpha"], params["beta"]
        return z**2 * sp.diff(uz, z, 2) + alpha * z * sp.diff(uz, z) + beta * uz
    if family is CanonicalEquationFamily.BESSEL:
        nu = params["nu"]
        return z**2 * sp.diff(uz, z, 2) + z * sp.diff(uz, z) + (z**2 - nu**2) * uz
    if family is CanonicalEquationFamily.MODIFIED_BESSEL:
        nu = params["nu"]
        return z**2 * sp.diff(uz, z, 2) + z * sp.diff(uz, z) - (z**2 + nu**2) * uz
    if family is CanonicalEquationFamily.HYPERGEOMETRIC:
        a, b, c = params["a"], params["b"], params["c"]
        return (
            z * (1 - z) * sp.diff(uz, z, 2)
            + (c - (a + b + 1) * z) * sp.diff(uz, z)
            - a * b * uz
        )
    if family is CanonicalEquationFamily.CONFLUENT_HYPERGEOMETRIC:
        a, c = params["a"], params["c"]
        return z * sp.diff(uz, z, 2) + (c - z) * sp.diff(uz, z) - a * uz
    raise ValueError(f"unsupported canonical family: {family}")


def _projective_recognition(
    family: CanonicalEquationFamily,
    op: LinearDifferentialOperator,
    mobius: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    params: dict[str, sp.Expr],
    *,
    dependent_gauge: sp.Expr = sp.S.One,
    gauge_log_derivative: sp.Expr = sp.S.Zero,
) -> CanonicalEquationRecognition:
    x = op.variable
    a, b, c, d = map(sp.simplify, mobius)
    determinant = sp.simplify(a * d - b * c)
    if determinant == 0:
        raise ValueError("Möbius transformation must have nonzero determinant")
    z_of_x = sp.cancel((a * x + b) / (c * x + d))
    z = sp.Symbol("z")
    return CanonicalEquationRecognition(
        family=family,
        operator=op,
        canonical_variable=z,
        variable_transform=z_of_x,
        mobius_coefficients=(a, b, c, d),
        dependent_gauge=sp.simplify(dependent_gauge),
        gauge_log_derivative=sp.cancel(gauge_log_derivative),
        parameters=tuple(sorted(params.items())),
        canonical_expression=_canonical_expression(family, z, params),
    )


def _recognition(
    family: CanonicalEquationFamily,
    op: LinearDifferentialOperator,
    scale: sp.Expr,
    shift: sp.Expr,
    params: dict[str, sp.Expr],
) -> CanonicalEquationRecognition:
    return _projective_recognition(
        family,
        op,
        (scale, shift, sp.S.Zero, sp.S.One),
        params,
    )


def _recognize_euler(
    op: LinearDifferentialOperator,
) -> CanonicalEquationRecognition | None:
    """Recognize an exact Euler--Cauchy equation at the origin."""
    x = op.variable
    normalized = op.normalized()
    p, q = normalized.coefficients[1], normalized.coefficients[0]
    alpha = sp.cancel(x * p)
    beta = sp.cancel(x**2 * q)
    if alpha.has(x) or beta.has(x):
        return None
    result = _recognition(
        CanonicalEquationFamily.EULER,
        op,
        sp.S.One,
        sp.S.Zero,
        {"alpha": alpha, "beta": beta},
    )
    return result if result.verify() else None


def _recognize_airy(
    op: LinearDifferentialOperator,
) -> CanonicalEquationRecognition | None:
    x = op.variable
    normalized = op.normalized()
    p, q = normalized.coefficients[1], normalized.coefficients[0]
    if sp.simplify(p) != 0:
        return None
    try:
        poly = sp.Poly(sp.cancel(q), x)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    slope = poly.coeff_monomial(x)
    intercept = poly.coeff_monomial(1)
    scale = (
        sp.real_root(-slope, 3)
        if slope.is_real is True
        else (-slope) ** sp.Rational(1, 3)
    )
    if sp.simplify(scale) == 0:
        return None
    shift = sp.simplify(-intercept / scale**2)
    result = _recognition(CanonicalEquationFamily.AIRY, op, scale, shift, {})
    return result if result.verify() else None


def _bessel_candidate(
    op: LinearDifferentialOperator, modified: bool
) -> CanonicalEquationRecognition | None:
    x = op.variable
    normalized = op.normalized()
    p, q = map(sp.cancel, (normalized.coefficients[1], normalized.coefficients[0]))
    # p = 1/(x-x0) fixes the affine shift exactly.
    numerator, denominator = sp.fraction(sp.cancel(1 / p)) if p != 0 else (0, 1)
    try:
        affine = sp.Poly(sp.cancel(numerator / denominator), x)
    except sp.PolynomialError:
        return None
    if affine.degree() != 1 or sp.simplify(affine.coeff_monomial(x) - 1) != 0:
        return None
    x0 = sp.simplify(-affine.coeff_monomial(1))
    t = sp.expand(x - x0)
    if sp.simplify(p - 1 / t) != 0:
        return None
    constant = sp.simplify(sp.limit(q, x, sp.oo))
    if modified and constant.is_positive is True:
        return None
    if not modified and constant.is_negative is True:
        return None
    scale_sq = -constant if modified else constant
    if sp.simplify(scale_sq) == 0:
        return None
    nu_sq = sp.simplify(t**2 * ((-scale_sq if modified else scale_sq) - q))
    if x in nu_sq.free_symbols:
        return None
    scale = sp.sqrt(scale_sq)
    nu = sp.sqrt(nu_sq)
    family = (
        CanonicalEquationFamily.MODIFIED_BESSEL
        if modified
        else CanonicalEquationFamily.BESSEL
    )
    result = _recognition(family, op, scale, sp.simplify(-scale * x0), {"nu": nu})
    return result if result.verify() else None


def _recognize_confluent(
    op: LinearDifferentialOperator,
) -> CanonicalEquationRecognition | None:
    x = op.variable
    p = sp.cancel(op.normalized().coefficients[1])
    q = sp.cancel(op.normalized().coefficients[0])
    # After z=s(x-x0): p = c/(x-x0)-s and q = -a*s/(x-x0).
    den = sp.denom(sp.together(p))
    roots = sp.solve(den, x)
    if len(roots) != 1:
        return None
    x0 = roots[0]
    t = x - x0
    scale = sp.simplify(-sp.limit(p, x, sp.oo))
    if sp.simplify(scale) == 0:
        return None
    c = sp.simplify(sp.limit(t * (p + scale), x, x0))
    a = sp.simplify(-t * q / scale)
    if (
        x in a.free_symbols
        or sp.simplify(p - (c / t - scale)) != 0
        or sp.simplify(q + a * scale / t) != 0
    ):
        return None
    result = _recognition(
        CanonicalEquationFamily.CONFLUENT_HYPERGEOMETRIC,
        op,
        scale,
        sp.simplify(-scale * x0),
        {"a": a, "c": c},
    )
    return result if result.verify() else None


def _recognize_hypergeometric(
    op: LinearDifferentialOperator,
) -> CanonicalEquationRecognition | None:
    x = op.variable
    normalized = op.normalized()
    p, q = map(sp.cancel, (normalized.coefficients[1], normalized.coefficients[0]))
    # Affine pullbacks have exactly two finite poles x0,x1. Map them to 0,1.
    den = sp.lcm(sp.denom(sp.together(p)), sp.denom(sp.together(q)))
    roots = sp.solve(den, x)
    if len(roots) != 2:
        return None
    for x0, x1 in (roots, roots[::-1]):
        delta = sp.simplify(x1 - x0)
        if sp.simplify(delta) == 0:
            continue
        scale = sp.simplify(1 / delta)
        z = sp.simplify((x - x0) / delta)
        # p = scale * [c-(a+b+1)z]/[z(1-z)]
        # q = -scale^2*a*b/[z(1-z)].
        c = sp.simplify(sp.limit(z * p / scale, x, x0))
        sum_ab = sp.simplify(c - 1 - sp.cancel(p * z * (1 - z) / scale).coeff(z, 1))
        # coeff(z,1) is unreliable because z is an expression; infer using endpoint residue.
        res1 = sp.simplify(sp.limit((1 - z) * p / scale, x, x1))
        sum_ab = sp.simplify(c - res1 - 1)
        product = sp.simplify(-q * z * (1 - z) / scale**2)
        if x in product.free_symbols:
            continue
        disc = sp.simplify(sum_ab**2 - 4 * product)
        a = sp.simplify((sum_ab + sp.sqrt(disc)) / 2)
        b = sp.simplify((sum_ab - sp.sqrt(disc)) / 2)
        expected_p = sp.cancel(scale * (c - (a + b + 1) * z) / (z * (1 - z)))
        expected_q = sp.cancel(-(scale**2) * a * b / (z * (1 - z)))
        if sp.simplify(p - expected_p) != 0 or sp.simplify(q - expected_q) != 0:
            continue
        result = _recognition(
            CanonicalEquationFamily.HYPERGEOMETRIC,
            op,
            scale,
            sp.simplify(-scale * x0),
            {"a": a, "b": b, "c": c},
        )
        if result.verify():
            return result
    return None


def _mobius_to_zero_one_infinity(
    x: sp.Symbol,
    p0: sp.Expr,
    p1: sp.Expr,
    pinf: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return a projective map sending ``p0,p1,pinf`` to ``0,1,oo``."""

    if pinf == sp.oo:
        delta = sp.simplify(p1 - p0)
        return (sp.simplify(1 / delta), sp.simplify(-p0 / delta), sp.S.Zero, sp.S.One)
    if p0 == sp.oo:
        return (sp.S.Zero, sp.simplify(p1 - pinf), sp.S.One, sp.simplify(-pinf))
    if p1 == sp.oo:
        return (sp.S.One, sp.simplify(-p0), sp.S.One, sp.simplify(-pinf))
    scale = sp.simplify((p1 - pinf) / (p1 - p0))
    return (scale, sp.simplify(-scale * p0), sp.S.One, sp.simplify(-pinf))


def _regular_singularity_exponents(
    op: LinearDifferentialOperator,
) -> tuple[tuple[sp.Expr, tuple[sp.Expr, ...]], ...] | None:
    analysis = analyze_ode_singularities(op, include_infinity=True)
    singularities = list(analysis.finite)
    if (
        analysis.infinity is not None
        and analysis.infinity.kind is not ODESingularityKind.ORDINARY
    ):
        singularities.append(analysis.infinity)
    if len(singularities) != 3:
        return None
    result: list[tuple[sp.Expr, tuple[sp.Expr, ...]]] = []
    for singularity in singularities:
        if singularity.kind is not ODESingularityKind.REGULAR:
            return None
        exponents = _complete_exponents(singularity)
        if exponents is None or len(exponents) != 2:
            return None
        result.append((singularity.point, exponents))
    return tuple(result)


def _recognize_projective_hypergeometric(
    op: LinearDifferentialOperator,
) -> CanonicalEquationRecognition | None:
    """Recognize a general three-regular-singularity Riemann P equation."""

    singularities = _regular_singularity_exponents(op)
    if singularities is None:
        return None
    x = op.variable

    for ordered in permutations(singularities):
        (p0, exp0), (p1, exp1), (pinf, expinf) = ordered
        try:
            mobius = _mobius_to_zero_one_infinity(x, p0, p1, pinf)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        a_m, b_m, c_m, d_m = mobius
        z_of_x = sp.cancel((a_m * x + b_m) / (c_m * x + d_m))
        z_prime = sp.diff(z_of_x, x)
        for alpha in exp0:
            other0 = exp0[1] if alpha == exp0[0] else exp0[0]
            c_param = sp.simplify(1 + alpha - other0)
            for beta in exp1:
                other1 = exp1[1] if beta == exp1[0] else exp1[0]
                a_param = sp.simplify(expinf[0] + alpha + beta)
                b_param = sp.simplify(expinf[1] + alpha + beta)
                exponent_gap = sp.simplify(c_param - a_param - b_param)
                if sp.simplify(other1 - beta - exponent_gap) != 0:
                    continue
                h = sp.cancel(alpha * z_prime / z_of_x - beta * z_prime / (1 - z_of_x))
                gauge = sp.simplify(z_of_x**alpha * (1 - z_of_x) ** beta)
                result = _projective_recognition(
                    CanonicalEquationFamily.HYPERGEOMETRIC,
                    op,
                    mobius,
                    {"a": a_param, "b": b_param, "c": c_param},
                    dependent_gauge=gauge,
                    gauge_log_derivative=h,
                )
                if result.verify():
                    return result
    return None


def recognize_canonical_equation(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> CanonicalEquationRecognition | None:
    """Recognize an exact classical equation under affine or projective/gauge pullback."""

    op = _coerce_linear_operator(ode, function, variable)
    if not op.is_homogeneous or op.order != 2:
        return None
    for recognizer in (
        _recognize_euler,
        _recognize_airy,
        lambda candidate: _bessel_candidate(candidate, False),
        lambda candidate: _bessel_candidate(candidate, True),
        _recognize_hypergeometric,
        _recognize_confluent,
        _recognize_projective_hypergeometric,
    ):
        result = recognizer(op)
        if result is not None:
            return result
    return None


def transform_to_canonical(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> CanonicalEquationRecognition:
    """Return a verified canonical transformation or raise ``ValueError``."""

    result = recognize_canonical_equation(ode, function, variable)
    if result is None:
        raise ValueError("equation is not in a recognized classical canonical family")
    return result
