"""Turning points, Liouville normal form, and uniform WKB reductions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._symbolic_errors import SYMBOLIC_FAILURES
from .operator import LinearDifferentialOperator, _coerce_linear_operator


class TurningPointKind(Enum):
    """Multiplicity class of a zero of the Liouville normal-form potential."""

    SIMPLE = "simple"
    DOUBLE = "double"
    HIGHER = "higher"


@dataclass(frozen=True)
class LiouvilleNormalForm:
    """Exact reduction of a second-order scalar equation to ``u'' = Q u``.

    For a monic equation ``y'' + p y' + q y = 0``, the substitution
    ``y = g u`` with ``g'/g = -p/2`` gives
    ``u'' = Q u`` where ``Q = p'/2 + p**2/4 - q``.
    """

    operator: LinearDifferentialOperator
    p: sp.Expr
    q: sp.Expr
    potential: sp.Expr
    gauge: sp.Expr
    gauge_log_derivative: sp.Expr

    def verify(self) -> bool:
        """Recompute the normal-form potential and gauge identity exactly."""

        x = self.operator.variable
        op = self.operator.normalized()
        if op.order != 2 or not op.is_homogeneous:
            return False
        p = sp.cancel(op.coefficients[1])
        q = sp.cancel(op.coefficients[0])
        expected = sp.cancel(sp.diff(p, x) / 2 + p**2 / 4 - q)
        return (
            sp.simplify(self.p - p) == 0
            and sp.simplify(self.q - q) == 0
            and sp.simplify(self.potential - expected) == 0
            and sp.simplify(self.gauge_log_derivative + p / 2) == 0
        )


@dataclass(frozen=True)
class TurningPoint:
    """A finite zero of the Liouville normal-form potential."""

    point: sp.Expr
    multiplicity: int
    kind: TurningPointKind
    leading_coefficient: sp.Expr
    normal_form: LiouvilleNormalForm

    def verify(self) -> bool:
        """Verify the zero multiplicity and leading local coefficient."""

        x = self.normal_form.operator.variable
        q = self.normal_form.potential
        h = sp.Symbol("_h")
        local = sp.cancel(q.subs(x, self.point + h))
        for order in range(self.multiplicity):
            if sp.simplify(sp.limit(local / h**order, h, 0)) not in (0, sp.S.Zero):
                return False
        coeff = sp.simplify(sp.limit(local / h**self.multiplicity, h, 0))
        return coeff != 0 and sp.simplify(coeff - self.leading_coefficient) == 0


@dataclass(frozen=True)
class TurningPointAnalysis:
    """Resolved finite turning points of a second-order normal-form equation."""

    normal_form: LiouvilleNormalForm
    points: tuple[TurningPoint, ...]
    complete: bool
    limitation: str | None = None

    def verify(self) -> bool:
        """Verify every reported point and, when complete, the full numerator degree."""

        if not self.normal_form.verify() or not all(
            point.verify() for point in self.points
        ):
            return False
        if not self.complete:
            return True
        x = self.normal_form.operator.variable
        numerator, denominator = sp.fraction(sp.cancel(self.normal_form.potential))
        try:
            degree = sp.Poly(numerator, x).degree()
        except sp.PolynomialError:
            return False
        multiplicity = sum(point.multiplicity for point in self.points)
        if degree != multiplicity:
            return False
        return all(
            sp.simplify(denominator.subs(x, point.point)) != 0 for point in self.points
        )


@dataclass(frozen=True)
class WKBExpansion:
    """Formal Riccati/WKB expansion for ``epsilon**2 u'' = Q u``.

    ``coefficients[n]`` is ``S_n`` in
    ``S = sum(epsilon**n*S_n)`` with ``epsilon*S' + S**2 = Q``.
    The associated normal-form solution is
    ``u = exp(Integral(S, x)/epsilon)``.
    """

    normal_form: LiouvilleNormalForm
    parameter: sp.Symbol
    branch: int
    coefficients: tuple[sp.Expr, ...]
    log_derivative_series: sp.Expr
    normal_form_solution: sp.Expr
    original_solution: sp.Expr

    @property
    def order(self) -> int:
        """Highest computed Riccati coefficient index."""

        return len(self.coefficients) - 1

    def verify(self) -> bool:
        """Replay the Riccati recurrence through the requested order."""

        x = self.normal_form.operator.variable
        eps = self.parameter
        if self.branch not in (-1, 1) or not self.coefficients:
            return False
        s0 = self.coefficients[0]
        if sp.simplify(s0**2 - self.normal_form.potential) != 0:
            return False
        for n in range(1, len(self.coefficients)):
            convolution = sum(
                self.coefficients[j] * self.coefficients[n - j] for j in range(1, n)
            )
            expected = sp.cancel(
                -(sp.diff(self.coefficients[n - 1], x) + convolution) / (2 * s0)
            )
            if sp.simplify(self.coefficients[n] - expected) != 0:
                return False
        expected_series = sp.Add(
            *(eps**n * value for n, value in enumerate(self.coefficients))
        )
        return sp.simplify(self.log_derivative_series - expected_series) == 0


@dataclass(frozen=True)
class UniformWKBReduction:
    """Uniform Liouville-Green reduction near a simple or double turning point.

    The exact transformed equation has the form
    ``epsilon**2 W'' = (canonical_potential + epsilon**2*residual) W``.
    ``residual == 0`` therefore means the reduction is an exact canonical
    equation rather than only a uniform leading model.
    """

    turning_point: TurningPoint
    parameter: sp.Symbol
    canonical_variable: sp.Symbol
    variable_transform: sp.Expr
    phase_integral: sp.Expr
    amplitude: sp.Expr
    canonical_family: str
    canonical_potential: sp.Expr
    residual: sp.Expr

    @property
    def original_amplitude(self) -> sp.Expr:
        """Combined Liouville and uniformizing amplitude in the original equation."""

        return sp.simplify(self.turning_point.normal_form.gauge * self.amplitude)

    @property
    def is_exact(self) -> bool:
        """Whether the transformed equation has no Liouville-Green residual."""

        return self.residual == 0

    def verify(self) -> bool:
        """Verify the defining potential map and transformed residual exactly."""

        x = self.turning_point.normal_form.operator.variable
        zeta = self.variable_transform
        zp = sp.diff(zeta, x)
        if sp.simplify(zp) == 0:
            return False
        q = self.turning_point.normal_form.potential
        phase_prime = sp.diff(self.phase_integral, x)
        if sp.simplify(phase_prime**2 - q) != 0:
            return False
        if self.canonical_family == "airy":
            expected_transform = sp.Pow(
                sp.Rational(3, 2) * self.phase_integral, sp.Rational(2, 3)
            )
        elif self.canonical_family == "weber":
            expected_transform = sp.sqrt(2 * self.phase_integral)
        else:
            return False
        if zeta != expected_transform and sp.simplify(zeta - expected_transform) != 0:
            return False
        zpp = sp.diff(zp, x)
        zppp = sp.diff(zpp, x)
        expected_residual = zppp / (2 * zp**3) - 3 * zpp**2 / (4 * zp**4)
        return (
            self.residual == expected_residual
            or sp.simplify(self.residual - expected_residual) == 0
        )


def liouville_normal_form(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> LiouvilleNormalForm:
    """Reduce a homogeneous second-order scalar equation to Liouville normal form."""

    op = _coerce_linear_operator(ode, function, variable)
    if op.order != 2:
        raise ValueError("Liouville normal form requires a second-order equation")
    if not op.is_homogeneous:
        raise ValueError("Liouville normal form requires a homogeneous equation")
    normalized = op.normalized()
    x = normalized.variable
    p = sp.cancel(normalized.coefficients[1])
    q = sp.cancel(normalized.coefficients[0])
    h = sp.cancel(-p / 2)
    potential = sp.cancel(sp.diff(p, x) / 2 + p**2 / 4 - q)
    gauge = sp.exp(sp.Integral(h, x))
    return LiouvilleNormalForm(normalized, p, q, potential, gauge, h)


def _resolved_polynomial_roots(
    polynomial: sp.Expr,
    variable: sp.Symbol,
) -> tuple[tuple[sp.Expr, int], bool]:
    """Return exact roots with multiplicity and whether all roots were resolved."""

    try:
        poly = sp.Poly(polynomial, variable)
    except sp.PolynomialError:
        return (), False
    if poly.is_zero:
        return (), False
    degree = int(poly.degree())
    try:
        roots = sp.roots(poly.as_expr(), variable, cubics=False, quartics=False)
    except SYMBOLIC_FAILURES:
        roots = {}
    result = [(sp.simplify(root), int(mult)) for root, mult in roots.items()]
    if sum(mult for _, mult in result) != degree:
        try:
            all_roots = poly.all_roots(radicals=False)
        except SYMBOLIC_FAILURES:
            all_roots = []
        if len(all_roots) == degree:
            counts: dict[sp.Expr, int] = {}
            for root in all_roots:
                root = sp.simplify(root)
                counts[root] = counts.get(root, 0) + 1
            result = list(counts.items())
    result.sort(key=lambda item: sp.default_sort_key(item[0]))
    return tuple(result), sum(mult for _, mult in result) == degree


def _turning_kind(multiplicity: int) -> TurningPointKind:
    if multiplicity == 1:
        return TurningPointKind.SIMPLE
    if multiplicity == 2:
        return TurningPointKind.DOUBLE
    return TurningPointKind.HIGHER


def analyze_turning_points(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> TurningPointAnalysis:
    """Find finite turning points as zeros of the Liouville normal-form potential."""

    normal = liouville_normal_form(ode, function, variable)
    x = normal.operator.variable
    numerator, denominator = sp.fraction(sp.cancel(normal.potential))
    roots, complete = _resolved_polynomial_roots(numerator, x)
    points: list[TurningPoint] = []
    for root, multiplicity in roots:
        den_value = sp.simplify(denominator.subs(x, root))
        if den_value == 0:
            continue
        h = sp.Symbol("_h")
        local = sp.cancel(normal.potential.subs(x, root + h))
        leading = sp.simplify(sp.limit(local / h**multiplicity, h, 0))
        points.append(
            TurningPoint(
                point=root,
                multiplicity=multiplicity,
                kind=_turning_kind(multiplicity),
                leading_coefficient=leading,
                normal_form=normal,
            )
        )
    limitation = (
        None
        if complete
        else "could not resolve every zero of the normal-form potential"
    )
    return TurningPointAnalysis(normal, tuple(points), complete, limitation)


def turning_points(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> tuple[TurningPoint, ...]:
    """Return the resolved finite turning points of a second-order equation."""

    return analyze_turning_points(ode, function, variable).points


def classify_turning_point(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr,
) -> TurningPoint:
    """Classify a specified finite point by the zero multiplicity of ``Q``."""

    normal = liouville_normal_form(ode, function, variable)
    x = normal.operator.variable
    h = sp.Symbol("_h")
    local = sp.cancel(normal.potential.subs(x, sp.sympify(point) + h))
    if sp.simplify(sp.limit(local, h, 0)) != 0:
        raise ValueError(f"point {point!s} is not a turning point")
    numerator, denominator = sp.fraction(local)
    try:
        poly = sp.Poly(numerator, h)
        powers = [monomial[0] for monomial, coeff in poly.terms() if coeff != 0]
        multiplicity = min(powers) if powers else 0
    except sp.PolynomialError:
        multiplicity = 0
    if multiplicity < 1 or sp.simplify(denominator.subs(h, 0)) == 0:
        raise ValueError(f"could not certify turning-point multiplicity at {point!s}")
    leading = sp.simplify(sp.limit(local / h**multiplicity, h, 0))
    return TurningPoint(
        sp.sympify(point), multiplicity, _turning_kind(multiplicity), leading, normal
    )


def wkb_expansion(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    order: int = 2,
    parameter: sp.Symbol | None = None,
) -> tuple[WKBExpansion, WKBExpansion]:
    """Return both formal WKB branches through Riccati coefficient ``order``.

    The normal-form equation is interpreted as ``epsilon**2 u'' = Q u``.
    Setting ``epsilon=1`` recovers the ordinary Liouville-Green ansatz, while
    retaining the symbol makes the asymptotic order bookkeeping explicit.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")
    normal = liouville_normal_form(ode, function, variable)
    if sp.simplify(normal.potential) == 0:
        raise ValueError("WKB expansion requires a nonzero normal-form potential")
    x = normal.operator.variable
    eps = parameter if parameter is not None else sp.Symbol("epsilon", positive=True)
    if eps == x:
        raise ValueError("WKB parameter must differ from the ODE variable")
    result: list[WKBExpansion] = []
    for branch in (1, -1):
        coefficients = [sp.simplify(branch * sp.sqrt(normal.potential))]
        for n in range(1, order + 1):
            convolution = sum(
                coefficients[j] * coefficients[n - j] for j in range(1, n)
            )
            value = sp.cancel(
                -(sp.diff(coefficients[n - 1], x) + convolution) / (2 * coefficients[0])
            )
            coefficients.append(value)
        series = sp.Add(*(eps**n * value for n, value in enumerate(coefficients)))
        exponent = sp.Integral(series, x) / eps
        u = sp.exp(exponent)
        result.append(
            WKBExpansion(
                normal,
                eps,
                branch,
                tuple(coefficients),
                series,
                u,
                normal.gauge * u,
            )
        )
    return result[0], result[1]


def _uniform_coordinate(
    turning_point: TurningPoint,
    family: str,
) -> tuple[sp.Expr, sp.Expr]:
    x = turning_point.normal_form.operator.variable
    q = turning_point.normal_form.potential
    x0 = turning_point.point
    local_power = (x - x0) ** turning_point.multiplicity
    quotient = sp.cancel(q / local_power)
    if x not in quotient.free_symbols:
        exponent = sp.Rational(turning_point.multiplicity, 2) + 1
        phase = sp.sqrt(quotient) * (x - x0) ** exponent / exponent
    else:
        t = sp.Dummy("t")
        integrand = sp.sqrt(q).subs(x, t)
        phase = sp.Integral(integrand, (t, x0, x))
    if family == "airy":
        zeta = sp.Pow(sp.Rational(3, 2) * phase, sp.Rational(2, 3))
        return zeta, phase
    if family == "weber":
        return sp.sqrt(2 * phase), phase
    raise ValueError(f"unknown uniform family {family!r}")


def _uniform_reduction(
    turning_point: TurningPoint,
    *,
    family: str,
    parameter: sp.Symbol | None,
    canonical_variable: sp.Symbol | None,
) -> UniformWKBReduction:
    x = turning_point.normal_form.operator.variable
    eps = parameter if parameter is not None else sp.Symbol("epsilon", positive=True)
    z = canonical_variable if canonical_variable is not None else sp.Symbol("zeta")
    zeta, phase = _uniform_coordinate(turning_point, family)
    zp = sp.diff(zeta, x)
    amplitude = zp ** sp.Rational(-1, 2)
    canonical = z if family == "airy" else z**2
    zpp = sp.diff(zp, x)
    zppp = sp.diff(zpp, x)
    residual = zppp / (2 * zp**3) - 3 * zpp**2 / (4 * zp**4)
    return UniformWKBReduction(
        turning_point,
        eps,
        z,
        zeta,
        phase,
        amplitude,
        family,
        canonical,
        residual,
    )


def airy_uniformization(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr,
    parameter: sp.Symbol | None = None,
    canonical_variable: sp.Symbol | None = None,
) -> UniformWKBReduction:
    """Construct the Liouville-Green Airy reduction at a simple turning point."""

    turning = classify_turning_point(ode, function, variable, point=point)
    if turning.kind is not TurningPointKind.SIMPLE:
        raise ValueError("Airy uniformization requires a simple turning point")
    return _uniform_reduction(
        turning,
        family="airy",
        parameter=parameter,
        canonical_variable=canonical_variable,
    )


def weber_uniformization(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr,
    parameter: sp.Symbol | None = None,
    canonical_variable: sp.Symbol | None = None,
) -> UniformWKBReduction:
    """Construct the degenerate Weber reduction at a double turning point.

    This function treats an isolated double zero, whose canonical leading
    potential is ``zeta**2``.  Uniformization of two distinct coalescing turning
    points with a nonzero Weber parameter requires parameter-dependent
    confluence analysis and is outside this routine's contract.
    """

    turning = classify_turning_point(ode, function, variable, point=point)
    if turning.kind is not TurningPointKind.DOUBLE:
        raise ValueError("Weber uniformization requires a double turning point")
    return _uniform_reduction(
        turning,
        family="weber",
        parameter=parameter,
        canonical_variable=canonical_variable,
    )
