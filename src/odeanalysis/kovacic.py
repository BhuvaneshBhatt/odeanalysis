"""Kovacic Liouvillian analysis for rational second-order equations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import factorial

import sympy as sp

from .factorization import FirstOrderFactorization, factor_differential_operator
from .operator import LinearDifferentialOperator, _coerce_linear_operator


class KovacicOutcome(Enum):
    """Outcome of the Kovacic decision procedure."""

    CASE_1 = "case_1"
    CASE_2 = "case_2"
    CASE_3 = "case_3"
    NO_LIOUVILLIAN_SOLUTION = "no_liouvillian_solution"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class KovacicCase2Certificate:
    """Polynomial and Riccati evidence for Kovacic Case 2."""

    degree: int
    theta: sp.Expr
    polynomial: sp.Expr
    logarithmic_derivatives: tuple[sp.Expr, ...]

    def verify(self, r: sp.Expr, x: sp.Symbol) -> bool:
        """Replay the Case-2 auxiliary equation and Riccati identities."""

        p = self.polynomial
        theta = self.theta
        auxiliary = (
            sp.diff(p, x, 3)
            + 3 * theta * sp.diff(p, x, 2)
            + (3 * theta**2 + 3 * sp.diff(theta, x) - 4 * r) * sp.diff(p, x)
            + (
                sp.diff(theta, x, 2)
                + 3 * theta * sp.diff(theta, x)
                + theta**3
                - 4 * r * theta
                - 2 * sp.diff(r, x)
            )
            * p
        )
        if sp.simplify(auxiliary) != 0:
            return False
        return bool(self.logarithmic_derivatives) and all(
            sp.simplify(sp.diff(w, x) + w**2 - r) == 0
            for w in self.logarithmic_derivatives
        )


@dataclass(frozen=True)
class KovacicCase3Certificate:
    """Polynomial recurrence evidence for Kovacic Case 3."""

    n: int
    degree: int
    theta: sp.Expr
    pole_polynomial: sp.Expr
    polynomial: sp.Expr
    recurrence: tuple[sp.Expr, ...]
    algebraic_log_derivative: sp.Expr

    def verify(self, r: sp.Expr, x: sp.Symbol) -> bool:
        """Replay the Case-3 recurrence and terminal polynomial identity."""

        n = self.n
        s = self.pole_polynomial
        theta = self.theta
        # Stored order is P_n, P_{n-1}, ..., P_-1.
        values = self.recurrence
        if len(values) != n + 2 or sp.simplify(values[0] + self.polynomial) != 0:
            return False
        p_by_index = {n - offset: value for offset, value in enumerate(values)}
        for i in range(n, -1, -1):
            pi = p_by_index[i]
            pnext = p_by_index.get(i + 1, sp.S.Zero)
            expected = (
                -s * sp.diff(pi, x)
                + ((n - i) * sp.diff(s, x) - s * theta) * pi
                - (n - i) * (i + 1) * s**2 * r * pnext
            )
            if sp.simplify(p_by_index[i - 1] - expected) != 0:
                return False
        if sp.simplify(p_by_index[-1]) != 0:
            return False
        omega = sp.Symbol("omega")
        expected_poly = sp.expand(
            sum(
                s**i * p_by_index[i] * omega**i / factorial(n - i) for i in range(n + 1)
            )
        )
        return sp.simplify(expected_poly - self.algebraic_log_derivative) == 0


@dataclass(frozen=True)
class KovacicAnalysis:
    """Certificate-oriented result of Kovacic's algorithm."""

    operator: LinearDifferentialOperator
    normal_form_potential: sp.Expr
    outcome: KovacicOutcome
    case: int | None
    factorizations: tuple[FirstOrderFactorization, ...]
    finite_pole_orders: tuple[tuple[sp.Expr, int], ...]
    infinity_order: int
    reason: str
    case2_certificates: tuple[KovacicCase2Certificate, ...] = ()
    case3_certificates: tuple[KovacicCase3Certificate, ...] = ()

    @property
    def is_liouvillian(self) -> bool | None:
        """Return the exact three-valued Liouvillian decision."""

        if self.outcome in {
            KovacicOutcome.CASE_1,
            KovacicOutcome.CASE_2,
            KovacicOutcome.CASE_3,
        }:
            return True
        if self.outcome is KovacicOutcome.NO_LIOUVILLIAN_SOLUTION:
            return False
        return None

    def verify(self) -> bool:
        """Replay the stored Kovacic certificate."""

        r = _normal_form_potential(self.operator)
        if sp.simplify(self.normal_form_potential - r) != 0:
            return False
        if self.outcome is KovacicOutcome.CASE_1:
            return bool(self.factorizations) and all(
                f.verify() for f in self.factorizations
            )
        if self.outcome is KovacicOutcome.CASE_2:
            return bool(self.case2_certificates) and all(
                certificate.verify(r, self.operator.variable)
                for certificate in self.case2_certificates
            )
        if self.outcome is KovacicOutcome.CASE_3:
            return bool(self.case3_certificates) and all(
                certificate.verify(r, self.operator.variable)
                for certificate in self.case3_certificates
            )
        return True


def _normal_form_potential(op: LinearDifferentialOperator) -> sp.Expr:
    normalized = op.normalized()
    x = op.variable
    p = normalized.coefficients[1]
    q = normalized.coefficients[0]
    return sp.cancel(p**2 / 4 + sp.diff(p, x) / 2 - q)


def _pole_data(
    r: sp.Expr, x: sp.Symbol
) -> tuple[tuple[tuple[sp.Expr, int], ...], int, sp.Expr]:
    numerator, denominator = sp.cancel(r).as_numer_denom()
    try:
        den_poly = sp.Poly(denominator, x, extension=True)
        num_poly = sp.Poly(numerator, x, extension=True)
    except sp.PolynomialError:
        return (), 0, sp.S.Zero
    roots = sp.roots(den_poly.as_expr(), x)
    finite = tuple(
        sorted(
            ((root, int(mult)) for root, mult in roots.items()),
            key=lambda item: sp.default_sort_key(item[0]),
        )
    )
    infinity_order = int(den_poly.degree() - num_poly.degree())
    infinity_b = (
        sp.cancel(num_poly.LC() / den_poly.LC()) if infinity_order == 2 else sp.S.Zero
    )
    return finite, infinity_order, infinity_b


def _integer_value(value: sp.Expr) -> int | None:
    simplified = sp.simplify(value)
    if simplified.is_integer is True and simplified.is_nonnegative is True:
        return int(simplified)
    return None


def _integer_set(values: list[sp.Expr]) -> tuple[sp.Expr, ...]:
    result = []
    for value in values:
        simplified = sp.simplify(value)
        if simplified.is_integer is True and simplified not in result:
            result.append(simplified)
    return tuple(result)


def _pole_b(r: sp.Expr, x: sp.Symbol, pole: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.limit((x - pole) ** 2 * r, x, pole))


def _monic_auxiliary_solution(
    expr_builder, degree: int, x: sp.Symbol
) -> sp.Expr | None:
    if degree == 0:
        return sp.S.One if sp.simplify(expr_builder(sp.S.One)) == 0 else None
    coeffs = sp.symbols(f"_k0:{degree}")
    polynomial = x**degree + sum(coeffs[i] * x**i for i in range(degree))
    expression = sp.cancel(expr_builder(polynomial))
    numerator = sp.together(expression).as_numer_denom()[0]
    try:
        equations = sp.Poly(sp.expand(numerator), x).all_coeffs()
    except sp.PolynomialError:
        return None
    solution_set = sp.linsolve(equations, coeffs)
    if solution_set is sp.EmptySet:
        return None
    for solution in solution_set:
        if any(value.free_symbols & set(coeffs) for value in solution):
            continue
        candidate = sp.expand(polynomial.subs(dict(zip(coeffs, solution, strict=True))))
        if sp.simplify(expr_builder(candidate)) == 0:
            return candidate
    return None


def _case2(
    r: sp.Expr,
    x: sp.Symbol,
    finite: tuple[tuple[sp.Expr, int], ...],
    infinity_order: int,
    infinity_b: sp.Expr,
) -> tuple[KovacicCase2Certificate, ...]:
    choices: list[tuple[sp.Expr, ...]] = []
    poles: list[sp.Expr] = []
    for pole, order in finite:
        poles.append(pole)
        if order == 1:
            choices.append((sp.Integer(4),))
        elif order == 2:
            root = sp.sqrt(1 + 4 * _pole_b(r, x, pole))
            choices.append(_integer_set([2, 2 + 2 * root, 2 - 2 * root]))
        else:
            choices.append((sp.Integer(order),))
    if any(not values for values in choices):
        return ()
    if infinity_order > 2:
        infinity_choices = (sp.Integer(0), sp.Integer(2), sp.Integer(4))
    elif infinity_order == 2:
        root = sp.sqrt(1 + 4 * infinity_b)
        infinity_choices = _integer_set([2, 2 + 2 * root, 2 - 2 * root])
    else:
        infinity_choices = (sp.Integer(infinity_order),)
    certificates = []
    for finite_values in product(*choices):
        for e_inf in infinity_choices:
            degree = _integer_value((e_inf - sum(finite_values)) / 2)
            if degree is None:
                continue
            theta = sp.cancel(
                sp.Rational(1, 2)
                * sum(
                    e / (x - pole) for e, pole in zip(finite_values, poles, strict=True)
                )
            )

            def auxiliary(p, theta=theta):
                return (
                    sp.diff(p, x, 3)
                    + 3 * theta * sp.diff(p, x, 2)
                    + (3 * theta**2 + 3 * sp.diff(theta, x) - 4 * r) * sp.diff(p, x)
                    + (
                        sp.diff(theta, x, 2)
                        + 3 * theta * sp.diff(theta, x)
                        + theta**3
                        - 4 * r * theta
                        - 2 * sp.diff(r, x)
                    )
                    * p
                )

            p = _monic_auxiliary_solution(auxiliary, degree, x)
            if p is None:
                continue
            phi = sp.cancel(theta + sp.diff(p, x) / p)
            constant = sp.cancel(sp.diff(phi, x) / 2 + phi**2 / 2 - r)
            discriminant = sp.cancel(phi**2 - 4 * constant)
            roots = tuple(
                sp.simplify((phi + sign * sp.sqrt(discriminant)) / 2)
                for sign in (1, -1)
            )
            valid = tuple(
                w for w in roots if sp.simplify(sp.diff(w, x) + w**2 - r) == 0
            )
            if valid:
                certificates.append(KovacicCase2Certificate(degree, theta, p, valid))
    return tuple(certificates)


def _case3_sets(
    n: int,
    r: sp.Expr,
    x: sp.Symbol,
    finite: tuple[tuple[sp.Expr, int], ...],
    infinity_order: int,
    infinity_b: sp.Expr,
):
    if any(order not in {1, 2} for _, order in finite) or infinity_order < 2:
        return None
    choices = []
    poles = []
    for pole, order in finite:
        poles.append(pole)
        if order == 1:
            choices.append((sp.Integer(12),))
        else:
            root = sp.sqrt(1 + 4 * _pole_b(r, x, pole))
            values = [
                6 + sp.Rational(12 * k, n) * root for k in range(-n // 2, n // 2 + 1)
            ]
            choices.append(_integer_set(values))
    root_inf = sp.sqrt(1 + 4 * infinity_b)
    infinity_choices = _integer_set(
        [6 + sp.Rational(12 * k, n) * root_inf for k in range(-n // 2, n // 2 + 1)]
    )
    if any(not values for values in choices) or not infinity_choices:
        return None
    return poles, choices, infinity_choices


def _case3(
    r: sp.Expr,
    x: sp.Symbol,
    finite: tuple[tuple[sp.Expr, int], ...],
    infinity_order: int,
    infinity_b: sp.Expr,
) -> tuple[KovacicCase3Certificate, ...]:
    certificates = []
    for n in (4, 6, 12):
        sets = _case3_sets(n, r, x, finite, infinity_order, infinity_b)
        if sets is None:
            continue
        poles, choices, infinity_choices = sets
        s = sp.prod(x - pole for pole in poles)
        for finite_values in product(*choices):
            for e_inf in infinity_choices:
                degree = _integer_value(
                    sp.Rational(n, 12) * (e_inf - sum(finite_values))
                )
                if degree is None:
                    continue
                theta = sp.cancel(
                    sp.Rational(n, 12)
                    * sum(
                        e / (x - pole)
                        for e, pole in zip(finite_values, poles, strict=True)
                    )
                )

                def terminal(polynomial, n=n, s=s, theta=theta):
                    by_index = {n + 1: sp.S.Zero, n: -polynomial}
                    for i in range(n, -1, -1):
                        by_index[i - 1] = sp.cancel(
                            -s * sp.diff(by_index[i], x)
                            + ((n - i) * sp.diff(s, x) - s * theta) * by_index[i]
                            - (n - i) * (i + 1) * s**2 * r * by_index[i + 1]
                        )
                    return by_index[-1]

                polynomial = _monic_auxiliary_solution(terminal, degree, x)
                if polynomial is None:
                    continue
                by_index = {n + 1: sp.S.Zero, n: -polynomial}
                for i in range(n, -1, -1):
                    by_index[i - 1] = sp.cancel(
                        -s * sp.diff(by_index[i], x)
                        + ((n - i) * sp.diff(s, x) - s * theta) * by_index[i]
                        - (n - i) * (i + 1) * s**2 * r * by_index[i + 1]
                    )
                if sp.simplify(by_index[-1]) != 0:
                    continue
                omega = sp.Symbol("omega")
                equation = sp.expand(
                    sum(
                        s**i * by_index[i] * omega**i / factorial(n - i)
                        for i in range(n + 1)
                    )
                )
                certificate = KovacicCase3Certificate(
                    n=n,
                    degree=degree,
                    theta=theta,
                    pole_polynomial=sp.expand(s),
                    polynomial=polynomial,
                    recurrence=tuple(by_index[i] for i in range(n, -2, -1)),
                    algebraic_log_derivative=equation,
                )
                if certificate.verify(r, x):
                    certificates.append(certificate)
        if certificates:
            return tuple(certificates)
    return ()


def kovacic_analysis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
) -> KovacicAnalysis:
    """Run Kovacic's Cases 1--3 for a rational homogeneous second-order ODE."""

    op = _coerce_linear_operator(ode, function, variable)
    if not op.is_homogeneous or op.order != 2:
        raise ValueError(
            "Kovacic analysis requires a homogeneous second-order operator"
        )
    normalized = op.normalized()
    x = normalized.variable
    if not all(
        sp.cancel(c).is_rational_function(x) for c in normalized.coefficients[:-1]
    ):
        raise ValueError("Kovacic analysis requires rational-function coefficients")
    potential = _normal_form_potential(op)
    finite, infinity_order, infinity_b = _pole_data(potential, x)
    factors = factor_differential_operator(op)
    if factors:
        return KovacicAnalysis(
            op,
            potential,
            KovacicOutcome.CASE_1,
            1,
            factors,
            finite,
            infinity_order,
            "a rational Riccati solution certifies Kovacic Case 1",
        )
    case2 = _case2(potential, x, finite, infinity_order, infinity_b)
    if case2:
        return KovacicAnalysis(
            op,
            potential,
            KovacicOutcome.CASE_2,
            2,
            (),
            finite,
            infinity_order,
            "a quadratic algebraic Riccati certificate proves Kovacic Case 2",
            case2_certificates=case2,
        )
    case3 = _case3(potential, x, finite, infinity_order, infinity_b)
    if case3:
        return KovacicAnalysis(
            op,
            potential,
            KovacicOutcome.CASE_3,
            3,
            (),
            finite,
            infinity_order,
            "the finite-group polynomial recurrence certifies Kovacic Case 3",
            case3_certificates=case3,
        )
    return KovacicAnalysis(
        op,
        potential,
        KovacicOutcome.NO_LIOUVILLIAN_SOLUTION,
        None,
        (),
        finite,
        infinity_order,
        "Kovacic Cases 1, 2, and 3 all failed",
    )
