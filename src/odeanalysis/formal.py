"""Riccati/Bell refinement and formal WKB amplitude series.

For a scalar operator ``L = sum_j a_j(h) D_h**j`` and a nonzero formal
solution ``y``, put ``w = D_h(log(y))``.  Then

``D_h**j(y) / y = B_j(w, w', ..., w**(j-1))``,

where the complete differential Bell polynomials satisfy
``B_0 = 1`` and ``B_{j+1} = D_h(B_j) + w B_j``.  This turns the linear ODE
into its Riccati equation ``sum_j a_j B_j = 0``.

A Newton edge supplies the first term of ``w``.  In a ramified coordinate
``h = t**r`` we recursively cancel the Riccati equation through the
``h**(-1)`` term.  Integrating the terms below ``h**(-1)`` gives the complete
finite exponential polynomial for that branch; the ``h**(-1)`` coefficient
is the algebraic power prefactor.  The remaining conjugated equation then
produces a formal amplitude series in ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_FAILURES
from .irregular import FormalExponentialPart, formal_exponential_parts
from .newton import (
    LocalizedOperator,
    local_order_and_leading_coefficient,
    localize_operator,
)
from .operator import LinearDifferentialOperator
from .series import SparseLaurentSeries


class FormalRefinementError(NotImplementedError):
    """Raised when a formal branch requires a refinement outside the supported representation."""


@dataclass(frozen=True)
class LogDerivativeCoefficient:
    """One coefficient ``c*t**power`` in a ramified logarithmic derivative."""

    power: int
    coefficient: sp.Expr


@dataclass(frozen=True)
class RiccatiRefinementStep:
    """One Newton--Puiseux step in a formal Riccati branch.

    ``local_power`` is the exponent of the correction in the local coordinate
    ``h`` (so a correction is ``coefficient*h**local_power``).  A zero
    coefficient records a zero characteristic root: that branch survives the
    current secondary Newton edge and must be tested at a higher power.
    ``ramification_before`` and ``ramification_after`` expose any new
    uniformizing cover introduced by a nonintegral correction exponent.
    """

    local_power: sp.Rational
    coefficient: sp.Expr
    characteristic_polynomial: sp.Expr
    root_multiplicity: int
    ramification_before: int
    ramification_after: int

    @property
    def introduces_ramification(self) -> bool:
        return self.ramification_after > self.ramification_before


@dataclass(frozen=True)
class CompleteFormalExponentialPart:
    """A Newton exponential branch refined through its algebraic prefactor.

    ``local_exponential_polynomial`` contains every negative-power term of the
    integrated logarithmic derivative.  ``algebraic_power`` is the coefficient
    of ``log(h)`` and hence gives the factor ``h**algebraic_power``.
    """

    leading_part: FormalExponentialPart
    local_coordinate: sp.Symbol
    local_parameter: sp.Symbol
    ramification_index: int
    logarithmic_derivative: sp.Expr
    logarithmic_derivative_parameter: sp.Expr
    coefficients: tuple[LogDerivativeCoefficient, ...]
    local_exponential_polynomial: sp.Expr
    exponential_polynomial: sp.Expr
    algebraic_power: sp.Expr
    local_algebraic_prefactor: sp.Expr
    algebraic_prefactor: sp.Expr
    multiplicity: int = 1
    refinement_steps: tuple[RiccatiRefinementStep, ...] = ()

    @property
    def point(self) -> sp.Expr:
        return self.leading_part.point

    @property
    def exponential_factor(self) -> sp.Expr:
        return sp.exp(self.exponential_polynomial)

    @property
    def local_exponential_factor(self) -> sp.Expr:
        return sp.exp(self.local_exponential_polynomial)

    @property
    def local_prefactor(self) -> sp.Expr:
        return self.local_exponential_factor * self.local_algebraic_prefactor

    @property
    def prefactor(self) -> sp.Expr:
        return self.exponential_factor * self.algebraic_prefactor


@dataclass(frozen=True)
class FormalAmplitudeSeries:
    """Formal amplitude after exponential and algebraic factors are removed.

    ``coefficients[k]`` multiplies ``t**k`` in the uniformizing parameter
    ``h=t**ramification_index``.  The normalization is ``coefficients[0] = 1``.
    """

    exponential_part: CompleteFormalExponentialPart
    coefficients: tuple[sp.Expr, ...]
    local_parameter: sp.Symbol
    local_series: sp.Expr
    series: sp.Expr
    residual: sp.Expr
    residual_valuation: sp.Rational | None


@dataclass(frozen=True)
class FormalAsymptoticSolution:
    """A complete finite exponential/power prefactor with a formal amplitude."""

    point: sp.Expr
    exponential_part: CompleteFormalExponentialPart
    amplitude: FormalAmplitudeSeries
    local_expression: sp.Expr
    expression: sp.Expr


@dataclass(frozen=True)
class _LogDerivativeState:
    parameter: sp.Symbol
    ramification: int
    expression: sp.Expr
    coefficients: tuple[LogDerivativeCoefficient, ...]
    multiplicity: int
    search_floor: sp.Rational
    refinement_steps: tuple[RiccatiRefinementStep, ...] = ()


@dataclass(frozen=True)
class _RiccatiSupportTerm:
    monomial: tuple[int, ...]
    valuation: sp.Rational
    leading_coefficient: sp.Expr
    degree: int
    derivative_weight: int

    @property
    def intercept(self) -> sp.Rational:
        raise AttributeError("intercept depends on the current ramification")


@dataclass(frozen=True)
class _SecondaryBalance:
    parameter_power: sp.Rational
    characteristic_variable: sp.Symbol
    characteristic_polynomial: sp.Expr
    roots: tuple[tuple[sp.Expr, int], ...]


def differential_bell_polynomials(
    logarithmic_derivative: sp.Expr,
    variable: sp.Symbol,
    order: int,
    *,
    derivative=None,
) -> tuple[sp.Expr, ...]:
    """Return ``B_0, ..., B_order`` using the incremental differential recurrence.

    These are the complete exponential Bell polynomials specialized to
    ``(w, w', ..., w**(n-1))``.  The recurrence is retained as the production
    symbolic path because a Riccati reduction naturally needs all orders in
    succession.  :func:`complete_exponential_bell_polynomial` provides the
    independent partition-based combinatorial construction.

    ``derivative`` may provide a custom derivation, which is used for ramified
    coordinates where ``D_h = (r*t**(r-1))**-1 D_t``.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")
    deriv = derivative or (lambda expr: sp.diff(expr, variable))
    bells: list[sp.Expr] = [sp.S.One]
    for _ in range(order):
        bells.append(sp.expand(deriv(bells[-1]) + logarithmic_derivative * bells[-1]))
    return tuple(bells)


def differential_bell_laurent_series(
    logarithmic_derivative: SparseLaurentSeries,
    order: int,
    *,
    ramification_index: int = 1,
    min_power: int | None = None,
    max_power: int | None = None,
) -> tuple[SparseLaurentSeries, ...]:
    """Return differential Bell polynomials as sparse truncated Laurent series.

    Powers are integral in the uniformizing variable ``t`` with local
    coordinate ``h=t**ramification_index``.  Truncation is applied after each
    differentiation and multiplication, so formal Riccati calculations can
    avoid materializing generic expanded Bell expressions outside the
    valuation window that is actually needed.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")
    if ramification_index < 1:
        raise ValueError("ramification_index must be positive")

    bells: list[SparseLaurentSeries] = [SparseLaurentSeries.one(logarithmic_derivative.variable)]
    for _ in range(order):
        derivative_part = bells[-1].derivative(
            ramification_index=ramification_index,
            min_power=min_power,
            max_power=max_power,
        )
        product_part = logarithmic_derivative.multiply(
            bells[-1],
            min_power=min_power,
            max_power=max_power,
        )
        bells.append(
            derivative_part.add(
                product_part,
                min_power=min_power,
                max_power=max_power,
            )
        )
    return tuple(bells)


def riccati_expression(
    operator: LinearDifferentialOperator,
    logarithmic_derivative: sp.Expr,
    *,
    derivative=None,
) -> sp.Expr:
    """Return the Bell-polynomial Riccati expression ``L[y]/y``."""

    if not operator.is_homogeneous:
        raise ValueError("Riccati refinement requires a homogeneous operator")
    bells = differential_bell_polynomials(
        logarithmic_derivative,
        operator.variable,
        operator.order,
        derivative=derivative,
    )
    return sp.expand(sum(operator.coefficients[j] * bells[j] for j in range(operator.order + 1)))


def _ramified_derivative(parameter: sp.Symbol, ramification: int):
    def derivative(expression: sp.Expr) -> sp.Expr:
        return sp.diff(expression, parameter) / (ramification * parameter ** (ramification - 1))

    derivative.parameter = parameter
    derivative.ramification_index = ramification
    return derivative


def _ramified_operator_data(
    localized: LocalizedOperator,
    parameter: sp.Symbol,
    ramification: int,
) -> tuple[tuple[sp.Expr, ...], callable]:
    h = localized.local_variable
    coefficients = tuple(
        sp.cancel(sp.together(c.subs(h, parameter**ramification)))
        for c in localized.operator.coefficients
    )
    return coefficients, _ramified_derivative(parameter, ramification)


def _solve_leading_correction(
    expression: sp.Expr,
    parameter: sp.Symbol,
    unknown: sp.Symbol,
) -> tuple[sp.Expr, ...]:
    """Solve the leading coefficient equation for a prescribed series term.

    The amplitude recurrence still uses a fixed integral parameter grid; the
    Newton--Puiseux Riccati refinement below uses the more general secondary
    balance machinery.
    """

    expression = sp.cancel(sp.together(sp.expand(expression)))
    if expression == 0:
        return (sp.S.Zero,)
    try:
        _, leading = local_order_and_leading_coefficient(expression, parameter)
    except (ValueError, NotImplementedError) as exc:
        raise FormalRefinementError(
            f"could not determine the next formal coefficient from {expression!s}"
        ) from exc
    leading = sp.factor(leading)
    if not leading.has(unknown):
        raise FormalRefinementError(
            "the current residual cannot be cancelled at the requested series power"
        )
    try:
        solutions = sp.solve(sp.Eq(leading, 0), unknown)
    except SYMBOLIC_FAILURES as exc:
        raise FormalRefinementError(
            f"could not solve formal coefficient equation {leading!s} = 0"
        ) from exc
    cleaned: list[sp.Expr] = []
    for solution in solutions:
        solution = sp.simplify(solution)
        if solution.has(unknown):
            continue
        if not any(sp.simplify(solution - old) == 0 for old in cleaned):
            cleaned.append(solution)
    if not cleaned:
        raise FormalRefinementError(
            f"formal coefficient equation {leading!s} = 0 has no resolved branch"
        )
    return tuple(sorted(cleaned, key=sp.default_sort_key))


def _differential_perturbation_polynomial(
    coefficients: tuple[sp.Expr, ...],
    base_log_derivative: sp.Expr,
    parameter: sp.Symbol,
    ramification: int,
) -> tuple[sp.Poly, tuple[sp.Symbol, ...]]:
    """Return the Riccati residual as a differential polynomial in a perturbation.

    If ``w = w0 + z``, symbols ``z0, z1, ...`` stand for
    ``z, D_h z, ...``.  The derivation is implemented algebraically, avoiding
    an internal undefined SymPy function, keeping the Newton support easy to
    inspect exactly.
    """

    order = len(coefficients) - 1
    if order < 1:
        raise FormalRefinementError("secondary Riccati refinement requires positive order")
    z = sp.symbols(f"_z0:{order}")

    def derivative(expression: sp.Expr) -> sp.Expr:
        result = sp.diff(expression, parameter) / (ramification * parameter ** (ramification - 1))
        for k in range(order - 1):
            result += sp.diff(expression, z[k]) * z[k + 1]
        return sp.expand(result)

    bells: list[sp.Expr] = [sp.S.One]
    w = base_log_derivative + z[0]
    for _ in range(order):
        bells.append(sp.expand(derivative(bells[-1]) + w * bells[-1]))
    residual = sp.expand(sum(coefficients[j] * bells[j] for j in range(order + 1)))
    try:
        return sp.Poly(residual, *z), z
    except sp.PolynomialError as exc:
        raise FormalRefinementError(
            "could not represent the translated Riccati equation as a differential polynomial"
        ) from exc


def _riccati_newton_support(
    coefficients: tuple[sp.Expr, ...],
    base_log_derivative: sp.Expr,
    parameter: sp.Symbol,
    ramification: int,
) -> tuple[_RiccatiSupportTerm, ...]:
    polynomial, _ = _differential_perturbation_polynomial(
        coefficients, base_log_derivative, parameter, ramification
    )
    support: list[_RiccatiSupportTerm] = []
    for monomial, coefficient in polynomial.terms():
        coefficient = sp.cancel(sp.together(coefficient))
        if coefficient == 0:
            continue
        try:
            valuation, leading = local_order_and_leading_coefficient(coefficient, parameter)
        except (ValueError, NotImplementedError) as exc:
            raise FormalRefinementError(
                "could not determine a coefficient valuation in the secondary "
                "Riccati Newton polygon"
            ) from exc
        support.append(
            _RiccatiSupportTerm(
                monomial=tuple(int(e) for e in monomial),
                valuation=sp.Rational(valuation),
                leading_coefficient=sp.simplify(leading),
                degree=sum(monomial),
                derivative_weight=sum(k * e for k, e in enumerate(monomial)),
            )
        )
    return tuple(support)


def _falling_derivative_factor(
    parameter_power: sp.Rational,
    ramification: int,
    derivative_order: int,
) -> sp.Expr:
    exponent = sp.Rational(parameter_power, ramification)
    result = sp.S.One
    for j in range(derivative_order):
        result *= exponent - j
    return sp.simplify(result)


def _characteristic_roots_with_multiplicity(
    polynomial: sp.Expr,
    variable: sp.Symbol,
) -> tuple[tuple[sp.Expr, int], ...]:
    """Resolve every characteristic root, including zero, exactly when possible."""

    polynomial = sp.factor(polynomial)
    try:
        poly = sp.Poly(polynomial, variable)
    except sp.PolynomialError as exc:
        raise FormalRefinementError(
            f"secondary characteristic expression is not polynomial in {variable!s}"
        ) from exc
    degree = int(poly.degree())
    if degree <= 0:
        return ()

    try:
        root_map = sp.roots(poly, cubics=False, quartics=False)
    except SYMBOLIC_FAILURES:
        root_map = {}
    roots = {sp.simplify(root): int(mult) for root, mult in root_map.items()}

    if sum(roots.values()) != degree:
        try:
            all_roots = poly.all_roots(radicals=False)
        except SYMBOLIC_FAILURES:
            all_roots = []
        if len(all_roots) == degree:
            roots = {}
            for root in all_roots:
                root = sp.simplify(root)
                roots[root] = roots.get(root, 0) + 1

    if sum(roots.values()) != degree:
        try:
            solved = sp.solve(sp.Eq(polynomial, 0), variable)
        except SYMBOLIC_FAILURES:
            solved = []
        if solved and len(solved) == degree:
            roots = {}
            for root in solved:
                root = sp.simplify(root)
                roots[root] = roots.get(root, 0) + 1

    if sum(roots.values()) != degree:
        raise FormalRefinementError(
            f"could not resolve all roots of secondary characteristic polynomial {polynomial!s}"
        )
    return tuple(sorted(roots.items(), key=lambda item: sp.default_sort_key(item[0])))


def _secondary_characteristic_at_power(
    support: tuple[_RiccatiSupportTerm, ...],
    parameter_power: sp.Rational,
    ramification: int,
    variable: sp.Symbol,
) -> sp.Expr:
    active: list[tuple[_RiccatiSupportTerm, sp.Expr, sp.Rational]] = []
    for item in support:
        derivative_factor = sp.S.One
        for order, exponent in enumerate(item.monomial):
            if exponent:
                derivative_factor *= (
                    _falling_derivative_factor(parameter_power, ramification, order) ** exponent
                )
        derivative_factor = sp.simplify(derivative_factor)
        if derivative_factor == 0:
            continue
        intercept = item.valuation - ramification * item.derivative_weight
        value = sp.simplify(intercept + item.degree * parameter_power)
        active.append((item, derivative_factor, sp.Rational(value)))

    if not active:
        return sp.S.Zero
    minimum = min(value for _, _, value in active)
    characteristic = sp.expand(
        sum(
            item.leading_coefficient * derivative_factor * variable**item.degree
            for item, derivative_factor, value in active
            if value == minimum
        )
    )
    return sp.factor(characteristic)


def _next_secondary_balance(
    coefficients: tuple[sp.Expr, ...],
    state: _LogDerivativeState,
) -> _SecondaryBalance | None:
    """Find the next Newton--Puiseux correction of a translated Riccati branch."""

    support = _riccati_newton_support(
        coefficients,
        state.expression,
        state.parameter,
        state.ramification,
    )
    # If the translated equation has no constant term, the current logarithmic
    # derivative is already an exact formal solution; positive-degree support
    # describes perturbations of it, not further coefficients of this branch.
    if not any(item.degree == 0 for item in support):
        return None

    candidates: set[sp.Rational] = set()
    for index, left in enumerate(support):
        left_intercept = left.valuation - state.ramification * left.derivative_weight
        for right in support[index + 1 :]:
            if left.degree == right.degree:
                continue
            right_intercept = right.valuation - state.ramification * right.derivative_weight
            power = sp.simplify((right_intercept - left_intercept) / (left.degree - right.degree))
            if power.is_Rational is not True:
                continue
            power = sp.Rational(power)
            if state.search_floor < power <= -state.ramification:
                candidates.add(power)

    characteristic_variable = sp.Dummy("c")
    for power in sorted(candidates):
        characteristic = _secondary_characteristic_at_power(
            support, power, state.ramification, characteristic_variable
        )
        if not characteristic.has(characteristic_variable):
            continue
        roots = _characteristic_roots_with_multiplicity(characteristic, characteristic_variable)
        if roots:
            return _SecondaryBalance(
                parameter_power=power,
                characteristic_variable=characteristic_variable,
                characteristic_polynomial=characteristic,
                roots=roots,
            )
    return None


def _reramify_state(
    localized: LocalizedOperator,
    state: _LogDerivativeState,
    multiplier: int,
) -> tuple[_LogDerivativeState, tuple[sp.Expr, ...]]:
    if multiplier == 1:
        coefficients, _ = _ramified_operator_data(localized, state.parameter, state.ramification)
        return state, coefficients
    new_parameter = sp.Dummy("t", positive=True)
    new_ramification = state.ramification * multiplier
    new_expression = sp.expand(state.expression.subs(state.parameter, new_parameter**multiplier))
    new_coefficients_history = tuple(
        LogDerivativeCoefficient(item.power * multiplier, item.coefficient)
        for item in state.coefficients
    )
    reramified = _LogDerivativeState(
        parameter=new_parameter,
        ramification=new_ramification,
        expression=new_expression,
        coefficients=new_coefficients_history,
        multiplicity=state.multiplicity,
        search_floor=sp.Rational(state.search_floor * multiplier),
        refinement_steps=state.refinement_steps,
    )
    coefficients, _ = _ramified_operator_data(localized, new_parameter, new_ramification)
    return reramified, coefficients


def _log_derivative_to_local(
    expression: sp.Expr,
    parameter: sp.Symbol,
    local_coordinate: sp.Symbol,
    ramification: int,
) -> sp.Expr:
    result = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(expression)):
        coefficient, power = term.as_coeff_exponent(parameter)
        power = sp.sympify(power)
        if not power.is_Integer:
            raise FormalRefinementError("ramified logarithmic derivative has a noninteger t-power")
        result += coefficient * local_coordinate ** (sp.Rational(int(power), ramification))
    return sp.simplify(result)


def _integrated_exponential_and_power(
    coefficients: tuple[LogDerivativeCoefficient, ...],
    local_coordinate: sp.Symbol,
    ramification: int,
) -> tuple[sp.Expr, sp.Expr]:
    q = sp.S.Zero
    alpha = sp.S.Zero
    for item in coefficients:
        m = item.power
        c = item.coefficient
        if c == 0:
            continue
        if m < -ramification:
            h_power = sp.Rational(m, ramification) + 1
            q += c * local_coordinate**h_power / h_power
        elif m == -ramification:
            alpha += c
    return sp.simplify(q), sp.simplify(alpha)


def _refine_one_leading_part(
    localized: LocalizedOperator,
    leading: FormalExponentialPart,
    *,
    max_branches: int,
) -> tuple[CompleteFormalExponentialPart, ...]:
    h = localized.local_variable
    initial_ramification = int(leading.ramification_index)
    parameter = sp.Dummy("t", positive=True)

    rho = sp.Rational(leading.edge.slope)
    first_power_expr = -initial_ramification * (rho + 1)
    if not first_power_expr.is_Integer:
        raise FormalRefinementError(
            "Newton ramification did not integralize the leading Riccati power"
        )
    first_power = int(first_power_expr)
    initial = sp.simplify(leading.characteristic_root * parameter**first_power)
    pending: list[_LogDerivativeState] = [
        _LogDerivativeState(
            parameter=parameter,
            ramification=initial_ramification,
            expression=initial,
            coefficients=(LogDerivativeCoefficient(first_power, leading.characteristic_root),),
            multiplicity=int(leading.multiplicity),
            search_floor=sp.Rational(first_power),
        )
    ]
    finished: list[_LogDerivativeState] = []

    while pending:
        state = pending.pop()
        if state.search_floor >= -state.ramification:
            finished.append(state)
            continue

        coefficients, _ = _ramified_operator_data(localized, state.parameter, state.ramification)
        balance = _next_secondary_balance(coefficients, state)
        if balance is None:
            finished.append(state)
            continue

        power = balance.parameter_power
        zero_multiplicity = 0
        nonzero_roots: list[tuple[sp.Expr, int]] = []
        for root, multiplicity in balance.roots:
            if sp.simplify(root) == 0:
                zero_multiplicity += multiplicity
            else:
                nonzero_roots.append((root, multiplicity))

        # A zero characteristic root represents unresolved branches whose next
        # nonzero correction occurs at a strictly higher power.  Keep the same
        # translated Riccati equation but advance its Newton search floor.
        if zero_multiplicity:
            pending.append(
                _LogDerivativeState(
                    parameter=state.parameter,
                    ramification=state.ramification,
                    expression=state.expression,
                    coefficients=state.coefficients,
                    multiplicity=zero_multiplicity,
                    search_floor=power,
                    refinement_steps=(
                        *state.refinement_steps,
                        RiccatiRefinementStep(
                            local_power=sp.Rational(power, state.ramification),
                            coefficient=sp.S.Zero,
                            characteristic_polynomial=balance.characteristic_polynomial,
                            root_multiplicity=zero_multiplicity,
                            ramification_before=state.ramification,
                            ramification_after=state.ramification,
                        ),
                    ),
                )
            )

        if nonzero_roots:
            multiplier = int(power.q)
            base_state, _ = _reramify_state(localized, state, multiplier)
            integral_power_expr = sp.simplify(power * multiplier)
            if integral_power_expr.is_Integer is not True:
                raise FormalRefinementError(
                    "secondary Newton ramification failed to integralize a correction power"
                )
            integral_power = int(integral_power_expr)
            for root, multiplicity in nonzero_roots:
                step = RiccatiRefinementStep(
                    local_power=sp.Rational(power, state.ramification),
                    coefficient=sp.simplify(root),
                    characteristic_polynomial=balance.characteristic_polynomial,
                    root_multiplicity=multiplicity,
                    ramification_before=state.ramification,
                    ramification_after=base_state.ramification,
                )
                pending.append(
                    _LogDerivativeState(
                        parameter=base_state.parameter,
                        ramification=base_state.ramification,
                        expression=sp.expand(
                            base_state.expression
                            + sp.simplify(root) * base_state.parameter**integral_power
                        ),
                        coefficients=(
                            *base_state.coefficients,
                            LogDerivativeCoefficient(integral_power, sp.simplify(root)),
                        ),
                        multiplicity=multiplicity,
                        search_floor=sp.Rational(integral_power),
                        refinement_steps=(*state.refinement_steps, step),
                    )
                )

        if len(pending) + len(finished) > max_branches:
            raise FormalRefinementError(
                f"formal Riccati refinement produced more than {max_branches} branches"
            )

    result: list[CompleteFormalExponentialPart] = []
    for state in finished:
        local_q, alpha = _integrated_exponential_and_power(
            state.coefficients, h, state.ramification
        )
        local_w = _log_derivative_to_local(state.expression, state.parameter, h, state.ramification)
        result.append(
            CompleteFormalExponentialPart(
                leading_part=leading,
                local_coordinate=h,
                local_parameter=state.parameter,
                ramification_index=state.ramification,
                logarithmic_derivative=local_w,
                logarithmic_derivative_parameter=state.expression,
                coefficients=state.coefficients,
                local_exponential_polynomial=local_q,
                exponential_polynomial=localized.to_original(local_q),
                algebraic_power=alpha,
                local_algebraic_prefactor=sp.simplify(h**alpha),
                algebraic_prefactor=localized.to_original(h**alpha),
                multiplicity=state.multiplicity,
                refinement_steps=state.refinement_steps,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                sp.default_sort_key(item.local_exponential_polynomial),
                sp.default_sort_key(item.algebraic_power),
            ),
        )
    )


def complete_formal_exponential_parts(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    max_branches: int = 64,
) -> tuple[CompleteFormalExponentialPart, ...]:
    """Refine Newton leading parts through the complete exponential polynomial.

    The refinement solves the Bell-polynomial Riccati equation in the Newton
    uniformizer through logarithmic-derivative order ``h**(-1)``.  Thus every
    term that integrates to a negative power is included in the exponential
    polynomial and the remaining ``h**(-1)`` coefficient is returned as the
    algebraic power.

    Degenerate characteristic roots are handled recursively: the translated
    Riccati differential polynomial is given its own Newton--Puiseux support,
    zero characteristic roots are propagated to later edges, and any rational
    correction exponent introduces the additional ramification required to
    integralize it.  Refinement stops after the ``h**(-1)`` coefficient, since
    later terms belong to the formal amplitude rather than the finite
    exponential/power prefactor.
    """

    localized = localize_operator(ode, function, variable, point=point)
    result: list[CompleteFormalExponentialPart] = []
    for leading in formal_exponential_parts(ode, function, variable, point=point):
        result.extend(_refine_one_leading_part(localized, leading, max_branches=max_branches))
    return tuple(result)


def _amplitude_conjugated_coefficients(
    localized: LocalizedOperator,
    completed: CompleteFormalExponentialPart,
) -> tuple[tuple[sp.Expr, ...], callable]:
    parameter = completed.local_parameter
    ramification = completed.ramification_index
    base_coefficients, derivative = _ramified_operator_data(localized, parameter, ramification)
    try:
        logarithmic_series = SparseLaurentSeries.from_expr(
            completed.logarithmic_derivative_parameter, parameter
        )
    except ValueError:
        bells = differential_bell_polynomials(
            completed.logarithmic_derivative_parameter,
            parameter,
            localized.operator.order,
            derivative=derivative,
        )
    else:
        bells = tuple(
            item.to_expr()
            for item in differential_bell_laurent_series(
                logarithmic_series,
                localized.operator.order,
                ramification_index=ramification,
            )
        )
    transformed: list[sp.Expr] = []
    n = localized.operator.order
    for k in range(n + 1):
        transformed.append(
            sp.expand(
                sum(
                    base_coefficients[j] * sp.binomial(j, k) * bells[j - k] for j in range(k, n + 1)
                )
            )
        )
    return tuple(transformed), derivative


def _amplitude_residual(
    transformed_coefficients: tuple[sp.Expr, ...],
    amplitude: sp.Expr,
    derivative,
) -> sp.Expr:
    derivatives = [amplitude]
    for _ in range(1, len(transformed_coefficients)):
        derivatives.append(sp.expand(derivative(derivatives[-1])))
    return sp.expand(sum(c * derivatives[k] for k, c in enumerate(transformed_coefficients)))


def formal_amplitude_series(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 8,
    max_branches: int = 64,
) -> tuple[FormalAmplitudeSeries, ...]:
    """Generate normalized formal amplitude series for all completed branches.

    ``terms`` is the number of coefficients in the ramified parameter,
    including the normalized constant coefficient.  Zeros are retained, so a
    series whose natural step is ``t**3`` will contain two explicit zero slots
    between successive nonzero coefficients.
    """

    if terms < 1:
        raise ValueError("terms must be at least 1")
    localized = localize_operator(ode, function, variable, point=point)
    completed_parts = complete_formal_exponential_parts(
        ode,
        function,
        variable,
        point=point,
        max_branches=max_branches,
    )
    result: list[FormalAmplitudeSeries] = []

    for branch_index, completed in enumerate(completed_parts):
        parameter = completed.local_parameter
        transformed, derivative = _amplitude_conjugated_coefficients(localized, completed)
        amplitude = sp.S.One
        values: list[sp.Expr] = [sp.S.One]
        for power in range(1, terms):
            unknown = sp.Dummy(f"a_{branch_index}_{power}")
            trial = amplitude + unknown * parameter**power
            residual = _amplitude_residual(transformed, trial, derivative)
            solutions = _solve_leading_correction(residual, parameter, unknown)
            if len(solutions) != 1:
                raise FormalRefinementError(
                    "amplitude recurrence branched; logarithmic or resonant amplitude "
                    "solutions require a separate formal basis construction"
                )
            value = solutions[0]
            amplitude = sp.expand(amplitude + value * parameter**power)
            values.append(value)

        residual = sp.cancel(sp.together(_amplitude_residual(transformed, amplitude, derivative)))
        if residual == 0:
            residual_valuation = None
        else:
            try:
                residual_valuation, _ = local_order_and_leading_coefficient(residual, parameter)
            except (ValueError, NotImplementedError):
                residual_valuation = None

        h = completed.local_coordinate
        local_series = sp.expand(
            sum(values[k] * h ** sp.Rational(k, completed.ramification_index) for k in range(terms))
        )
        x = localized.original_operator.variable
        if sp.sympify(point) == sp.oo:
            original_series = sp.simplify(local_series.subs(h, 1 / x))
        else:
            original_series = sp.simplify(local_series.subs(h, x - sp.sympify(point)))
        result.append(
            FormalAmplitudeSeries(
                exponential_part=completed,
                coefficients=tuple(values),
                local_parameter=parameter,
                local_series=local_series,
                series=original_series,
                residual=residual,
                residual_valuation=residual_valuation,
            )
        )
    return tuple(result)


def formal_asymptotic_solutions(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    *,
    point: sp.Expr = 0,
    terms: int = 8,
    max_branches: int = 64,
) -> tuple[FormalAsymptoticSolution, ...]:
    """Return formal WKB solutions through the requested amplitude order."""

    amplitudes = formal_amplitude_series(
        ode,
        function,
        variable,
        point=point,
        terms=terms,
        max_branches=max_branches,
    )
    result: list[FormalAsymptoticSolution] = []
    for amplitude in amplitudes:
        completed = amplitude.exponential_part
        local_expression = sp.exp(completed.local_exponential_polynomial)
        local_expression *= completed.local_algebraic_prefactor * amplitude.local_series
        result.append(
            FormalAsymptoticSolution(
                point=sp.sympify(point),
                exponential_part=completed,
                amplitude=amplitude,
                local_expression=analytic_powsimp(
                    sp.expand_power_base(local_expression, force=False)
                ),
                expression=sp.simplify(
                    sp.exp(completed.exponential_polynomial)
                    * completed.algebraic_prefactor
                    * amplitude.series
                ),
            )
        )
    return tuple(result)
