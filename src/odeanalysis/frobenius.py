"""Arbitrary-order Frobenius analysis at regular singular points."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from funcprops import normalize_assumptions

from ._assumptions import assumption_substitutions, zero_status
from ._symbolic_errors import SYMBOLIC_FAILURES
from .diagnostics import ReductionDiagnostic
from .operator import LinearDifferentialOperator


@dataclass(frozen=True)
class FrobeniusResonance:
    """A pair of indicial roots separated by a positive integer."""

    larger_root: sp.Expr
    smaller_root: sp.Expr
    difference: int


@dataclass(frozen=True)
class FrobeniusBranch:
    """A formal Frobenius branch through a chosen indicial exponent."""

    exponent: sp.Expr
    multiplicity: int
    coefficients: tuple[sp.Expr, ...]
    series: sp.Expr
    resonant_orders: tuple[int, ...] = ()
    obstructed_orders: tuple[int, ...] = ()
    free_orders: tuple[int, ...] = ()

    @property
    def logarithm_may_be_required(self) -> bool:
        return self.multiplicity > 1 or bool(self.obstructed_orders)

    @property
    def logarithm_required(self) -> bool:
        """Whether the pure-power recurrence has a certified obstruction."""
        return bool(self.obstructed_orders)


@dataclass(frozen=True)
class FrobeniusAnalysis:
    """Formal data for a regular singular point of an arbitrary-order ODE."""

    point: sp.Expr
    operator: LinearDifferentialOperator
    indicial_polynomial: sp.Expr
    indicial_variable: sp.Symbol
    root_multiplicities: tuple[tuple[sp.Expr, int], ...]
    resonances: tuple[FrobeniusResonance, ...]
    branches: tuple[FrobeniusBranch, ...]
    terms: int
    diagnostics: tuple[ReductionDiagnostic, ...] = ()

    @property
    def roots(self) -> tuple[sp.Expr, ...]:
        return tuple(root for root, mult in self.root_multiplicities for _ in range(mult))

    @property
    def has_resonance(self) -> bool:
        return bool(self.resonances) or any(mult > 1 for _, mult in self.root_multiplicities)

    @property
    def logarithm_required(self) -> bool:
        return any(branch.logarithm_required for branch in self.branches)

    @property
    def logarithm_may_be_required(self) -> bool:
        return any(branch.logarithm_may_be_required for branch in self.branches)


def _regularized_coefficients(
    operator: LinearDifferentialOperator,
    point: sp.Expr,
) -> tuple[sp.Expr, ...]:
    """Return analytic ``b_j=h^(n-j) p_j`` for a monic regular-singular operator."""

    normalized = operator.normalized()
    x = operator.variable
    n = operator.order
    return tuple(
        sp.cancel(sp.together((x - point) ** (n - j) * normalized.coefficients[j]))
        for j in range(n + 1)
    )


def _taylor_coeff(expr: sp.Expr, variable: sp.Symbol, point: sp.Expr, k: int) -> sp.Expr:
    if k == 0:
        return sp.simplify(sp.limit(expr, variable, point))
    deriv = sp.diff(expr, variable, k)
    return sp.simplify(sp.limit(deriv, variable, point) / sp.factorial(k))


def _indicial_polynomial(
    b: tuple[sp.Expr, ...],
    variable: sp.Symbol,
    point: sp.Expr,
    r: sp.Symbol,
) -> sp.Expr:
    return sp.factor(
        sp.expand(
            sum(_taylor_coeff(bj, variable, point, 0) * sp.ff(r, j) for j, bj in enumerate(b))
        )
    )


def _root_data(poly: sp.Expr, r: sp.Symbol) -> tuple[tuple[sp.Expr, int], ...]:
    # Avoid forcing generic cubic/quartic radicals with symbolic parameters.
    # SymPy can still recover rational/factorable roots with radicals disabled.
    try:
        roots = sp.roots(poly, r, cubics=False, quartics=False)
    except SYMBOLIC_FAILURES:
        roots = {}
    if roots:
        return tuple(sorted(roots.items(), key=lambda item: sp.default_sort_key(item[0])))
    try:
        p = sp.Poly(poly, r)
    except SYMBOLIC_FAILURES:
        return ()
    if poly.free_symbols - {r}:
        return ()
    try:
        all_roots = p.all_roots()
    except SYMBOLIC_FAILURES:
        return ()
    counts: dict[sp.Expr, int] = {}
    for root in all_roots:
        counts[root] = counts.get(root, 0) + 1
    return tuple(sorted(counts.items(), key=lambda item: sp.default_sort_key(item[0])))


def _integer_difference(a: sp.Expr, b: sp.Expr) -> int | None:
    d = sp.simplify(a - b)
    if d.is_Integer and d.is_positive:
        return int(d)
    if d.is_number:
        try:
            val = int(d)
        except TypeError:
            return None
        if sp.simplify(d - val) == 0 and val > 0:
            return val
    return None


def _resonances(
    roots: tuple[tuple[sp.Expr, int], ...],
) -> tuple[FrobeniusResonance, ...]:
    result: list[FrobeniusResonance] = []
    distinct = [root for root, _ in roots]
    for a in distinct:
        for b in distinct:
            diff = _integer_difference(a, b)
            if diff is not None:
                result.append(FrobeniusResonance(a, b, diff))
    unique = {(x.larger_root, x.smaller_root, x.difference): x for x in result}
    return tuple(
        sorted(
            unique.values(),
            key=lambda x: (x.difference, sp.default_sort_key(x.smaller_root)),
        )
    )


def _branch(
    b: tuple[sp.Expr, ...],
    variable: sp.Symbol,
    point: sp.Expr,
    indicial: sp.Expr,
    r: sp.Symbol,
    root: sp.Expr,
    multiplicity: int,
    terms: int,
    assumptions: sp.Expr | bool = True,
) -> FrobeniusBranch:
    # b[j][q] is the q-th Taylor coefficient of h^(n-j) p_j.
    bcoeff = [tuple(_taylor_coeff(bj, variable, point, q) for q in range(terms)) for bj in b]
    coeffs: list[sp.Expr] = [sp.S.One]
    resonant: list[int] = []
    obstructed: list[int] = []
    free: list[int] = []

    for m in range(1, terms):
        numerator = sp.S.Zero
        for q in range(1, m + 1):
            prev = coeffs[m - q]
            inner = sum(bcoeff[j][q] * sp.ff(root + m - q, j) for j in range(len(b)))
            numerator += prev * inner
        numerator = sp.simplify(numerator)
        denominator = sp.simplify(indicial.subs(r, root + m))
        znum = zero_status(numerator, assumptions)
        if znum is True and denominator.is_zero is None and denominator.free_symbols:
            coeffs.append(sp.S.Zero)
            continue
        zden = zero_status(denominator, assumptions)
        if zden is None and znum is True:
            # Zero is always a valid normalized coefficient here.  A special
            # parameter stratum may make the diagonal factor vanish and add a
            # free coefficient, but it does not obstruct this branch.
            coeffs.append(sp.S.Zero)
            continue
        if zden is True:
            resonant.append(m)
            if znum is True:
                # A genuinely free coefficient belongs to the homogeneous solution
                # space.  Use a named symbol so downstream code can retain it.
                am = sp.Symbol(f"a{m}")
                coeffs.append(am)
                free.append(m)
            else:
                # A pure power Frobenius branch cannot satisfy this coefficient
                # equation; a logarithmic companion is generally required.
                coeffs.append(sp.S.Zero)
                obstructed.append(m)
            continue
        if zden is None:
            # Keep the formal quotient without declaring resonance.
            coeffs.append(sp.cancel(-numerator / denominator))
            continue
        coeffs.append(sp.cancel(-numerator / denominator))

    h = variable - point
    series = sp.expand(h**root * sum(coeffs[m] * h**m for m in range(len(coeffs))))
    return FrobeniusBranch(
        exponent=root,
        multiplicity=multiplicity,
        coefficients=tuple(coeffs),
        series=series,
        resonant_orders=tuple(resonant),
        obstructed_orders=tuple(obstructed),
        free_orders=tuple(free),
    )


def frobenius_analysis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    point: sp.Expr = 0,
    *,
    terms: int = 6,
    assumptions: sp.Expr | bool = True,
) -> FrobeniusAnalysis:
    """Analyze a regular singular point using the arbitrary-order Frobenius recurrence.

    The recurrence is derived from ``h^n L[y]=0`` with ``h=x-x0``.  If
    ``b_j(h)=h^(n-j) p_j(h)`` and ``y=h^r sum a_m h^m``, then the coefficient
    of ``h^(r+m)`` gives a recurrence whose diagonal factor is the indicial
    polynomial evaluated at ``r+m``.  Vanishing diagonal factors are retained
    explicitly as resonant/free/obstructed orders rather than divided away.
    """

    if terms < 1:
        raise ValueError("terms must be at least one")
    assumptions = normalize_assumptions(assumptions)
    if isinstance(ode, LinearDifferentialOperator):
        operator = ode
    else:
        if function is None or variable is None:
            raise TypeError("function and variable are required when ode is not an operator")
        operator = LinearDifferentialOperator.from_ode(ode, function, variable)
    if not operator.is_homogeneous:
        raise ValueError("Frobenius analysis requires a homogeneous equation")
    substitutions = assumption_substitutions(assumptions)
    if substitutions:
        operator = LinearDifferentialOperator(
            operator.variable,
            operator.function,
            tuple(sp.cancel(c.subs(substitutions)) for c in operator.coefficients),
            sp.cancel(operator.inhomogeneous.subs(substitutions)),
        )

    from .singularities import ODESingularityKind, classify_ode_point

    local = classify_ode_point(operator, point=point, assumptions=assumptions)
    if local.kind is not ODESingularityKind.REGULAR:
        raise ValueError("Frobenius analysis requires a regular singular point")

    r = sp.Symbol("r")
    b = _regularized_coefficients(operator, sp.sympify(point))
    indicial = _indicial_polynomial(b, operator.variable, sp.sympify(point), r)
    roots = _root_data(indicial, r)
    branches = tuple(
        _branch(
            b,
            operator.variable,
            sp.sympify(point),
            indicial,
            r,
            root,
            multiplicity,
            terms,
            assumptions,
        )
        for root, multiplicity in roots
    )
    diagnostics: list[ReductionDiagnostic] = []
    for branch in branches:
        for order in branch.obstructed_orders:
            diagnostics.append(
                ReductionDiagnostic(
                    stage="frobenius",
                    code="logarithmic-obstruction",
                    message=f"pure-power recurrence is obstructed at order {order}",
                    order=order,
                )
            )
        if branch.multiplicity > 1:
            diagnostics.append(
                ReductionDiagnostic(
                    stage="frobenius",
                    code="repeated-indicial-root",
                    message=f"indicial root has multiplicity {branch.multiplicity}",
                    rank=branch.multiplicity,
                )
            )

    return FrobeniusAnalysis(
        point=sp.sympify(point),
        operator=operator,
        indicial_polynomial=indicial,
        indicial_variable=r,
        root_multiplicities=roots,
        resonances=_resonances(roots),
        branches=branches,
        terms=terms,
        diagnostics=tuple(diagnostics),
    )
