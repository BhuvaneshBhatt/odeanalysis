"""Parameter-aware local analysis of scalar linear ODEs."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from funcprops import normalize_assumptions
from semialg import implies, is_satisfiable, parametric_cad

from ._assumptions import assumption_substitutions
from .frobenius import FrobeniusAnalysis, frobenius_analysis
from .operator import LinearDifferentialOperator, _coerce_linear_operator
from .singularities import ODESingularity, ODESingularityKind, classify_ode_point


@dataclass(frozen=True)
class LocalAnalysisStratum:
    """One parameter region with a stable local ODE classification."""

    condition: sp.Expr
    singularity: ODESingularity
    frobenius: FrobeniusAnalysis | None = None


@dataclass(frozen=True)
class ResonanceStratum:
    """Finite resonance condition for a quadratic indicial family."""

    condition: sp.Expr
    difference: int | None
    resonant: bool | None


@dataclass(frozen=True)
class ParameterizedLocalAnalysis:
    """Finite stratification of parameter-dependent local behavior."""

    point: sp.Expr
    assumptions: sp.Expr
    strata: tuple[LocalAnalysisStratum, ...]
    exhaustive: bool
    resonance_conditions: tuple[sp.Expr, ...] = ()
    resonance_strata: tuple[ResonanceStratum, ...] = ()

    def select(self, assumptions: sp.Expr | bool = True) -> LocalAnalysisStratum | None:
        query = normalize_assumptions(sp.And(self.assumptions, assumptions))
        parameters = tuple(sorted(query.free_symbols, key=sp.default_sort_key))
        matches = [
            stratum for stratum in self.strata if implies(query, stratum.condition, parameters)
        ]
        return matches[0] if len(matches) == 1 else None


def _local_operator(
    operator: LinearDifferentialOperator, point: sp.Expr
) -> tuple[LinearDifferentialOperator, sp.Expr]:
    if point != sp.oo:
        return operator, point
    t = sp.Dummy("t", positive=True)
    u = sp.Function("_u")
    return operator.reciprocal_transform(u, t), sp.S.Zero


def _valuation_atoms(operator: LinearDifferentialOperator, point: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return parameter expressions whose vanishing can change pole orders."""
    op, local_point = _local_operator(operator, point)
    x = op.variable
    h = sp.Dummy("h")
    atoms: set[sp.Expr] = set()
    normalized = op.normalized().coefficients
    for j, coeff in enumerate(normalized[:-1]):
        try:
            local = sp.cancel(sp.together(coeff.subs(x, local_point + h)))
            num, den = sp.fraction(local)
            den_poly = sp.Poly(den, h)
            den_val = next((k for k in range(den_poly.degree() + 1) if den_poly.nth(k) != 0), 0)
            # A cancellation matters to ordinary/regular/irregular classification
            # only when the generic denominator order exceeds the Fuchs bound.
            if den_val <= op.order - j:
                continue
            for poly_expr in (num,):
                poly = sp.Poly(poly_expr, h)
                # Only coefficients before the first structurally nonzero term
                # can alter the local valuation.
                for k in range(poly.degree() + 1):
                    c = sp.factor(poly.nth(k))
                    if c == 0:
                        continue
                    params = c.free_symbols - {x, h}
                    if params:
                        atoms.add(c)
                    if not params:
                        break
        except (sp.PolynomialError, TypeError, ValueError):
            continue
    return tuple(sorted(atoms, key=sp.default_sort_key))


def _indicial_atoms(
    operator: LinearDifferentialOperator,
    point: sp.Expr,
    assumptions: sp.Expr,
) -> tuple[sp.Expr, ...]:
    op, local_point = _local_operator(operator, point)
    local = classify_ode_point(op, point=local_point, assumptions=assumptions)
    poly = local.indicial_polynomial
    if local.kind is not ODESingularityKind.REGULAR or poly is None or operator.order != 2:
        return ()
    r = sp.Symbol("r")
    try:
        p = sp.Poly(poly, r)
        disc = sp.factor(sp.discriminant(p.as_expr(), r))
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    return (disc,) if disc.free_symbols else ()


@dataclass(frozen=True)
class _ParameterCase:
    condition: sp.Expr
    signature: tuple[bool | None, ...]


def _transition_signature(
    premise: sp.Expr,
    transition_polynomials: tuple[sp.Expr, ...],
    parameters: tuple[sp.Symbol, ...],
) -> tuple[bool | None, ...]:
    """Return certified zero/nonzero facts for one semialgebraic cell."""
    signature: list[bool | None] = []
    for polynomial in transition_polynomials:
        zero = sp.Eq(polynomial, 0, evaluate=False)
        if implies(premise, zero, parameters):
            signature.append(True)
        elif implies(premise, sp.Ne(polynomial, 0, evaluate=False), parameters):
            signature.append(False)
        else:
            signature.append(None)
    return tuple(signature)


def _parameter_cases(
    transition_polynomials: tuple[sp.Expr, ...],
    assumptions: sp.Expr,
) -> tuple[tuple[_ParameterCase, ...], bool]:
    """Return ODE-signature-grouped semialgebraic cells."""
    parameters = tuple(
        sorted(
            set().union(
                *(poly.free_symbols for poly in transition_polynomials),
                assumptions.free_symbols,
            ),
            key=sp.default_sort_key,
        )
    )
    if not parameters:
        cases = (_ParameterCase(sp.S.true, ()),) if is_satisfiable(assumptions, ()) else ()
        return cases, True

    marker = (
        sp.Ne(sp.prod(transition_polynomials), 0, evaluate=False)
        if transition_polynomials
        else sp.S.true
    )
    geometry = parametric_cad(
        marker,
        (),
        parameters=parameters,
        assumptions=assumptions,
        output="result",
    )
    groups: dict[tuple[bool | None, ...], list[sp.Expr]] = {}
    for case in geometry.cases:
        premise = sp.And(assumptions, case.condition)
        if not is_satisfiable(premise, parameters):
            continue
        signature = _transition_signature(premise, transition_polynomials, parameters)
        groups.setdefault(signature, []).append(case.condition)
    cases = tuple(_ParameterCase(sp.Or(*cells), signature) for signature, cells in groups.items())
    # A complete semialg result already certifies coverage of the supplied
    # parameter domain; re-proving the disjunction with equivalent() is both
    # redundant and substantially more expensive on larger decompositions.
    return cases, geometry.status == "complete"


def _case_assumptions(
    case: _ParameterCase,
    assumptions: sp.Expr,
    transition_polynomials: tuple[sp.Expr, ...],
) -> sp.Expr:
    """Materialize the zero/nonzero facts already certified for a grouped cell."""
    facts: list[sp.Expr] = [assumptions]
    for polynomial, state in zip(transition_polynomials, case.signature, strict=True):
        if state is True:
            facts.append(sp.Eq(polynomial, 0, evaluate=False))
        elif state is False:
            facts.append(sp.Ne(polynomial, 0, evaluate=False))
    return normalize_assumptions(sp.And(*facts))


@dataclass(frozen=True)
class _LocalContext:
    operator: LinearDifferentialOperator
    point: sp.Expr
    display_point: sp.Expr

    @classmethod
    def build(cls, operator: LinearDifferentialOperator, point: sp.Expr) -> _LocalContext:
        local_operator, local_point = _local_operator(operator, point)
        return cls(local_operator, local_point, point)

    def classify(self, assumptions: sp.Expr) -> ODESingularity:
        local = classify_ode_point(self.operator, point=self.point, assumptions=assumptions)
        if self.display_point != sp.oo:
            return local
        return ODESingularity(
            point=sp.oo,
            kind=local.kind,
            order=local.order,
            normalized_coefficients=local.normalized_coefficients,
            pole_orders=local.pole_orders,
            indicial_polynomial=local.indicial_polynomial,
            indicial_roots=local.indicial_roots,
            transformed_equation=self.operator.expression,
        )

    def frobenius(self, assumptions: sp.Expr, terms: int) -> FrobeniusAnalysis | None:
        substitutions = assumption_substitutions(assumptions)
        operator = self.operator
        if substitutions:
            operator = LinearDifferentialOperator(
                operator.variable,
                operator.function,
                tuple(c.subs(substitutions) for c in operator.coefficients),
                operator.inhomogeneous.subs(substitutions),
            )
        try:
            return frobenius_analysis(
                operator, point=self.point, terms=terms, assumptions=assumptions
            )
        except (ValueError, NotImplementedError):
            return None


def _finite_resonance_strata(
    discriminant: sp.Expr | None,
    assumptions: sp.Expr,
    max_order: int,
) -> tuple[ResonanceStratum, ...]:
    if discriminant is None or not discriminant.free_symbols:
        return ()
    parameters = tuple(
        sorted(
            set().union(discriminant.free_symbols, assumptions.free_symbols),
            key=sp.default_sort_key,
        )
    )
    equations = tuple(sp.Eq(discriminant, n * n, evaluate=False) for n in range(1, max_order + 1))
    strata = [
        ResonanceStratum(sp.And(assumptions, equation), n, True)
        for n, equation in enumerate(equations, 1)
        if is_satisfiable(sp.And(assumptions, equation), parameters)
    ]
    if equations:
        complement = sp.And(
            assumptions,
            *(sp.Ne(discriminant, n * n, evaluate=False) for n in range(1, max_order + 1)),
        )
        if is_satisfiable(complement, parameters):
            strata.append(ResonanceStratum(complement, None, None))
    return tuple(strata)


def local_parameter_analysis(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None = None,
    variable: sp.Symbol | None = None,
    point: sp.Expr = 0,
    *,
    terms: int = 6,
    assumptions: sp.Expr | bool = True,
    max_resonance_order: int = 4,
) -> ParameterizedLocalAnalysis:
    """Classify local behavior on finite parameter strata.

    Strata are generated only from algebraic conditions that can change local
    coefficient valuations or the multiplicity of a quadratic indicial
    polynomial.  Unresolved transcendental/integer resonances remain explicit
    conditions rather than being guessed.
    """
    operator = _coerce_linear_operator(ode, function, variable)
    assumptions = normalize_assumptions(assumptions)
    if max_resonance_order < 0:
        raise ValueError("max_resonance_order must be nonnegative")
    point = sp.sympify(point)
    atoms = _valuation_atoms(operator, point)
    atoms += tuple(a for a in _indicial_atoms(operator, point, assumptions) if a not in atoms)
    # Resonance hypersurfaces are part of the same parameter geometry as
    # valuation and repeated-root transitions.
    op0, p0 = _local_operator(operator, point)
    local0 = classify_ode_point(op0, point=p0, assumptions=assumptions)
    discriminant = None
    if (
        local0.kind is ODESingularityKind.REGULAR
        and local0.indicial_polynomial is not None
        and operator.order == 2
    ):
        r = sp.Symbol("r")
        try:
            discriminant = sp.factor(
                sp.discriminant(sp.Poly(local0.indicial_polynomial, r).as_expr(), r)
            )
        except (sp.PolynomialError, TypeError, ValueError):
            discriminant = None
    transition_polynomials = tuple(dict.fromkeys(atoms))
    cases, geometry_certified = _parameter_cases(transition_polynomials, assumptions)
    strata: list[LocalAnalysisStratum] = []
    context = _LocalContext.build(operator, point)
    for case in cases:
        local_assumptions = _case_assumptions(case, assumptions, transition_polynomials)
        local = context.classify(local_assumptions)
        frob = (
            context.frobenius(local_assumptions, terms)
            if local.kind is ODESingularityKind.REGULAR
            else None
        )
        candidate = LocalAnalysisStratum(case.condition, local, frob)
        for index, existing in enumerate(strata):
            if (
                existing.singularity == candidate.singularity
                and existing.frobenius == candidate.frobenius
            ):
                strata[index] = LocalAnalysisStratum(
                    sp.Or(existing.condition, case.condition), local, frob
                )
                break
        else:
            strata.append(candidate)

    # Merge strata with identical local signatures only when no Frobenius
    # degeneracy distinction would be lost.
    resonance: list[sp.Expr] = []
    for stratum in strata:
        f = stratum.frobenius
        if f is None or len(f.root_multiplicities) != 2:
            continue
        a, b = (root for root, _ in f.root_multiplicities)
        resonance.append(sp.Contains(sp.simplify(a - b), sp.S.Integers))
    resonance_strata = _finite_resonance_strata(discriminant, assumptions, max_resonance_order)

    return ParameterizedLocalAnalysis(
        point=point,
        assumptions=assumptions,
        strata=tuple(strata),
        exhaustive=geometry_certified,
        resonance_conditions=tuple(dict.fromkeys(resonance)),
        resonance_strata=resonance_strata,
    )
