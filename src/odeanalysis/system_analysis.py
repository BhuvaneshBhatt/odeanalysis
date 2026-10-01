"""Public local, formal, Stokes, and parameter analysis for linear systems."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp
from semialg import is_satisfiable, parametric_cad

from ._local import rational_valuation
from ._zero import ZeroStatus, exact_zero_status
from .block_decomposition import LeveltTurrittinReduction, levelt_turrittin_reduce
from .matrix_series import MatrixLaurentSeries
from .system import FirstOrderSystem


@dataclass(frozen=True)
class SystemResonance:
    source: sp.Expr
    target: sp.Expr
    difference: sp.Expr


@dataclass(frozen=True)
class SystemSingularityAnalysis:
    point: sp.Expr
    kind: str
    pole_order: int
    poincare_rank: int
    leading_matrix: sp.ImmutableMatrix
    leading_rank: int | None
    residue: sp.ImmutableMatrix | None
    exponents: tuple[sp.Expr, ...]
    resonances: tuple[SystemResonance, ...]
    local_system: FirstOrderSystem

    @property
    def regular_singular(self) -> bool:
        return self.kind == "regular_singular"

    @property
    def irregular(self) -> bool:
        return self.kind == "irregular"


def _local_system(system: FirstOrderSystem, point: sp.Expr) -> FirstOrderSystem:
    point = sp.sympify(point)
    t = sp.Dummy("t", positive=True)
    if point == sp.oo:
        return system.change_variable(t, 1 / t)
    return system.change_variable(t, point + t)


def _matrix_pole_data(system: FirstOrderSystem) -> tuple[int, sp.ImmutableMatrix]:
    t = system.variable
    vals = []
    for entry in system.matrix:
        status = exact_zero_status(entry)
        if status is ZeroStatus.ZERO:
            continue
        if status is ZeroStatus.UNKNOWN and (entry.free_symbols - {t}):
            raise NotImplementedError(
                "system singularity analysis cannot certify a local valuation "
                "for a parameter-dependent entry whose zero status is unresolved; "
                "stratify or specialize the parameters first"
            )
        # An expression depending only on the local coordinate may have unknown
        # SymPy ``is_zero`` status while still being a nonzero meromorphic
        # function (for example (t**2 + 2)/t**3).  Its valuation is safe to
        # compute directly.
        valuation = rational_valuation(entry, t, 0)
        if valuation is None:
            raise NotImplementedError(
                "system singularity analysis requires finite rational local valuations"
            )
        vals.append(valuation)
    minimum = min(vals, default=0)
    pole = max(0, -minimum)
    scale = t**pole
    leading = sp.Matrix(system.matrix).applyfunc(
        lambda e: sp.simplify(sp.limit(scale * e, t, 0))
    )
    return pole, sp.ImmutableMatrix(leading)


def _resonances(exponents: tuple[sp.Expr, ...]) -> tuple[SystemResonance, ...]:
    result: list[SystemResonance] = []
    for a, b in combinations(exponents, 2):
        d = sp.simplify(a - b)
        # Resonance requires a *nonzero* integral difference.  Syntactic
        # ``d != 0`` is unsound for parameterized integer symbols because it
        # is true even when the parameter may specialize to zero.
        if d.is_zero is False and d.is_integer is True:
            result.append(SystemResonance(a, b, d))
        elif d.is_zero is False and (-d).is_integer is True:
            result.append(SystemResonance(b, a, -d))
    return tuple(result)


def analyze_system_singularity(
    system: FirstOrderSystem, point: sp.Expr = 0
) -> SystemSingularityAnalysis:
    """Classify a finite point or infinity of ``Y' = A(x)Y+b(x)``.

    Exponents are residue eigenvalues for regular-singular systems.  Resonance
    records nonzero integral differences between residue eigenvalues.  For an
    irregular point, ``poincare_rank`` is the raw connection pole order minus
    one; formal gauge reduction may lower it.
    """
    local = _local_system(system, point)
    pole, leading = _matrix_pole_data(local)
    kind = "ordinary" if pole == 0 else "regular_singular" if pole == 1 else "irregular"
    residue = leading if pole == 1 else None
    exponents: tuple[sp.Expr, ...] = ()
    if residue is not None:
        roots = sp.Matrix(residue).eigenvals()
        exponents = tuple(root for root, mult in roots.items() for _ in range(mult))
        exponents = tuple(sorted(exponents, key=sp.default_sort_key))
    rank = leading.rank() if not leading.free_symbols else None
    return SystemSingularityAnalysis(
        sp.sympify(point),
        kind,
        pole,
        max(0, pole - 1),
        leading,
        rank,
        residue,
        exponents,
        _resonances(exponents),
        local,
    )


@dataclass(frozen=True)
class FormalReductionCertificate:
    """Evidence describing a bounded or adaptive formal reduction search."""

    verified: bool
    complete: bool
    adaptive: bool
    attempted_depths: tuple[int, ...]
    attempted_cover_indices: tuple[int | None, ...]
    final_depth: int
    final_cover_index: int | None
    ramification_index: int
    limitation: str | None


@dataclass(frozen=True)
class FormalSystemAnalysis:
    point: sp.Expr
    singularity: SystemSingularityAnalysis
    connection: MatrixLaurentSeries
    reduction: LeveltTurrittinReduction
    exponential_parts: tuple[sp.Expr, ...]
    certificate: FormalReductionCertificate

    @property
    def complete(self) -> bool:
        return self.reduction.complete

    @property
    def ramification_index(self) -> int:
        return self.reduction.ramification_index

    @property
    def limitation(self) -> str | None:
        return self.reduction.limitation

    def verify(self) -> bool:
        return self.reduction.verify()


def formal_system_analysis(
    system: FirstOrderSystem,
    point: sp.Expr = 0,
    *,
    max_power: int = 3,
    max_depth: int = 2,
    max_cover_index: int | None = None,
    adaptive: bool = False,
    max_adaptive_depth: int = 6,
    max_adaptive_cover_index: int | None = 12,
) -> FormalSystemAnalysis:
    """Run the exact bounded Moser/Levelt--Turrittin pipeline on a system."""
    singularity = analyze_system_singularity(system, point)
    connection = MatrixLaurentSeries.from_matrix(
        singularity.local_system.matrix, singularity.local_system.variable
    )
    if max_adaptive_depth < max_depth:
        raise ValueError("max_adaptive_depth must be at least max_depth")
    attempted_depths: list[int] = []
    attempted_covers: list[int | None] = []
    depth = max_depth
    cover = max_cover_index
    while True:
        attempted_depths.append(depth)
        attempted_covers.append(cover)
        reduction = levelt_turrittin_reduce(
            connection,
            max_power=max_power,
            max_depth=depth,
            max_cover_index=cover,
        )
        if reduction.complete or not adaptive or depth >= max_adaptive_depth:
            break
        depth = min(max_adaptive_depth, max(depth + 1, 2 * max(1, depth)))
        if max_adaptive_cover_index is not None:
            if cover is None:
                cover = min(4, max_adaptive_cover_index)
            elif cover < max_adaptive_cover_index:
                cover = min(max_adaptive_cover_index, max(cover + 1, 2 * cover))
    q_parts: list[sp.Expr] = []
    if reduction.complete:
        for block in reduction.final_diagonalization.blocks:
            q = sp.S.Zero
            dimension = block.rows
            for power, coefficient in block.terms:
                if power >= -1:
                    continue
                matrix = sp.Matrix(coefficient)
                scalar = sp.simplify(sp.trace(matrix) / dimension)
                if matrix == scalar * sp.eye(dimension):
                    q += scalar * block.variable ** (power + 1) / (power + 1)
            q_parts.append(sp.expand(q))
    certificate = FormalReductionCertificate(
        verified=reduction.verify(),
        complete=reduction.complete,
        adaptive=adaptive,
        attempted_depths=tuple(attempted_depths),
        attempted_cover_indices=tuple(attempted_covers),
        final_depth=depth,
        final_cover_index=cover,
        ramification_index=reduction.ramification_index,
        limitation=reduction.limitation,
    )
    return FormalSystemAnalysis(
        point, singularity, connection, reduction, tuple(q_parts), certificate
    )


@dataclass(frozen=True)
class SystemStokesPair:
    blocks: tuple[int, int]
    difference: sp.Expr
    equal_magnitude_rays: tuple[sp.Expr, ...]
    phase_alignment_rays: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class SystemStokesGeometry:
    point: sp.Expr
    exponential_parts: tuple[sp.Expr, ...]
    pairs: tuple[SystemStokesPair, ...]
    structural_only: bool = True


def _leading_monomial(q: sp.Expr, t: sp.Symbol) -> tuple[sp.Expr, int] | None:
    q = sp.expand(q)
    terms = []
    for term in sp.Add.make_args(q):
        c, p = term.as_coeff_exponent(t)
        if p.is_Integer and p < 0 and not c.has(t):
            terms.append((int(p), c))
    if not terms:
        return None
    p, c = min(terms, key=lambda z: z[0])
    return c, -p


def _rays(c: sp.Expr, k: int, phase: bool) -> tuple[sp.Expr, ...]:
    if c.is_real is not True and c.is_number is not True:
        return ()
    arg = sp.arg(c)
    offset = 0 if phase else sp.pi / 2
    return tuple(
        sp.simplify((arg - offset - m * sp.pi) / k % (2 * sp.pi)) for m in range(2 * k)
    )


def system_stokes_geometry(
    formal: FormalSystemAnalysis,
    *,
    exponential_parts: tuple[sp.Expr, ...] | None = None,
) -> SystemStokesGeometry:
    """Return structural Stokes rays between formal exponential blocks.

    ``exponential_parts`` may be supplied when the bounded reducer's block
    objects do not expose scalar exponential polynomials.  Analytic Stokes
    multipliers are intentionally outside this result contract.
    """
    if exponential_parts is None:
        exponential_parts = formal.exponential_parts
    t = formal.reduction.transformed_connection.variable
    pairs = []
    for i, j in combinations(range(len(exponential_parts)), 2):
        d = sp.expand(exponential_parts[i] - exponential_parts[j])
        lead = _leading_monomial(d, t)
        if lead is None:
            pairs.append(SystemStokesPair((i, j), d, (), ()))
            continue
        c, k = lead
        pairs.append(SystemStokesPair((i, j), d, _rays(c, k, False), _rays(c, k, True)))
    return SystemStokesGeometry(formal.point, tuple(exponential_parts), tuple(pairs))


@dataclass(frozen=True)
class SystemParameterStratum:
    condition: sp.Expr


@dataclass(frozen=True)
class ParameterizedSystemAnalysis:
    point: sp.Expr
    parameters: tuple[sp.Symbol, ...]
    transition_polynomials: tuple[sp.Expr, ...]
    rank_loci: tuple[sp.Expr, ...]
    collision_loci: tuple[sp.Expr, ...]
    resonance_loci: tuple[sp.Expr, ...]
    block_loci: tuple[sp.Expr, ...]
    stokes_loci: tuple[sp.Expr, ...]
    strata: tuple[SystemParameterStratum, ...]
    exhaustive: bool


def _parameter_factors(
    expr: sp.Expr, parameters: tuple[sp.Symbol, ...]
) -> list[sp.Expr]:
    expr = sp.factor(expr)
    if expr == 0:
        return []
    try:
        factors = sp.factor_list(expr)[1]
    except (sp.PolynomialError, TypeError, ValueError):
        return []
    return [sp.factor(f) for f, _ in factors if f.free_symbols & set(parameters)]


def system_parameter_analysis(
    system: FirstOrderSystem,
    parameters: tuple[sp.Symbol, ...],
    point: sp.Expr = 0,
    *,
    assumptions: sp.Expr | bool = True,
    max_resonance_order: int = 4,
    exponential_parts: tuple[sp.Expr, ...] = (),
) -> ParameterizedSystemAnalysis:
    """Stratify exact algebraic system transition loci with ``semialg``.

    The initial public contract detects leading-rank transitions and residue
    eigenvalue collisions.  More refined formal-block and Stokes loci can be
    added without changing the result shape.
    """
    assumptions = sp.sympify(assumptions)
    local = _local_system(system, point)
    t = local.variable
    structural_vals: list[int] = []
    for entry in local.matrix:
        if entry == 0:
            continue
        num, den = sp.fraction(sp.cancel(entry))
        pn, pd = sp.Poly(num, t), sp.Poly(den, t)
        nv = min(k for k in range(pn.degree() + 1) if pn.nth(k) != 0)
        dv = min(k for k in range(pd.degree() + 1) if pd.nth(k) != 0)
        structural_vals.append(nv - dv)
    minimum = min(structural_vals, default=0)
    pole = max(0, -minimum)
    leading = sp.ImmutableMatrix(
        sp.Matrix(local.matrix).applyfunc(
            lambda e: sp.cancel(sp.limit(t**pole * e, t, 0))
        )
    )
    rank_loci: list[sp.Expr] = []
    collision_loci: list[sp.Expr] = []
    resonance_loci: list[sp.Expr] = []
    block_loci: list[sp.Expr] = []
    stokes_loci: list[sp.Expr] = []
    # Rank transitions: all nonconstant minors can change the leading rank.
    for size in range(1, min(leading.shape) + 1):
        for rows in combinations(range(leading.rows), size):
            for cols in combinations(range(leading.cols), size):
                rank_loci.extend(
                    _parameter_factors(
                        sp.Matrix(leading).extract(rows, cols).det(), parameters
                    )
                )
    # Regular-singular eigenvalue collisions and, for 2x2 residues, exact
    # bounded integer-resonance hypersurfaces from the characteristic discriminant.
    if pole == 1:
        lam = sp.Symbol("_lambda")
        char_poly = sp.Poly(sp.Matrix(leading).charpoly(lam).as_expr(), lam)
        disc = sp.factor(sp.discriminant(char_poly.as_expr(), char_poly.gens[0]))
        collision_loci.extend(_parameter_factors(disc, parameters))
        block_loci.extend(_parameter_factors(disc, parameters))
        if leading.rows == 2:
            for n in range(1, max_resonance_order + 1):
                resonance_loci.extend(_parameter_factors(disc - n * n, parameters))
    # Optional formal exponential blocks expose algebraic degeneracy and phase
    # alignment transitions.  Complex coefficients must be represented with
    # explicit real symbols for these expressions to be semialgebraic.
    for qi, qj in combinations(exponential_parts, 2):
        diff = sp.expand(qi - qj)
        if point == sp.oo:
            diff = sp.cancel(diff.subs(system.variable, 1 / local.variable))
        else:
            diff = sp.cancel(
                diff.subs(system.variable, sp.sympify(point) + local.variable)
            )
        lead = _leading_monomial(diff, local.variable)
        if lead is None:
            continue
        coefficient, _ = lead
        re, im = sp.expand_complex(coefficient).as_real_imag()
        stokes_loci.extend(_parameter_factors(sp.expand(re**2 + im**2), parameters))
        # Phase transitions relative to the real axis occur when Im(c)=0 or Re(c)=0.
        stokes_loci.extend(_parameter_factors(re, parameters))
        stokes_loci.extend(_parameter_factors(im, parameters))
    rank_loci = list(dict.fromkeys(rank_loci))
    collision_loci = list(dict.fromkeys(collision_loci))
    resonance_loci = list(dict.fromkeys(resonance_loci))
    block_loci = list(dict.fromkeys(block_loci))
    stokes_loci = list(dict.fromkeys(stokes_loci))
    loci = list(
        dict.fromkeys(
            [*rank_loci, *collision_loci, *resonance_loci, *block_loci, *stokes_loci]
        )
    )
    marker = sp.Ne(sp.prod(loci), 0, evaluate=False) if loci else sp.S.true
    geometry = parametric_cad(
        marker, (), parameters=parameters, assumptions=assumptions, output="result"
    )
    strata = tuple(
        SystemParameterStratum(case.condition)
        for case in geometry.cases
        if is_satisfiable(sp.And(assumptions, case.condition), parameters)
    )
    return ParameterizedSystemAnalysis(
        sp.sympify(point),
        parameters,
        tuple(loci),
        tuple(rank_loci),
        tuple(collision_loci),
        tuple(resonance_loci),
        tuple(block_loci),
        tuple(stokes_loci),
        strata,
        geometry.status == "complete",
    )


@dataclass(frozen=True)
class SystemFormalTypeSignature:
    """Discrete formal invariants used to compare parameter strata."""

    singularity_kind: str
    leading_rank: int | None
    spectral_multiplicities: tuple[int, ...]
    resonance_orders: tuple[int, ...]
    ramification_index: int | None
    block_dimensions: tuple[int, ...]
    exponential_parts: tuple[sp.Expr, ...]
    stokes_ray_counts: tuple[int, ...]
    complete: bool


@dataclass(frozen=True)
class SystemFormalTypeStratum:
    condition: sp.Expr
    sample: tuple[tuple[sp.Symbol, sp.Expr], ...]
    signature: SystemFormalTypeSignature
    certified: bool


@dataclass(frozen=True)
class ParameterizedSystemFormalTypes:
    """Semialgebraic parameter cells annotated by verified formal signatures."""

    base: ParameterizedSystemAnalysis
    strata: tuple[SystemFormalTypeStratum, ...]
    exhaustive: bool


def _formal_type_signature(
    system: FirstOrderSystem,
    point: sp.Expr,
    *,
    max_power: int,
    max_depth: int,
    max_cover_index: int | None,
    max_resonance_order: int,
) -> SystemFormalTypeSignature:
    singularity = analyze_system_singularity(system, point)
    formal = formal_system_analysis(
        system,
        point,
        max_power=max_power,
        max_depth=max_depth,
        max_cover_index=max_cover_index,
        adaptive=True,
        max_adaptive_depth=max(max_depth, 6),
        max_adaptive_cover_index=max_cover_index if max_cover_index is not None else 12,
    )
    block_dimensions: tuple[int, ...] = ()
    if formal.reduction.formal_stages:
        block_dimensions = tuple(
            block.rows for block in formal.reduction.final_diagonalization.blocks
        )
    stokes = system_stokes_geometry(formal)
    return SystemFormalTypeSignature(
        singularity.kind,
        singularity.leading_rank,
        tuple(sorted(sp.Matrix(singularity.residue).eigenvals().values()))
        if singularity.residue is not None
        else (),
        tuple(
            sorted(
                abs(int(r.difference))
                for r in singularity.resonances
                if r.difference.is_Integer
                and abs(int(r.difference)) <= max_resonance_order
            )
        ),
        formal.ramification_index if formal.certificate.verified else None,
        block_dimensions,
        tuple(formal.exponential_parts),
        tuple(len(pair.equal_magnitude_rays) for pair in stokes.pairs),
        formal.complete and formal.certificate.verified,
    )


def system_formal_type_stratification(
    system: FirstOrderSystem,
    parameters: tuple[sp.Symbol, ...],
    point: sp.Expr = 0,
    *,
    assumptions: sp.Expr | bool = True,
    max_resonance_order: int = 4,
    exponential_parts: tuple[sp.Expr, ...] = (),
    max_power: int = 3,
    max_depth: int = 2,
    max_cover_index: int | None = None,
) -> ParameterizedSystemFormalTypes:
    """Annotate certified transition cells with exact representative formal types.

    Certification is conservative: a cell is certified only when the semialgebraic
    transition decomposition is exhaustive and the representative formal reduction
    independently verifies and completes.  The transition set covers leading-rank,
    residue collision/resonance and supplied exponential/Stokes degeneracy loci.
    """
    base = system_parameter_analysis(
        system,
        parameters,
        point,
        assumptions=assumptions,
        max_resonance_order=max_resonance_order,
        exponential_parts=exponential_parts,
    )
    # Recompute the CAD result to retain exact representative samples.
    marker = (
        sp.Ne(sp.prod(base.transition_polynomials), 0, evaluate=False)
        if base.transition_polynomials
        else sp.S.true
    )
    geometry = parametric_cad(
        marker,
        (),
        parameters=parameters,
        assumptions=sp.sympify(assumptions),
        output="result",
    )
    strata: list[SystemFormalTypeStratum] = []
    for case in geometry.cases:
        sample = dict(case.sample)
        specialized = FirstOrderSystem(
            system.variable,
            sp.ImmutableMatrix(sp.Matrix(system.matrix).subs(sample)),
            sp.ImmutableMatrix(sp.Matrix(system.forcing).subs(sample))
            if system.forcing is not None
            else None,
            system.ramification_index,
        )
        signature = _formal_type_signature(
            specialized,
            point,
            max_power=max_power,
            max_depth=max_depth,
            max_cover_index=max_cover_index,
            max_resonance_order=max_resonance_order,
        )
        # The initial transition vocabulary is complete enough to certify
        # bounded discrete formal type for ordinary/regular-singular 2x2
        # families.  Irregular formal-block/ramification transitions require
        # additional transition polynomials and remain representative-only.
        cell_certified = (
            base.exhaustive
            and signature.complete
            and system.dimension <= 2
            and signature.singularity_kind in {"ordinary", "regular_singular"}
        )
        strata.append(
            SystemFormalTypeStratum(
                case.condition,
                tuple(
                    sorted(
                        sample.items(), key=lambda item: sp.default_sort_key(item[0])
                    )
                ),
                signature,
                cell_certified,
            )
        )
    return ParameterizedSystemFormalTypes(
        base, tuple(strata), base.exhaustive and all(s.certified for s in strata)
    )
