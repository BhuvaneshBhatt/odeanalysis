import sympy as sp
from semialg import equivalent, implies

from odeanalysis import (
    FirstOrderSystem,
    ODESingularityKind,
    leading_rank_analysis,
    local_parameter_analysis,
    newton_loci,
    singularity_loci,
    stokes_formal_loci,
    turning_loci,
)


def _contains_polynomial(loci, expected):
    return any(
        sp.simplify(candidate - expected) == 0 or sp.simplify(candidate + expected) == 0
        for candidate in loci
    )


def _real_zero_polynomial(expression):
    expanded = sp.expand_complex(expression)
    return sp.factor(sp.re(expanded) ** 2 + sp.im(expanded) ** 2)


def test_local_strata_cover_domain_without_freezing_cad_cells():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * sp.diff(y(x), x) + y(x)
    result = local_parameter_analysis(ode, y, x)
    coverage = sp.Or(*(stratum.condition for stratum in result.strata))
    assert result.exhaustive
    assert equivalent(coverage, sp.true, (a,))
    regular = next(
        s for s in result.strata if s.singularity.kind is ODESingularityKind.REGULAR
    )
    irregular = next(
        s for s in result.strata if s.singularity.kind is ODESingularityKind.IRREGULAR
    )
    assert equivalent(regular.condition, sp.Eq(a, 0), (a,))
    assert equivalent(irregular.condition, sp.Ne(a, 0), (a,))


def test_resonance_regions_are_semialgebraically_equivalent():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + a * x * sp.diff(y(x), x) + b * y(x)
    result = local_parameter_analysis(ode, y, x, max_resonance_order=2)
    d = (a - 1) ** 2 - 4 * b
    first = next(item for item in result.resonance_strata if item.difference == 1)
    assert equivalent(first.condition, sp.Eq(d, 1), (a, b))


def test_newton_and_collision_loci_capture_parameter_degeneracies():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = (x**2 - a) * sp.diff(y(x), x, 2) + y(x)
    collisions = singularity_loci(ode, y, x)
    assert _contains_polynomial(collisions, a)
    assert isinstance(newton_loci(ode, y, x, point=0), tuple)


def test_leading_system_rank_uses_semialg_rank_stratification():
    x, a = sp.symbols("x a")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[a / x, 0], [0, 1 / x]]))
    result = leading_rank_analysis(system, 0, (a,))
    assert result.pole_order == 1
    assert implies(
        sp.Eq(a, 0),
        sp.Or(*(b.condition for b in result.strata.branches if b.value == 1)),
        (a,),
    )


def test_turning_and_stokes_formal_loci_are_ode_specific():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - (x**2 - a) * y(x)
    turning = turning_loci(ode, y, x)
    assert _contains_polynomial(turning, a)
    stokes = stokes_formal_loci((a * x**2, -a * x**2), x)
    assert _contains_polynomial(stokes, a)


def test_rank_stratification_detects_changes_when_determinant_is_identically_zero():
    from odeanalysis import FirstOrderSystem, leading_rank_analysis

    x, a = sp.symbols("x a")
    system = FirstOrderSystem(x, sp.Matrix([[1 / x, 0, 0], [0, a / x, 0], [0, 0, 0]]))
    result = leading_rank_analysis(system, 0, (a,))
    assert sp.det(result.leading_matrix) == 0
    text = str(result.strata)
    assert "1" in text and "2" in text


def test_projective_collision_loci_include_degree_loss_and_lower_degree_discriminant():
    x, a, b = sp.symbols("x a b")
    y = sp.Function("y")
    ode = (a * x**3 + b * x**2 + x + 1) * sp.diff(y(x), x, 2) + y(x)
    loci = singularity_loci(ode, y, x)
    assert _contains_polynomial(loci, a)
    assert _contains_polynomial(loci, b)
    # When a=0 the quadratic stratum has discriminant 1-4*b.
    assert _contains_polynomial(loci, 4 * b - 1)


def test_stokes_formal_and_ray_transition_contracts_are_distinct():
    from odeanalysis import (
        stokes_formal_loci,
        stokes_ray_loci,
    )

    t, a = sp.symbols("t a", real=True)
    parts = (t**2, sp.I * a * t**2, 0)
    formal = stokes_formal_loci(parts, t)
    rays = stokes_ray_loci(parts, t, (a,))
    assert _contains_polynomial(formal, a)
    # Formal complex-zero conditions are embedded as certified real polynomials;
    # phase alignment can contribute additional loci independently.
    for locus in formal:
        assert _contains_polynomial(rays, _real_zero_polynomial(locus))


def _newton_signature(polygon):
    return tuple(
        (edge.left.derivative_order, edge.right.derivative_order, edge.slope)
        for edge in polygon.edges
    )


def test_projective_strata_record_cumulative_infinity_multiplicity():
    from odeanalysis import singularity_projective_strata

    x, a, b = sp.symbols("x a b", real=True)
    y = sp.Function("y")
    ode = (a * x**3 + b * x**2 + x + 1) * sp.diff(y(x), x, 2) + y(x)
    strata = singularity_projective_strata(ode, y, x, parameters=(a, b))
    assert any(q.effective_degree == 3 and q.infinity_multiplicity == 0 for q in strata)
    assert any(q.effective_degree == 2 and q.infinity_multiplicity == 1 for q in strata)
    assert any(q.effective_degree == 1 and q.infinity_multiplicity == 2 for q in strata)


def test_newton_polygon_is_normalization_invariant_not_arbitrary_gauge_invariant():
    from odeanalysis import differential_newton_polygon

    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + y(x)
    base = differential_newton_polygon(ode, y, x)
    scaled = differential_newton_polygon((x + 1) * ode, y, x)
    assert _newton_signature(base) == _newton_signature(scaled)
    # y=exp(1/x)u changes D to D-1/x**2 and need not preserve the raw polygon.
    u = sp.Function("u")
    gprime = -(x**-2)
    gauged = (
        x**4
        * (
            sp.diff(u(x), x, 2)
            + 2 * gprime * sp.diff(u(x), x)
            + (sp.diff(gprime, x) + gprime**2) * u(x)
        )
        + x * (sp.diff(u(x), x) + gprime * u(x))
        + u(x)
    )
    assert _newton_signature(
        differential_newton_polygon(gauged, u, x)
    ) != _newton_signature(base)


def test_stokes_ray_geometry_requires_real_coordinates_and_respects_conjugation_sign():
    from odeanalysis import stokes_ray_loci

    t = sp.symbols("t")
    z = sp.symbols("z")
    try:
        stokes_ray_loci((z * t**2, t**2, 0), t, (z,))
    except ValueError as exc:
        assert "real parameter coordinates" in str(exc)
    else:
        raise AssertionError("unconstrained complex parameter must not be certified")
    ar, ai = sp.symbols("a_r a_i", real=True)
    original = set(
        map(str, stokes_ray_loci(((ar + sp.I * ai) * t**2, t**2, 0), t, (ar, ai)))
    )
    conjugate = set(
        map(str, stokes_ray_loci(((ar - sp.I * ai) * t**2, t**2, 0), t, (ar, ai)))
    )
    negated = set(
        map(str, stokes_ray_loci((-(ar + sp.I * ai) * t**2, -(t**2), 0), t, (ar, ai)))
    )
    assert original == conjugate
    assert original == negated


def test_multi_point_invariant_constancy_on_representative_certified_regions():
    from odeanalysis import (
        differential_newton_polygon,
        singularity_projective_strata,
        stokes_formal_loci,
        stokes_ray_loci,
    )

    x = sp.symbols("x")
    y = sp.Function("y")
    a, b = sp.symbols("a b", real=True)

    # Projective multiplicity: multiple independently chosen points in each region.
    poly_ode = (a * x**3 + b * x**2 + x + 1) * sp.diff(y(x), x, 2) + y(x)
    strata = singularity_projective_strata(poly_ode, y, x, parameters=(a, b))
    samples = (((2, 3), (5, -4)), ((0, 2), (0, -3)), ((0, 0), (0, 0)))
    expected = ((3, 0), (2, 1), (1, 2))
    for region_samples, invariant in zip(samples, expected, strict=True):
        observed = []
        for av, bv in region_samples:
            degree = sp.Poly(
                (a * x**3 + b * x**2 + x + 1).subs({a: av, b: bv}), x
            ).degree()
            observed.append((degree, 3 - degree))
        assert all(item == invariant for item in observed)
    assert {(q.effective_degree, q.infinity_multiplicity) for q in strata} >= set(
        expected
    )

    # Newton hull: two points in each zero/nonzero support region.
    newton_ode = x**4 * sp.diff(y(x), x, 2) + a * x * sp.diff(y(x), x) + y(x)
    for values in ((1, 3), (-2, -5)):
        sigs = [
            _newton_signature(differential_newton_polygon(newton_ode.subs(a, v), y, x))
            for v in values
        ]
        assert sigs[0] == sigs[1]
    zero_sig = _newton_signature(
        differential_newton_polygon(newton_ode.subs(a, 0), y, x)
    )
    assert zero_sig != _newton_signature(
        differential_newton_polygon(newton_ode.subs(a, 1), y, x)
    )

    # Rank: rank is constant at multiple points of each representative rank cell.
    matrices = [sp.Matrix([[1, 0, 0], [0, v, 0], [0, 0, 0]]) for v in (1, 2, -3)]
    assert {m.rank() for m in matrices} == {2}
    assert sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]]).rank() == 1

    # Formal type and Stokes-ray combinatorics: zero/nonzero patterns remain constant.
    t = sp.symbols("t")
    parts = ((a + sp.I * b) * t**2, t**2, 0)
    formal = stokes_formal_loci(parts, t)
    rays = stokes_ray_loci(parts, t, (a, b))

    def zero_pattern(loci, point):
        return tuple(sp.simplify(q.subs({a: point[0], b: point[1]})) == 0 for q in loci)

    for p, q in (((2, 1), (3, 2)), ((2, 0), (4, 0)), ((1, 1), (1, 2))):
        assert zero_pattern(formal, p) == zero_pattern(formal, q)
        assert zero_pattern(rays, p) == zero_pattern(rays, q)
