import sympy as sp

from odeanalysis import (
    complete_formal_exponential_parts,
    stokes_geometry,
)
from odeanalysis.stokes import stokes_geometry_from_exponential_parts


def _angle_set(values):
    return {sp.simplify(value / sp.pi) for value in values}


def test_airy_stokes_geometry_on_ramified_cover_and_original_plane():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    geometry = stokes_geometry(ode, y, x, point=sp.oo)

    assert geometry.common_ramification == 2
    assert len(geometry.exponential_parts) == 2
    assert len(geometry.pairs) == 1
    pair = geometry.pairs[0]
    assert pair.exponential_order == sp.Rational(3, 2)
    assert pair.leading_parameter_power == -3
    assert len(pair.equal_magnitude_rays) == 6
    assert len(pair.phase_alignment_rays) == 6

    # Six lifted rays project to the three classical Airy equal-magnitude
    # directions in the x-plane.
    assert _angle_set(ray.original_angle for ray in pair.equal_magnitude_rays) == {
        sp.Rational(1, 3),
        sp.S.One,
        sp.Rational(5, 3),
    }
    assert _angle_set(ray.original_angle for ray in pair.phase_alignment_rays) == {
        sp.S.Zero,
        sp.Rational(2, 3),
        sp.Rational(4, 3),
    }

    assert geometry.sector_geometry_complete is True
    assert len(geometry.sectors) == 6
    assert all(sector.width == sp.pi / 3 for sector in geometry.sectors)
    assert all(len(sector.dominance_levels) == 2 for sector in geometry.sectors)
    # Crossing successive equal-magnitude rays reverses which Airy exponential
    # dominates on the uniformizing cover.
    dominant = [sector.dominant_branches[0] for sector in geometry.sectors]
    assert all(dominant[k] != dominant[(k + 1) % 6] for k in range(6))


def test_rank_one_equation_has_two_local_stokes_rays_and_sector_dominance():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)

    geometry = stokes_geometry(ode, y, x, point=0)
    assert geometry.common_ramification == 1
    pair = geometry.pairs[0]
    assert pair.exponential_order == 1
    assert _angle_set(ray.local_angle for ray in pair.equal_magnitude_rays) == {
        sp.Rational(1, 2),
        sp.Rational(3, 2),
    }
    assert _angle_set(ray.local_angle for ray in pair.phase_alignment_rays) == {
        sp.S.Zero,
        sp.S.One,
    }
    assert len(geometry.sectors) == 2
    assert {sector.dominant_branches for sector in geometry.sectors} == {(0,), (1,)}


def test_completed_difference_not_individual_leading_terms_controls_pair_geometry():
    # Construct two completed parts from a real equation, then replace their Qs
    # by expressions with a cancelling highest term.  The pair geometry must
    # use the first surviving term of Q0-Q1.
    from dataclasses import replace

    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)
    original = complete_formal_exponential_parts(ode, y, x, point=0)
    h = original[0].local_coordinate

    q0 = h**-3 + 2 * h**-1
    q1 = h**-3 - 3 * h**-1
    parts = (
        replace(original[0], local_exponential_polynomial=q0),
        replace(original[1], local_exponential_polynomial=q1),
    )
    geometry = stokes_geometry_from_exponential_parts(parts, point=0)
    pair = geometry.pairs[0]
    assert sp.expand(pair.difference_local_exponential_polynomial - 5 / h) == 0
    assert pair.leading_parameter_power == -1
    assert pair.exponential_order == 1


def test_equal_exponential_parts_are_grouped_in_sector_dominance():
    from dataclasses import replace

    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)
    base = complete_formal_exponential_parts(ode, y, x, point=0)
    h = base[0].local_coordinate

    # Branches 0 and 1 are exponentially equivalent; branch 2 differs.
    parts = (
        replace(base[0], local_exponential_polynomial=1 / h),
        replace(base[0], local_exponential_polynomial=1 / h),
        replace(base[1], local_exponential_polynomial=-1 / h),
    )
    geometry = stokes_geometry_from_exponential_parts(parts, point=0)
    assert len(geometry.pairs) == 2
    assert geometry.sector_geometry_complete
    assert any((0, 1) in sector.dominance_levels for sector in geometry.sectors)


def test_symbolic_coefficient_keeps_pairwise_rays_but_marks_sectors_incomplete():
    from dataclasses import replace

    x = sp.symbols("x", positive=True)
    a = sp.symbols("a")
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)
    base = complete_formal_exponential_parts(ode, y, x, point=0)
    h = base[0].local_coordinate
    parts = (
        replace(base[0], local_exponential_polynomial=a / h),
        replace(base[1], local_exponential_polynomial=0),
    )

    geometry = stokes_geometry_from_exponential_parts(parts, point=0)
    assert len(geometry.pairs[0].equal_magnitude_rays) == 2
    assert geometry.sector_geometry_complete is False
    assert geometry.sectors == ()


def test_stokes_geometry_uses_secondary_completed_exponential_split():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    w0 = x**-3
    ode = (
        sp.diff(y(x), x, 2)
        - 2 * w0 * sp.diff(y(x), x)
        + (w0**2 - sp.diff(w0, x) - x**-3) * y(x)
    )

    geometry = stokes_geometry(ode, y, x, point=0)
    assert geometry.common_ramification == 2
    assert len(geometry.pairs) == 1
    pair = geometry.pairs[0]
    # The common -1/(2*x**2) exponential cancels; Stokes geometry is governed
    # by the recursively discovered secondary difference 4/sqrt(x).
    assert pair.exponential_order == sp.Rational(1, 2)
    assert sp.simplify(abs(pair.leading_coefficient) - 4) == 0


def test_stokes_geometry_validates_exact_projection_and_sector_partition():
    x = sp.symbols("x")
    y = sp.Function("y")
    geometry = stokes_geometry(sp.diff(y(x), x, 2) - x * y(x), y, x, point=sp.oo)
    geometry.validate()
    assert (
        sp.simplify(sum(sector.width for sector in geometry.sectors) - 2 * sp.pi) == 0
    )
    assert all(
        set(sector.dominance_order) == set(range(len(geometry.exponential_parts)))
        for sector in geometry.sectors
    )


def test_stokes_ray_projection_respects_ramification_and_infinity_orientation():
    x = sp.symbols("x")
    y = sp.Function("y")
    geometry = stokes_geometry(sp.diff(y(x), x, 2) - x * y(x), y, x, point=sp.oo)
    for ray in geometry.equal_magnitude_rays + geometry.phase_alignment_rays:
        local = sp.Mod(geometry.common_ramification * ray.cover_angle, 2 * sp.pi)
        original = sp.Mod(-local, 2 * sp.pi)
        assert sp.simplify(sp.Mod(ray.local_angle, 2 * sp.pi) - local) == 0
        assert sp.simplify(sp.Mod(ray.original_angle, 2 * sp.pi) - original) == 0


def test_stokes_connection_patterns_expose_only_active_ray_couplings():
    from odeanalysis.stokes import (
        StokesGeometryError,
        stokes_connection_patterns,
    )

    x = sp.symbols("x")
    y = sp.Function("y")
    geometry = stokes_geometry(sp.diff(y(x), x, 2) - x * y(x), y, x, point=sp.oo)

    patterns = stokes_connection_patterns(geometry)
    assert len(patterns) == len(geometry.sector_boundaries)
    assert all(pattern.active_pairs == ((0, 1),) for pattern in patterns)

    matrix = patterns[0].symbolic_matrix()
    patterns[0].validate_matrix(matrix)
    bad = sp.eye(2)
    bad[0, 0] = 2
    try:
        patterns[0].validate_matrix(bad)
    except StokesGeometryError:
        pass
    else:
        raise AssertionError("non-unit-diagonal Stokes factor was accepted")


def test_stokes_validator_rejects_inconsistent_global_rays_and_sector_metadata():
    from dataclasses import replace

    from odeanalysis.stokes import StokesGeometryError

    x = sp.symbols("x")
    y = sp.Function("y")
    geometry = stokes_geometry(sp.diff(y(x), x, 2) - x * y(x), y, x, point=sp.oo)

    missing_ray = replace(
        geometry, equal_magnitude_rays=geometry.equal_magnitude_rays[:-1]
    )
    try:
        missing_ray.validate()
    except StokesGeometryError:
        pass
    else:
        raise AssertionError("inconsistent global Stokes rays were accepted")

    first = replace(geometry.sectors[0], index=3)
    bad_sector = replace(geometry, sectors=(first, *geometry.sectors[1:]))
    try:
        bad_sector.validate()
    except StokesGeometryError:
        pass
    else:
        raise AssertionError("inconsistent Stokes sector metadata was accepted")
