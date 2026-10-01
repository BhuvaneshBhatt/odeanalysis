import sympy as sp

from odeanalysis import (
    ODESingularityKind,
    classify_ode_point,
    formal_exponential_parts,
    wkb_ansatze,
)
from odeanalysis.irregular import irregular_singularity_invariants
from odeanalysis.newton import (
    DifferentialNewtonPolygon,
    differential_newton_polygon,
)


def test_rank_one_irregular_newton_polygon_and_exponential_parts():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)

    assert classify_ode_point(ode, y, x, 0).kind is ODESingularityKind.IRREGULAR
    polygon = differential_newton_polygon(ode, y, x, point=0)
    assert isinstance(polygon, DifferentialNewtonPolygon)
    assert tuple(
        (p.derivative_order, p.coefficient_valuation, p.height) for p in polygon.points
    ) == ((0, 0, 0), (2, 4, 2))
    assert len(polygon.edges) == 1
    assert polygon.edges[0].slope == 1
    assert polygon.katz_rank == 1
    assert polygon.irregularity == 2
    assert polygon.ramification_index == 1

    parts = formal_exponential_parts(ode, y, x, point=0)
    assert len(parts) == 2
    assert {sp.simplify(part.exponent) for part in parts} == {1 / x, -1 / x}
    assert all(
        sp.expand(part.characteristic_polynomial - (part.characteristic_variable**2 - 1)) == 0
        for part in parts
    )


def test_airy_infinity_has_half_integer_katz_rank_and_ramification_two():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    polygon = differential_newton_polygon(ode, y, x, point=sp.oo)
    assert len(polygon.irregular_edges) == 1
    assert polygon.irregular_edges[0].slope == sp.Rational(3, 2)
    assert polygon.katz_rank == sp.Rational(3, 2)
    assert polygon.irregularity == 3
    assert polygon.ramification_index == 2

    inv = irregular_singularity_invariants(ode, y, x, point=sp.oo)
    assert inv.katz_rank == sp.Rational(3, 2)
    assert inv.poincare_rank == 3
    assert inv.irregularity == 3
    assert inv.ramification_index == 2
    assert inv.euler_system_poincare_rank == 3

    parts = formal_exponential_parts(ode, y, x, point=sp.oo)
    expected = {
        sp.Rational(2, 3) * x ** sp.Rational(3, 2),
        -sp.Rational(2, 3) * x ** sp.Rational(3, 2),
    }
    assert {sp.simplify(part.exponent) for part in parts} == expected
    assert all(part.ramification_index == 2 for part in parts)


def test_regular_singular_equation_has_no_irregular_edges():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + y(x)
    polygon = differential_newton_polygon(ode, y, x, point=0)
    assert not polygon.is_irregular
    assert polygon.katz_rank == 0
    assert polygon.irregularity == 0
    assert polygon.ramification_index == 1
    assert formal_exponential_parts(ode, y, x, point=0) == ()


def test_collinear_terms_all_contribute_to_edge_characteristic_polynomial():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = y(x) + 3 * x**2 * sp.diff(y(x), x) + 2 * x**4 * sp.diff(y(x), x, 2)
    polygon = differential_newton_polygon(ode, y, x, point=0)
    edge = polygon.edges[0]
    lam = sp.symbols("lam")
    assert edge.slope == 1
    assert tuple(p.derivative_order for p in edge.points) == (0, 1, 2)
    assert sp.expand(edge.characteristic_polynomial(lam) - (1 + 3 * lam + 2 * lam**2)) == 0


def test_wkb_ansatz_includes_exact_exponential_gauge_transform():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)
    ansatze = wkb_ansatze(ode, y, x, point=0)
    assert len(ansatze) == 2
    for ansatz in ansatze:
        assert ansatz.expression.has(sp.exp)
        # The selected leading exponential cancels the zeroth-order x^-4 balance
        # after monic normalization, leaving a valid order-2 amplitude operator.
        assert ansatz.conjugated_operator.order == 2
        assert ansatz.conjugated_operator.is_homogeneous


def test_newton_polygon_is_invariant_under_scalar_operator_multiple():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode1 = x**4 * sp.diff(y(x), x, 2) - y(x)
    ode2 = x**7 * ode1
    p1 = differential_newton_polygon(ode1, y, x, point=0)
    p2 = differential_newton_polygon(ode2, y, x, point=0)
    assert tuple(edge.slope for edge in p1.edges) == tuple(edge.slope for edge in p2.edges)
    assert p1.katz_rank == p2.katz_rank
    assert p1.irregularity == p2.irregularity


def test_multiple_irregular_edges_contribute_weighted_irregularity():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    # Newton heights v(a_j)-j are 0, 1, 5 at derivative orders 0, 1, 3.
    # The lower edges therefore have slopes 1 (length 1) and 2 (length 2).
    ode = y(x) + x**2 * sp.diff(y(x), x) + x**8 * sp.diff(y(x), x, 3)
    polygon = differential_newton_polygon(ode, y, x, point=0)
    assert tuple(edge.slope for edge in polygon.edges) == (1, 2)
    assert tuple(edge.horizontal_length for edge in polygon.edges) == (1, 2)
    assert polygon.katz_rank == 2
    assert polygon.irregularity == 5
    assert polygon.ramification_index == 1


def test_first_class_slope_filtration_and_integral_poincare_rank():
    from odeanalysis import katz_rank, newton_slopes, poincare_rank, slope_filtration

    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    airy = sp.diff(y(x), x, 2) - x * y(x)

    polygon = differential_newton_polygon(airy, y, x, point=sp.oo)
    filtration = slope_filtration(polygon)
    assert polygon.slopes == (sp.Rational(3, 2), sp.Rational(3, 2))
    assert filtration.slopes == polygon.slopes
    assert filtration.rank == 2
    assert filtration.katz_rank == sp.Rational(3, 2)
    assert filtration.poincare_rank == 3
    assert filtration.irregularity == 3
    assert filtration.ramification_index == 2
    assert filtration.pieces[0].multiplicity == 2
    assert filtration.pieces[0].ramification_index == 2
    assert newton_slopes(airy, y, x, point=sp.oo) == polygon.slopes
    assert katz_rank(airy, y, x, point=sp.oo) == sp.Rational(3, 2)
    assert poincare_rank(airy, y, x, point=sp.oo) == 3


def test_regular_newton_filtration_has_only_zero_slopes():
    from odeanalysis import newton_slopes, poincare_rank, slope_filtration

    x = sp.symbols("x")
    y = sp.Function("y")
    euler = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + y(x)

    polygon = differential_newton_polygon(euler, y, x, point=0)
    filtration = slope_filtration(polygon)
    assert newton_slopes(euler, y, x, point=0) == (0, 0)
    assert filtration.pieces[0].slope == 0
    assert filtration.pieces[0].multiplicity == 2
    assert filtration.irregular_pieces == ()
    assert poincare_rank(euler, y, x, point=0) == 0
