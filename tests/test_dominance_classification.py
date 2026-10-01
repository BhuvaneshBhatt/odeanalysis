import sympy as sp

from odeanalysis import classify_solution_dominance


def test_airy_sectorial_dominant_and_recessive_branches_alternate():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    airy = sp.diff(y(x), x, 2) - x * y(x)

    result = classify_solution_dominance(airy, y, x, point=sp.oo)
    assert result.complete
    assert result.common_ramification == 2
    assert result.branch_count == 2
    assert len(result.sectors) == 6
    for sector in result.sectors:
        assert set(sector.dominant_branches) | set(sector.recessive_branches) == {0, 1}
        assert set(sector.dominant_branches).isdisjoint(sector.recessive_branches)
        assert len(sector.dominance_levels) == 2


def test_regular_singular_equation_has_no_exponential_dominance_classification():
    x = sp.symbols("x")
    y = sp.Function("y")
    euler = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + y(x)

    result = classify_solution_dominance(euler, y, x, point=0)
    assert result.complete is False
    assert result.sectors == ()
    assert result.limitation == "no irregular exponential branches were found"
