import sympy as sp

from odeanalysis import (
    complete_formal_exponential_parts,
    formal_asymptotic_solutions,
)
from odeanalysis.formal import (
    differential_bell_polynomials,
    formal_amplitude_series,
    riccati_expression,
)


def test_differential_bell_polynomials_and_riccati_expression():
    x = sp.symbols("x")
    y = sp.Function("y")
    w = sp.Function("w")(x)
    bells = differential_bell_polynomials(w, x, 3)
    assert bells[0] == 1
    assert sp.expand(bells[1] - w) == 0
    assert sp.expand(bells[2] - (w**2 + sp.diff(w, x))) == 0
    assert sp.expand(bells[3] - (w**3 + 3 * w * sp.diff(w, x) + sp.diff(w, x, 2))) == 0

    ode = sp.diff(y(x), x, 2) - x * y(x)
    from odeanalysis import LinearDifferentialOperator

    op = LinearDifferentialOperator.from_ode(ode, y, x)
    assert sp.expand(riccati_expression(op, w) - (sp.diff(w, x) + w**2 - x)) == 0


def test_airy_complete_exponential_power_and_amplitude_series():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    parts = complete_formal_exponential_parts(ode, y, x, point=sp.oo)
    assert len(parts) == 2
    assert {sp.simplify(p.exponential_polynomial) for p in parts} == {
        sp.Rational(2, 3) * x ** sp.Rational(3, 2),
        -sp.Rational(2, 3) * x ** sp.Rational(3, 2),
    }
    assert all(p.algebraic_power == sp.Rational(1, 4) for p in parts)
    assert all(sp.simplify(p.algebraic_prefactor - x ** sp.Rational(-1, 4)) == 0 for p in parts)

    amplitudes = formal_amplitude_series(ode, y, x, point=sp.oo, terms=7)
    by_sign = {sp.signsimp(a.exponential_part.exponential_polynomial): a for a in amplitudes}
    growing = next(
        a
        for a in amplitudes
        if sp.simplify(
            a.exponential_part.exponential_polynomial - sp.Rational(2, 3) * x ** sp.Rational(3, 2)
        )
        == 0
    )
    decaying = next(
        a
        for a in amplitudes
        if sp.simplify(
            a.exponential_part.exponential_polynomial + sp.Rational(2, 3) * x ** sp.Rational(3, 2)
        )
        == 0
    )
    assert growing.coefficients == (
        1,
        0,
        0,
        sp.Rational(5, 48),
        0,
        0,
        sp.Rational(385, 4608),
    )
    assert decaying.coefficients == (
        1,
        0,
        0,
        -sp.Rational(5, 48),
        0,
        0,
        sp.Rational(385, 4608),
    )
    assert (
        sp.simplify(
            growing.series
            - (1 + sp.Rational(5, 48) / x ** sp.Rational(3, 2) + sp.Rational(385, 4608) / x**3)
        )
        == 0
    )
    assert by_sign  # keep both branches materialized


def test_rank_one_equation_refines_to_exact_exponential_power_solution():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**4 * sp.diff(y(x), x, 2) - y(x)

    parts = complete_formal_exponential_parts(ode, y, x, point=0)
    assert {sp.simplify(p.exponential_polynomial) for p in parts} == {1 / x, -1 / x}
    assert all(p.algebraic_power == 1 for p in parts)
    assert all(sp.simplify(p.algebraic_prefactor - x) == 0 for p in parts)

    amplitudes = formal_amplitude_series(ode, y, x, point=0, terms=6)
    assert all(a.coefficients == (1, 0, 0, 0, 0, 0) for a in amplitudes)
    assert all(a.residual == 0 for a in amplitudes)

    solutions = formal_asymptotic_solutions(ode, y, x, point=0, terms=4)
    expected = {x * sp.exp(1 / x), x * sp.exp(-1 / x)}
    assert {sp.simplify(s.expression) for s in solutions} == expected
    for solution in solutions:
        assert sp.simplify(x**4 * sp.diff(solution.expression, x, 2) - solution.expression) == 0


def test_refinement_recovers_lower_exponential_terms_and_power_prefactor():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    alpha = sp.Rational(7, 5)
    target_q = 1 / x**2 + 3 / x
    target_w = sp.diff(target_q, x) + alpha / x
    potential = sp.expand(sp.diff(target_w, x) + target_w**2)
    ode = sp.diff(y(x), x, 2) - potential * y(x)

    parts = complete_formal_exponential_parts(ode, y, x, point=0)
    target = next(p for p in parts if sp.simplify(p.exponential_polynomial - target_q) == 0)
    assert target.algebraic_power == alpha
    assert sp.simplify(target.algebraic_prefactor - x**alpha) == 0

    amplitude = next(
        a
        for a in formal_amplitude_series(ode, y, x, point=0, terms=5)
        if sp.simplify(a.exponential_part.exponential_polynomial - target_q) == 0
    )
    assert amplitude.coefficients == (1, 0, 0, 0, 0)
    assert amplitude.residual == 0


def test_partition_complete_bell_matches_sympy_partial_bells():
    from odeanalysis.bell import (
        complete_exponential_bell_polynomial,
        complete_exponential_bell_via_sympy,
    )

    xs = sp.symbols("x1:9")
    for n in range(9):
        direct = complete_exponential_bell_polynomial(n, xs)
        reference = complete_exponential_bell_via_sympy(n, xs)
        assert sp.expand(direct - reference) == 0

    assert complete_exponential_bell_polynomial(4, xs) == (
        xs[0] ** 4 + 6 * xs[0] ** 2 * xs[1] + 3 * xs[1] ** 2 + 4 * xs[0] * xs[2] + xs[3]
    )


def test_differential_bell_recurrence_matches_complete_bell_construction():
    from odeanalysis.bell import complete_exponential_bell_polynomial

    x = sp.symbols("x")
    w = sp.Function("w")(x)
    bells = differential_bell_polynomials(w, x, 7)
    for n, actual in enumerate(bells):
        arguments = tuple(sp.diff(w, x, j) for j in range(n))
        expected = complete_exponential_bell_polynomial(n, arguments)
        assert sp.expand(actual - expected) == 0


def test_sparse_laurent_differential_bell_recurrence_and_truncation():
    from odeanalysis.formal import differential_bell_laurent_series
    from odeanalysis.series import SparseLaurentSeries

    t = sp.symbols("t")
    a, b = sp.symbols("a b")
    w_expr = a / t**4 + b / t**2
    w = SparseLaurentSeries.from_expr(w_expr, t)

    sparse = differential_bell_laurent_series(w, 4, ramification_index=2)

    def derivative(expr):
        return sp.diff(expr, t) / (2 * t)

    symbolic = differential_bell_polynomials(w_expr, t, 4, derivative=derivative)
    for sparse_bell, symbolic_bell in zip(sparse, symbolic, strict=True):
        assert sp.expand(sparse_bell.to_expr() - symbolic_bell) == 0

    truncated = differential_bell_laurent_series(
        w,
        4,
        ramification_index=2,
        min_power=-10,
        max_power=-2,
    )
    for full, cut in zip(sparse[1:], truncated[1:], strict=True):
        expected = full.truncate(min_power=-10, max_power=-2)
        assert cut == expected


def test_sparse_laurent_basic_operations():
    from odeanalysis.series import SparseLaurentSeries

    t = sp.symbols("t")
    a = SparseLaurentSeries.from_expr(1 / t**2 + 2 + 3 * t, t)
    b = SparseLaurentSeries.from_expr(2 / t - t, t)

    assert sp.expand(a.add(b).to_expr() - (1 / t**2 + 2 / t + 2 + 2 * t)) == 0
    assert sp.expand(a.multiply(b).to_expr() - sp.expand(a.to_expr() * b.to_expr())) == 0
    assert (
        sp.expand(a.derivative(ramification_index=2).to_expr() - sp.diff(a.to_expr(), t) / (2 * t))
        == 0
    )
    assert a.truncate(min_power=-1, max_power=0).to_expr() == 2


def test_recursive_secondary_riccati_newton_puiseux_refinement():
    """A repeated leading root must split only after a secondary ramification."""

    from odeanalysis import formal_exponential_parts

    x = sp.symbols("x", positive=True)
    y = sp.Function("y")

    # Conjugating by exp(-1/(2*x**2)) reduces this equation to
    #     v'' - x**(-3) v = 0.
    # The original Newton edge therefore sees the repeated leading logarithmic
    # derivative x**(-3), while the two branches split only at x**(-3/2).
    w0 = x**-3
    ode = sp.diff(y(x), x, 2) - 2 * w0 * sp.diff(y(x), x) + (w0**2 - sp.diff(w0, x) - x**-3) * y(x)

    leading = formal_exponential_parts(ode, y, x, point=0)
    assert len(leading) == 1
    assert leading[0].characteristic_root == 1
    assert leading[0].multiplicity == 2
    assert leading[0].ramification_index == 1

    completed = complete_formal_exponential_parts(ode, y, x, point=0)
    assert len(completed) == 2
    assert all(part.multiplicity == 1 for part in completed)
    assert all(part.ramification_index == 2 for part in completed)
    assert all(part.algebraic_power == sp.Rational(3, 4) for part in completed)

    expected_q = {
        -sp.Rational(1, 2) / x**2 - 2 / sp.sqrt(x),
        -sp.Rational(1, 2) / x**2 + 2 / sp.sqrt(x),
    }
    assert {sp.simplify(part.exponential_polynomial) for part in completed} == expected_q

    for part in completed:
        # The first secondary edge has only the zero root c^2; retaining it is
        # essential, because the actual split occurs later at local power -3/2.
        assert part.refinement_steps[0].coefficient == 0
        assert part.refinement_steps[0].root_multiplicity == 2
        split = next(step for step in part.refinement_steps if step.introduces_ramification)
        assert split.local_power == -sp.Rational(3, 2)
        assert split.ramification_before == 1
        assert split.ramification_after == 2
        assert split.coefficient in (-1, 1)

    amplitudes = formal_amplitude_series(ode, y, x, point=0, terms=4)
    assert len(amplitudes) == 2
    assert {amplitude.coefficients[1] for amplitude in amplitudes} == {
        -sp.Rational(3, 16),
        sp.Rational(3, 16),
    }
