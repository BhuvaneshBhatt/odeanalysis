"""Independent certification corpus for first-order system analysis."""

from dataclasses import dataclass

import pytest
import sympy as sp

from odeanalysis import (
    FirstOrderSystem,
    analyze_system_singularity,
    formal_system_analysis,
)


@dataclass(frozen=True)
class Case:
    name: str
    matrix: sp.ImmutableMatrix
    point: sp.Expr
    kind: str
    pole_order: int
    poincare_rank: int
    exponents: tuple[sp.Expr, ...] = ()
    leading_rank: int | None = None
    formal_complete: bool | None = None
    ramification: int | None = None


x = sp.symbols("x")
CASES = (
    Case(
        "ordinary-zero",
        sp.ImmutableMatrix.zeros(2),
        0,
        "ordinary",
        0,
        0,
        leading_rank=0,
    ),
    Case(
        "ordinary-constant",
        sp.ImmutableMatrix([[1, 1], [0, 2]]),
        0,
        "ordinary",
        0,
        0,
        leading_rank=2,
    ),
    Case(
        "fuchs-diagonal",
        sp.ImmutableMatrix.diag(0, 2) / x,
        0,
        "regular_singular",
        1,
        0,
        (0, 2),
        1,
    ),
    Case(
        "fuchs-jordan",
        sp.ImmutableMatrix([[1, 1], [0, 1]]) / x,
        0,
        "regular_singular",
        1,
        0,
        (1, 1),
        2,
    ),
    Case(
        "fuchs-fractional",
        sp.ImmutableMatrix.diag(sp.Rational(1, 3), sp.Rational(5, 4)) / x,
        0,
        "regular_singular",
        1,
        0,
        (sp.Rational(1, 3), sp.Rational(5, 4)),
        2,
    ),
    Case(
        "infinity-fuchs",
        sp.ImmutableMatrix.diag(1 / x, 2 / x),
        sp.oo,
        "regular_singular",
        1,
        0,
        (-2, -1),
        2,
    ),
    Case(
        "irregular-rank1",
        sp.ImmutableMatrix.diag(x**-2, -(x**-2)),
        0,
        "irregular",
        2,
        1,
        leading_rank=2,
        formal_complete=True,
        ramification=1,
    ),
    Case(
        "irregular-rank2",
        sp.ImmutableMatrix.diag(2 * x**-3, -(x**-3)),
        0,
        "irregular",
        3,
        2,
        leading_rank=2,
        formal_complete=True,
        ramification=1,
    ),
    Case(
        "irregular-scalar-block",
        sp.ImmutableMatrix([[x**-2, 1], [0, x**-2]]),
        0,
        "irregular",
        2,
        1,
        leading_rank=2,
    ),
    Case(
        "infinity-irregular",
        sp.ImmutableMatrix.diag(x, -x),
        sp.oo,
        "irregular",
        3,
        2,
        leading_rank=2,
        formal_complete=True,
        ramification=1,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_system_reference_corpus(case: Case):
    result = analyze_system_singularity(FirstOrderSystem(x, case.matrix), case.point)
    assert result.kind == case.kind
    assert result.pole_order == case.pole_order
    assert result.poincare_rank == case.poincare_rank
    assert result.exponents == case.exponents
    assert result.leading_rank == case.leading_rank
    if case.formal_complete is not None:
        formal = formal_system_analysis(FirstOrderSystem(x, case.matrix), case.point, adaptive=True)
        assert formal.verify()
        assert formal.complete is case.formal_complete
        assert formal.ramification_index == case.ramification


def test_resonance_corpus_distinguishes_repetition_from_nonzero_integer_shift():
    repeated = analyze_system_singularity(
        FirstOrderSystem(x, sp.ImmutableMatrix.diag(1 / x, 1 / x))
    )
    shifted = analyze_system_singularity(FirstOrderSystem(x, sp.ImmutableMatrix.diag(1 / x, 4 / x)))
    assert repeated.resonances == ()
    assert len(shifted.resonances) == 1
    assert abs(shifted.resonances[0].difference) == 3
