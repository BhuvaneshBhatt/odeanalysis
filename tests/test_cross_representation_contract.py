"""Consistency contracts between scalar, system, and interchange views."""

from __future__ import annotations

import sympy as sp

from odeanalysis import (
    LinearDifferentialOperator,
    complete_formal_exponential_parts,
    formal_ode_data,
    levelt_structure,
)
from odeanalysis.system import companion_system


def test_expression_and_operator_routes_have_identical_singularity_data():
    from odeanalysis import analyze_ode_singularities

    x = sp.symbols("x")
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
    operator = LinearDifferentialOperator.from_ode(ode, y, x)

    from_expression = analyze_ode_singularities(ode, y, x)
    from_operator = analyze_ode_singularities(operator)

    assert from_expression.order == from_operator.order
    assert tuple(item.kind for item in from_expression.finite) == tuple(
        item.kind for item in from_operator.finite
    )
    assert from_expression.infinity.kind == from_operator.infinity.kind


def test_companion_system_dimension_matches_scalar_operator_order():
    x = sp.symbols("x")
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 3) + x * sp.diff(y(x), x) + y(x)
    operator = LinearDifferentialOperator.from_ode(ode, y, x)
    system = companion_system(operator)

    assert system.dimension == operator.order
    assert system.variable == operator.variable


def test_levelt_structure_and_interchange_data_agree_on_invariants():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    airy = sp.diff(y(x), x, 2) - x * y(x)

    structure = levelt_structure(airy, y, x, point=sp.oo, terms=4)
    data = formal_ode_data(airy, y, x, point=sp.oo, terms=4, include_stokes=False)

    assert data.ramification_index == structure.ramification_index
    assert data.operator_order == structure.basis.operator_order
    assert tuple(block.dimension for block in data.blocks) == tuple(
        block.dimension for block in structure.blocks
    )
    assert tuple(block.exponential_polynomial for block in data.blocks) == tuple(
        block.exponential_polynomial for block in structure.blocks
    )
    assert data.cover_monodromy == structure.monodromy.cover_matrix


def test_completed_exponential_parts_match_levelt_exponential_blocks():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    airy = sp.diff(y(x), x, 2) - x * y(x)

    parts = complete_formal_exponential_parts(airy, y, x, point=sp.oo)
    structure = levelt_structure(airy, y, x, point=sp.oo, terms=4)

    assert {sp.simplify(part.exponential_polynomial) for part in parts} == {
        sp.simplify(block.exponential_polynomial) for block in structure.blocks
    }
    assert (
        max(part.ramification_index for part in parts) == structure.ramification_index
    )
